from __future__ import annotations

import argparse
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloader import (
    MaiMaiDataset,
    collate_fn_stage2,
    compute_mel_stats,
)
from src.stage2_model import Stage2Config, Stage2Model
from src.token_validator import Stage2Validator
from src.tokenizer import (
    END_ONSET,
    EOS,
    ONSET,
    PAD,
    STAGE2_VOCAB_SIZE,
    ChartTokenizer,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Training config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Stage2TrainConfig:
    data_dir: str = "./dataset"
    stats_file: str = "./dataset/mel_stats.pt"
    checkpoint_dir: str = "./checkpoints/stage2"

    # ── Model ──────────────────────────────────────────────────────────
    d_model: int = 512
    nhead: int = 8
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    encoder_num_layers: int = 3

    # ── Training ───────────────────────────────────────────────────────
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    eta_min: float = 1e-5
    grad_clip: float = 1.0
    val_fraction: float = 0.1

    # ── Logging / checkpointing ────────────────────────────────────────
    log_interval: int = 50
    val_interval: int = 500
    save_every_epochs: int = 5

    # ── Device ─────────────────────────────────────────────────────────
    device: str = "cuda"

    # ── Sequence filtering ─────────────────────────────────────────────
    max_tokens: int = 8192

    # ── Misc ───────────────────────────────────────────────────────────
    num_workers: int = 4
    seed: int = 42
    resume: bool = False

    # ── Scheduled onset sampling (optional, requires Stage 1 checkpoint) ─
    stage1_checkpoint: Optional[str] = None
    onset_sample_start: float = 0.0  # fraction of predicted onsets at epoch 1
    onset_sample_end: float = 0.0  # fraction of predicted onsets at final epoch


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class _PauseController:
    def __init__(self) -> None:
        self.pause_requested = False
        self.signal_count = 0

    def handler(self, signum, frame) -> None:
        self.signal_count += 1
        if self.signal_count == 1:
            self.pause_requested = True
            print(
                "\nCtrl-C pressed: pausing after the current training step. "
                "Press Ctrl-C again to force quit without saving.",
                flush=True,
            )
        else:
            print("\nForced exit without saving.", flush=True)
            sys.exit(1)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _WorkerInitFn:
    def __init__(self, base_seed: int) -> None:
        self.base_seed = base_seed

    def __call__(self, worker_id: int) -> None:
        torch.manual_seed(self.base_seed + worker_id)


def _capture_rng_state() -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, torch.Tensor]) -> None:
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _build_dataloader(
    cfg: Stage2TrainConfig,
    mel_stats: Tuple[torch.Tensor, torch.Tensor],
    split: str,
    epoch: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    ds = MaiMaiDataset(
        cfg.data_dir,
        split=split,
        val_fraction=cfg.val_fraction,
        mel_stats=mel_stats,
        max_tokens=cfg.max_tokens,
    )
    generator = torch.Generator().manual_seed(cfg.seed + epoch) if shuffle else None
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_stage2,
        pin_memory=True,
        drop_last=shuffle,
        generator=generator,
        worker_init_fn=_WorkerInitFn(cfg.seed + epoch) if shuffle else None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Chunked loss computation
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_loss_chunked(
    model: nn.Module,
    dec_out: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, int]:
    B, L, _ = dec_out.shape
    total_loss = torch.tensor(0.0, device=dec_out.device)
    total_tokens = 0
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        chunk_logits = model.decoder.output_head(dec_out[:, start:end, :])
        chunk_targets = targets[:, start:end]
        n_tokens = chunk_targets.numel()
        if n_tokens == 0:
            continue
        chunk_loss = criterion(
            chunk_logits.reshape(-1, STAGE2_VOCAB_SIZE), chunk_targets.reshape(-1)
        )
        total_loss = total_loss + chunk_loss * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1), total_tokens


def _compute_accuracy_on_sample(
    model: nn.Module,
    dec_out: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = PAD,
    max_tokens: int = 2048,
) -> float:
    B = dec_out.shape[0]
    flat_dec = dec_out.reshape(B, -1, dec_out.shape[-1])[:, :max_tokens, :]
    flat_tgt = targets.reshape(B, -1)[:, :max_tokens]
    logits = model.decoder.output_head(flat_dec)
    preds = logits.argmax(dim=-1)
    mask = flat_tgt != pad_id
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == flat_tgt[mask]).float().mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    best_loss: float,
    path: Path,
    batch_idx: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "batch_idx": batch_idx,
            "best_val_loss": best_loss,
            "rng_state": _capture_rng_state(),
            "config": {
                "d_model": model.config.d_model,
                "nhead": model.config.nhead,
                "num_decoder_layers": model.config.num_decoder_layers,
                "dim_feedforward": model.config.dim_feedforward,
                "dropout": model.config.dropout,
                "n_mels": model.config.n_mels,
                "encoder_num_layers": model.config.encoder_num_layers,
            },
        },
        path,
    )
    print(f"  Saved checkpoint: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_stage2(cfg: Stage2TrainConfig) -> None:
    _set_seed(cfg.seed)

    device_str = cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    # ── Mel stats ────────────────────────────────────────────────────────
    stats_path = Path(cfg.stats_file)
    if stats_path.exists():
        print(f"Loading mel stats from {stats_path}")
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        mel_mean, mel_std = stats["mean"], stats["std"]
    else:
        print("Computing mel stats …")
        mel_mean, mel_std = compute_mel_stats(cfg.data_dir, max_files=200)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mel_mean, "std": mel_std}, stats_path)
        print(f"Saved mel stats to {stats_path}")

    mel_stats = (mel_mean, mel_std)

    # ── Validation data ──────────────────────────────────────────────────
    dl_val = _build_dataloader(cfg, mel_stats, "val", shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"

    resume_ckpt: Optional[Dict] = None
    if cfg.resume and latest_path.exists():
        print(f"Loading resume metadata from {latest_path}")
        resume_ckpt = torch.load(latest_path, map_location="cpu", weights_only=True)

    if resume_ckpt is not None and "config" in resume_ckpt:
        model_cfg = Stage2Config(**resume_ckpt["config"])
        print(
            f"Resumed model config: d_model={model_cfg.d_model}, "
            f"nhead={model_cfg.nhead}, layers={model_cfg.num_decoder_layers}"
        )
    else:
        model_cfg = Stage2Config(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            encoder_num_layers=cfg.encoder_num_layers,
        )
    model = Stage2Model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage2 model params: {n_params / 1e6:.1f} M")

    # ── Stage 1 model for scheduled onset sampling (optional) ────────────
    stage1_model = None
    if cfg.stage1_checkpoint:
        print(f"Loading Stage 1 model from {cfg.stage1_checkpoint}")
        from src.stage1_model import Stage1Model
        s1_ckpt = torch.load(cfg.stage1_checkpoint, map_location="cpu", weights_only=True)
        s1_cfg = s1_ckpt.get("config", {})
        stage1_model = Stage1Model(
            n_mels=s1_cfg.get("n_mels", 80),
            cond_dim=s1_cfg.get("cond_dim", 1),
            gate_d_model=s1_cfg.get("gate_d_model", 512),
            use_difficulty_gate=s1_cfg.get("use_difficulty_gate", True),
        ).to(device)
        stage1_model.load_state_dict(s1_ckpt["model_state_dict"])
        stage1_model.eval()
        for p in stage1_model.parameters():
            p.requires_grad = False
        print("  Stage 1 model loaded and frozen")

    # ── Optimiser & scheduler ────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    dl_train_first = _build_dataloader(cfg, mel_stats, "train", epoch=1)
    steps_per_epoch = (
        len(dl_train_first) // cfg.gradient_accumulation_steps
    )
    total_steps = cfg.epochs * steps_per_epoch

    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0,
        total_iters=cfg.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - cfg.warmup_steps,
        eta_min=cfg.eta_min,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_steps],
    )

    # ── AMP ──────────────────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using AMP")

    # ── Resume state ─────────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 1
    resume_batch_idx = 0

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if "rng_state" in resume_ckpt:
            _restore_rng_state(resume_ckpt["rng_state"])
        start_epoch = resume_ckpt.get("epoch", 1)
        global_step = resume_ckpt.get("step", 0)
        best_val_loss = resume_ckpt.get("best_val_loss", float("inf"))
        resume_batch_idx = resume_ckpt.get("batch_idx", 0)
        print(
            f"Resumed training at epoch {start_epoch}, step {global_step}, "
            f"batch {resume_batch_idx}"
        )
        optimizer.zero_grad()

    # ── Pause handling ───────────────────────────────────────────────────
    pause = _PauseController()
    signal.signal(signal.SIGINT, pause.handler)

    # ── Onset sampling schedule ──────────────────────────────────────────
    onset_sample_final = cfg.onset_sample_end
    onset_sample_start = cfg.onset_sample_start

    print(
        f"Train samples: {len(dl_train_first.dataset)}, "
        f"Val samples: {len(dl_val.dataset)}"
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        dl_train = _build_dataloader(cfg, mel_stats, "train", epoch=epoch)

        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_batches = 0
        accum_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{cfg.epochs}")

        # ── Onset sampling fraction for this epoch ───────────────────────
        onset_frac = onset_sample_start
        if onset_sample_end > onset_sample_start and cfg.epochs > 1:
            onset_frac = onset_sample_start + (
                onset_sample_end - onset_sample_start
            ) * (epoch - 1) / (cfg.epochs - 1)

        for batch_idx, batch in enumerate(pbar):
            if epoch == start_epoch and batch_idx < resume_batch_idx:
                continue

            spectrogram = batch["spectrogram"].to(device)
            bpm_signal = batch["bpm_signal"].to(device)
            chart_constant = batch["chart_constant"].to(device)
            tokens_s2 = batch["tokens_stage2"].to(device)
            abs_times_s2 = batch["abs_times_stage2"].to(device)
            spec_mask = batch["spec_mask"].to(device)
            tok_mask = batch["tok_mask"].to(device)

            # ── Scheduled onset sampling (when enabled) ──────────────────
            if stage1_model is not None and onset_frac > 0.0:
                use_predicted = torch.rand(1).item() < onset_frac
                if use_predicted:
                    with torch.no_grad():
                        onset_logits = stage1_model(
                            spectrogram, chart_constant, bpm_signal
                        )
                        onset_probs = torch.sigmoid(onset_logits)
                        onset_mask = (onset_probs > 0.5).float()

                    # Build Stage 2 sequences from predicted onsets + GT notes
                    tokens_s2, abs_times_s2 = _build_tokens_from_onset_mask(
                        batch, onset_mask, device
                    )
                    if tokens_s2 is not None:
                        # Update tok_mask to match new sequence length
                        L = tokens_s2.shape[1]
                        tok_mask = torch.zeros(
                            tokens_s2.shape[0], L, dtype=torch.bool, device=device
                        )

            with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                dec_out, targets = model(
                    spectrogram,
                    bpm_signal,
                    chart_constant,
                    tokens_s2,
                    abs_times_s2,
                    spec_mask=spec_mask,
                    tok_mask=tok_mask,
                )
                loss, _ = _compute_loss_chunked(
                    model, dec_out, targets, criterion, chunk_size=512
                )
                loss = loss / cfg.gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item() * cfg.gradient_accumulation_steps
            epoch_loss += loss.item() * cfg.gradient_accumulation_steps
            epoch_acc += _compute_accuracy_on_sample(model, dec_out.float(), targets)
            epoch_batches += 1

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                ).item()
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_interval == 0:
                    pbar.set_postfix(
                        loss=f"{accum_loss:.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        gnorm=f"{grad_norm:.1f}",
                    )
                    accum_loss = 0.0

                if pause.pause_requested:
                    next_batch = batch_idx + 1
                    next_epoch = epoch
                    if next_batch >= len(dl_train):
                        next_epoch = epoch + 1
                        next_batch = 0
                        if next_epoch > cfg.epochs:
                            print("\nPause requested at end of training.")
                            return
                    _save_checkpoint(
                        model, optimizer, scheduler,
                        next_epoch, global_step, best_val_loss,
                        latest_path, batch_idx=next_batch,
                    )
                    print(
                        f"\nTraining paused at epoch {epoch}, step {global_step}. "
                        f"Run with --resume to continue from {latest_path}."
                    )
                    return

                if global_step % cfg.val_interval == 0:
                    val_loss = _validate(model, dl_val, device, criterion)
                    print(f"  Step {global_step}: val_loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_checkpoint(
                            model, optimizer, scheduler,
                            epoch, global_step, best_val_loss,
                            ckpt_dir / "best.pt",
                        )
                    model.train()

        # ── End of epoch ─────────────────────────────────────────────────
        avg_loss = epoch_loss / max(epoch_batches, 1)
        avg_acc = epoch_acc / max(epoch_batches, 1)
        print(f"Epoch {epoch} finished: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

        _val_decode_sample(model, dl_val, device)

        if epoch % cfg.save_every_epochs == 0:
            _save_checkpoint(
                model, optimizer, scheduler,
                epoch, global_step, best_val_loss,
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

    print("Training complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduled onset sampling helper
# ═══════════════════════════════════════════════════════════════════════════════


def _build_tokens_from_onset_mask(
    batch: Dict[str, torch.Tensor],
    onset_mask: torch.Tensor,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build stage‑2 token sequences from Stage 1 predicted onset mask.

    Returns ``(tokens_s2, abs_times_s2)`` or ``(None, None)`` on failure.
    """
    from src.tokenizer import compute_abs_times_stage2, CC as _CC, ONSET as _O, END_ONSET as _EO, EOS as _E, SOS as _S
    import numpy as np

    tokens_list = []
    abs_times_list = []
    max_len = 0

    onset_ms_tensors = batch["onset_times_ms_stage2"]
    gt_tokens_tensors = [batch["tokens_stage2"][i] for i in range(len(onset_ms_tensors))]

    for b in range(len(onset_ms_tensors)):
        # Get predicted onset frames
        valid_frames = onset_mask[b] > 0.5
        onset_frames = torch.nonzero(valid_frames).squeeze(1).tolist()
        onset_times = [f * 10 for f in onset_frames]  # frame → ms

        # Build minimal tokens: SOS, CC, ONSET blocks with GT notes at each onset
        # For now, use empty blocks (just ONSET END_ONSET) — the model learns to
        # output END_ONSET when there are no notes.
        tokens: list[int] = [_S, _CC]
        if onset_times:
            gt_onset_set = set(round(t.item()) for t in onset_ms_tensors[b])
        else:
            gt_onset_set = set()

        for ot_ms in onset_times:
            tokens.append(_O)
            # If GT has notes at this onset, include them
            if ot_ms in gt_onset_set:
                # Find the GT token block for this onset
                gt_tokens = gt_tokens_tensors[b].tolist()
                # Extract notes for this onset from GT tokens
                in_block = False
                start_found = False
                for tok in gt_tokens:
                    if tok == _O:
                        if start_found:
                            break
                        # Check if this onset time matches
                        # (simple: assume onset order matches)
                        start_found = True
                        in_block = True
                    elif in_block:
                        if tok == _EO or tok == _E or tok == PAD:
                            break
                        tokens.append(tok)
            tokens.append(_EO)

        if tokens[-1] != _E:
            tokens.append(_E)

        abs_t = compute_abs_times_stage2(tokens, onset_times)
        tokens_list.append(torch.tensor(tokens, dtype=torch.long))
        abs_times_list.append(torch.tensor(abs_t, dtype=torch.float32))
        max_len = max(max_len, len(tokens))

    if max_len == 0:
        return None, None

    # Pad to max length
    tokens_padded = []
    abs_padded = []
    for tl, at in zip(tokens_list, abs_times_list):
        pad = max_len - len(tl)
        tokens_padded.append(F.pad(tl, (0, pad), value=PAD))
        abs_padded.append(F.pad(at, (0, pad)))
    return (
        torch.stack(tokens_padded).to(device),
        torch.stack(abs_padded).to(device),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def _validate(
    model: nn.Module,
    dl_val: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int = 20,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    use_amp = device.type == "cuda"
    for batch in dl_val:
        if n >= max_batches:
            break
        spectrogram = batch["spectrogram"].to(device)
        bpm_signal = batch["bpm_signal"].to(device)
        chart_constant = batch["chart_constant"].to(device)
        tokens_s2 = batch["tokens_stage2"].to(device)
        abs_times_s2 = batch["abs_times_stage2"].to(device)
        spec_mask = batch["spec_mask"].to(device)
        tok_mask = batch["tok_mask"].to(device)

        with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
            dec_out, targets = model(
                spectrogram, bpm_signal, chart_constant,
                tokens_s2, abs_times_s2,
                spec_mask=spec_mask, tok_mask=tok_mask,
            )
            loss, _ = _compute_loss_chunked(
                model, dec_out, targets, criterion, chunk_size=512
            )
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def _val_decode_sample(
    model: Stage2Model,
    dl_val: DataLoader,
    device: torch.device,
) -> None:
    """Run greedy generation on one validation sample using GT onsets."""
    model.eval()
    try:
        batch = next(iter(dl_val))
    except StopIteration:
        return

    spectrogram = batch["spectrogram"][:1].to(device)
    bpm_signal = batch["bpm_signal"][:1].to(device)
    chart_constant = batch["chart_constant"][:1].to(device)
    gt_tokens = batch["tokens_stage2"][0].tolist()
    onset_times = batch["onset_times_ms_stage2"]
    gt_onset_ms = [round(t.item()) for t in onset_times[0]] if onset_times else []

    gt_tokens = [t for t in gt_tokens if t != PAD]

    try:
        gen_tokens = model.generate(
            spectrogram,
            bpm_signal,
            chart_constant,
            onset_times_ms=gt_onset_ms,
            max_notes_per_onset=32,
            temperature=0.0,
            validator=Stage2Validator(),
        )
    except Exception as e:
        print(f"  [generate error: {e}]")
        return

    try:
        gt_notes = ChartTokenizer._decode_notes_standalone_stage2(
            gt_tokens, gt_onset_ms
        )
    except Exception as e:
        gt_notes = []
        print(f"  [decode GT error: {e}]")

    try:
        gen_notes = ChartTokenizer._decode_notes_standalone_stage2(
            gen_tokens, gt_onset_ms
        )
    except Exception as e:
        gen_notes = []
        print(f"  [decode GEN error (expected for untrained model): {e}]")

    print(
        f"  [val sample] GT: {len(gt_notes)} notes, "
        f"{len(gt_tokens)} tokens | "
        f"Gen: {len(gen_notes)} notes, "
        f"{len(gen_tokens)} tokens"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 ChartGPT")
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--stats_file", default="./dataset/mel_stats.pt")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/stage2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stage1_checkpoint",
        help="Path to Stage 1 model for scheduled onset sampling",
    )
    args = parser.parse_args()

    cfg = Stage2TrainConfig(
        data_dir=args.data_dir,
        stats_file=args.stats_file,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
        stage1_checkpoint=args.stage1_checkpoint,
    )
    train_stage2(cfg)


if __name__ == "__main__":
    main()
