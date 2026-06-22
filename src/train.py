from __future__ import annotations

import argparse
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloader import (
    MaiMaiDataset,
    collate_fn,
    compute_mel_stats,
)
from src.model import ChartGPT, ChartGPTConfig
from src.token_validator import ChartValidator
from src.tokenizer import EOS, PAD, VOCAB_SIZE, ChartTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Training config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "./dataset"
    stats_file: str = "./dataset/mel_stats.pt"

    # Model
    d_model: int = 512
    nhead: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    eta_min: float = 1e-5  # minimum LR for cosine schedule
    grad_clip: float = 1.0
    val_fraction: float = 0.1

    # Logging / checkpointing
    log_interval: int = 50
    val_interval: int = 500
    checkpoint_dir: str = "./checkpoints"
    save_every_epochs: int = 5

    # Device
    device: str = "cuda"

    # Sequence filtering
    max_tokens: int = 8192  # drop charts longer than this

    # Misc
    num_workers: int = 4
    seed: int = 42

    # Pause / resume
    resume: bool = False  # auto-load checkpoints/latest.pt if present


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class _PauseController:
    """Handles Ctrl-C by pausing at the next clean optimizer-step boundary.

    The first SIGINT sets ``pause_requested`` and prints a message.  A second
    SIGINT forces an immediate exit without saving a checkpoint.
    """

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
    """Picklable worker init callable that seeds each DataLoader worker.

    This is defined as a class instead of a closure because Windows uses the
    ``spawn`` multiprocessing start method, which cannot pickle local
    functions created inside another function.
    """

    def __init__(self, base_seed: int) -> None:
        self.base_seed = base_seed

    def __call__(self, worker_id: int) -> None:
        torch.manual_seed(self.base_seed + worker_id)


def _build_train_dataloader(
    cfg: TrainConfig,
    mel_stats: Tuple[torch.Tensor, torch.Tensor],
    epoch: int,
) -> DataLoader:
    ds_train = MaiMaiDataset(
        cfg.data_dir,
        split="train",
        val_fraction=cfg.val_fraction,
        mel_stats=mel_stats,
        max_tokens=cfg.max_tokens,
    )
    # Deterministic per-epoch shuffling so mid-epoch resume replays the same
    # batch order.
    generator = torch.Generator().manual_seed(cfg.seed + epoch)
    return DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        generator=generator,
        worker_init_fn=_WorkerInitFn(cfg.seed + epoch),
    )


def _build_val_dataloader(
    cfg: TrainConfig, mel_stats: Tuple[torch.Tensor, torch.Tensor]
) -> DataLoader:
    ds_val = MaiMaiDataset(
        cfg.data_dir,
        split="val",
        val_fraction=cfg.val_fraction,
        mel_stats=mel_stats,
        max_tokens=cfg.max_tokens,
    )
    return DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def _compute_loss_chunked(
    model: nn.Module,
    dec_out: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, int]:
    """Compute cross-entropy loss in chunks to avoid ``(B, L, V)`` logits."""
    B, L, _ = dec_out.shape
    total_loss = torch.tensor(0.0, device=dec_out.device)
    total_tokens = 0
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        chunk_logits = model.output_head(dec_out[:, start:end, :])  # (B, C, V)
        chunk_targets = targets[:, start:end]  # (B, C)
        n_tokens = chunk_targets.numel()
        if n_tokens == 0:
            continue
        chunk_loss = criterion(
            chunk_logits.reshape(-1, VOCAB_SIZE), chunk_targets.reshape(-1)
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
    """Token accuracy on the first *max_tokens* valid positions only."""
    B = dec_out.shape[0]
    # Flatten and clip
    flat_dec = dec_out.reshape(B, -1, dec_out.shape[-1])[:, :max_tokens, :]
    flat_tgt = targets.reshape(B, -1)[:, :max_tokens]
    logits = model.output_head(flat_dec)  # (B, max_tokens, V)
    preds = logits.argmax(dim=-1)
    mask = flat_tgt != pad_id
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == flat_tgt[mask]).float().mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════


def _capture_rng_state() -> Dict[str, torch.Tensor]:
    """Capture CPU and (if available) CUDA RNG states for exact resume."""
    state: Dict[str, torch.Tensor] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, torch.Tensor]) -> None:
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def train(cfg: TrainConfig) -> None:
    _set_seed(cfg.seed)
    # Resolve device, falling back with a warning if the requested device is
    # unavailable, but never silently ignoring the user's choice.
    device_str = cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device_str = "cpu"
    elif device_str == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
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

    # ── Validation data (fixed order, built once) ────────────────────────
    dl_val = _build_val_dataloader(cfg, (mel_mean, mel_std))

    # ── Checkpoint dir ───────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"

    # ── Resume metadata ──────────────────────────────────────────────────
    resume_ckpt: Optional[Dict] = None
    if cfg.resume and latest_path.exists():
        print(f"Loading resume metadata from {latest_path}")
        resume_ckpt = torch.load(latest_path, map_location="cpu", weights_only=True)

    # ── Model ────────────────────────────────────────────────────────────
    if resume_ckpt is not None and "config" in resume_ckpt:
        model_cfg = ChartGPTConfig(**resume_ckpt["config"])
        print(
            f"Resumed model config: d_model={model_cfg.d_model}, "
            f"nhead={model_cfg.nhead}, layers={model_cfg.num_decoder_layers}"
        )
    else:
        model_cfg = ChartGPTConfig(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
    model = ChartGPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params / 1e6:.1f} M")

    # ── Optimiser & scheduler ────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # Linear warmup → cosine decay to eta_min over the rest of training.
    # We rebuild the train loader each epoch; its length is stable because
    # drop_last=True.
    steps_per_epoch = len(
        _build_train_dataloader(cfg, (mel_mean, mel_std), epoch=1)
    ) // cfg.gradient_accumulation_steps
    total_steps = cfg.epochs * steps_per_epoch
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

    # AMP scaler (CUDA only – no-op on CPU / MPS)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using AMP (automatic mixed precision)")

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
        # Discard any partial gradients that may have been in flight when the
        # previous run was paused.
        optimizer.zero_grad()

    # ── Pause handling ───────────────────────────────────────────────────
    pause = _PauseController()
    signal.signal(signal.SIGINT, pause.handler)

    for epoch in range(start_epoch, cfg.epochs + 1):
        dl_train = _build_train_dataloader(cfg, (mel_mean, mel_std), epoch)
        if epoch == start_epoch:
            print(
                f"Train charts: {len(dl_train.dataset)}, "
                f"Val charts: {len(dl_val.dataset)}"
            )
        if epoch == start_epoch and resume_batch_idx > 0:
            print(f"Skipping first {resume_batch_idx} batches of epoch {epoch}")

        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_batches = 0
        accum_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{cfg.epochs}")

        for batch_idx, batch in enumerate(pbar):
            if epoch == start_epoch and batch_idx < resume_batch_idx:
                continue

            spectrogram = batch["spectrogram"].to(device)
            bpm_signal = batch["bpm_signal"].to(device)
            chart_constant = batch["chart_constant"].to(device)
            tokens = batch["tokens"].to(device)
            abs_times = batch["abs_times"].to(device)
            spec_mask = batch["spec_mask"].to(device)
            tok_mask = batch["tok_mask"].to(device)

            with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                dec_out, targets = model(
                    spectrogram,
                    bpm_signal,
                    chart_constant,
                    tokens,
                    abs_times,
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
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
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

                # Pause on Ctrl-C at a clean optimizer-step boundary.
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
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        next_epoch,
                        best_val_loss,
                        latest_path,
                        batch_idx=next_batch,
                    )
                    print(
                        f"\nTraining paused at epoch {epoch}, step {global_step}. "
                        f"Run with --resume to continue from {latest_path}."
                    )
                    return

                # Validation
                if global_step % cfg.val_interval == 0:
                    val_loss = _validate(model, dl_val, device, criterion)
                    print(f"  Step {global_step}: val_loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            epoch,
                            best_val_loss,
                            ckpt_dir / "best.pt",
                        )
                    model.train()

        # ── End of epoch ─────────────────────────────────────────────────
        avg_loss = epoch_loss / max(epoch_batches, 1)
        avg_acc = epoch_acc / max(epoch_batches, 1)
        print(f"Epoch {epoch} finished: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

        # Quick validation decode on one sample
        _val_decode_sample(model, dl_val, device)

        if epoch % cfg.save_every_epochs == 0:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                global_step,
                epoch,
                best_val_loss,
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

    print("Training complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Validation helpers
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
        tokens = batch["tokens"].to(device)
        abs_times = batch["abs_times"].to(device)
        spec_mask = batch["spec_mask"].to(device)
        tok_mask = batch["tok_mask"].to(device)

        with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
            dec_out, targets = model(
                spectrogram, bpm_signal, chart_constant,
                tokens, abs_times,
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
    model: ChartGPT,
    dl_val: DataLoader,
    device: torch.device,
) -> None:
    """Run greedy generation on one validation sample and log stats."""
    model.eval()
    try:
        batch = next(iter(dl_val))
    except StopIteration:
        return

    spectrogram = batch["spectrogram"][:1].to(device)
    bpm_signal = batch["bpm_signal"][:1].to(device)
    chart_constant = batch["chart_constant"][:1].to(device)
    gt_tokens = batch["tokens"][0].tolist()

    # Remove padding from ground truth
    gt_tokens = [t for t in gt_tokens if t != PAD]

    try:
        gen_tokens = model.generate(
            spectrogram,
            bpm_signal,
            chart_constant,
            max_len=len(gt_tokens) + 100,
            temperature=0.0,
            validator=ChartValidator(),
        )
    except Exception as e:
        print(f"  [generate error: {e}]")
        return

    # Basic stats
    try:
        gt_notes = ChartTokenizer._decode_notes_standalone(gt_tokens)
    except Exception as e:
        gt_notes = []
        print(f"  [decode GT error: {e}]")

    try:
        gen_notes = ChartTokenizer._decode_notes_standalone(gen_tokens)
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
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
    best_loss: float,
    path: Path,
    batch_idx: int = 0,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
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
            },
        },
        path,
    )
    print(f"  Saved checkpoint: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Train ChartGPT")
    parser.add_argument("--data_dir", default="./dataset", help="Path to dataset root")
    parser.add_argument(
        "--stats_file",
        default="./dataset/mel_stats.pt",
        help="Path to precomputed mel stats",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="Device (cuda / cpu / mps)")
    parser.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/latest.pt if it exists",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        stats_file=args.stats_file,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
    )
    train(cfg)


if __name__ == "__main__":
    main()
