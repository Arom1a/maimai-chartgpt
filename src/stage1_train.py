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
    collate_fn,
    compute_mel_stats,
)
from src.stage1_model import Stage1Model


# ═══════════════════════════════════════════════════════════════════════════════
# Training config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Stage1TrainConfig:
    data_dir: str = "./dataset"
    stats_file: str = "./dataset/mel_stats.pt"
    checkpoint_dir: str = "./checkpoints/stage1"

    # ── Model ──────────────────────────────────────────────────────────
    n_mels: int = 80
    gate_d_model: int = 512

    # ── Phase 1: pretrain base detector (no gate) ──────────────────────
    pretrain_epochs: int = 20
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 8
    pretrain_cond_dim: int = 1

    # ── Phase 2: add difficulty gate ───────────────────────────────────
    gate_epochs: int = 30
    gate_lr: float = 1e-4
    gate_batch_size: int = 4

    # ── Gumbel-Softmax (phase 2) ───────────────────────────────────────
    temperature_start: float = 1.0
    temperature_end: float = 0.5
    temperature_anneal_epochs: int = 20

    # ── Common ─────────────────────────────────────────────────────────
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    log_interval: int = 50
    val_interval: int = 500
    save_every_epochs: int = 5
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    resume: bool = False
    max_tokens: int = 8192


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
    cfg: Stage1TrainConfig,
    mel_stats: Tuple[torch.Tensor, torch.Tensor],
    split: str,
    batch_size: int,
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
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,
        generator=generator,
        worker_init_fn=_WorkerInitFn(cfg.seed + epoch) if shuffle else None,
    )


def _compute_f1(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    """F1 score on valid (non-padded) frames."""
    p = preds[mask].long()
    t = targets[mask].long()
    tp = (p & t).sum().float()
    fp = (p & ~t).sum().float()
    fn = (~p & t).sum().float()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return (2 * prec * rec / (prec + rec + 1e-8)).item()


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_loss: float,
    path: Path,
    stage: str,
    batch_idx: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "batch_idx": batch_idx,
            "best_val_loss": best_loss,
            "rng_state": _capture_rng_state(),
            "stage": stage,
            "config": {
                "n_mels": model.n_mels,
                "gate_d_model": model.gate.d_model if model.use_difficulty_gate else 0,
                "cond_dim": model.cond_dim,
                "use_difficulty_gate": model.use_difficulty_gate,
            },
        },
        path,
    )
    print(f"  Saved checkpoint: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Pretrain base detector (no difficulty gate)
# ═══════════════════════════════════════════════════════════════════════════════


def _pretrain_onset(cfg: Stage1TrainConfig) -> str:
    """Pretrain the OnsetDetector without difficulty gating.

    Returns the path to the best pretrain checkpoint.
    """
    print("=" * 60)
    print("Phase 1: Pretraining base onset detector (no difficulty gate)")
    print("=" * 60)

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

    # ── Data ─────────────────────────────────────────────────────────────
    dl_train = _build_dataloader(
        cfg, mel_stats, "train", batch_size=cfg.pretrain_batch_size, epoch=1
    )
    dl_val = _build_dataloader(
        cfg, mel_stats, "val", batch_size=cfg.pretrain_batch_size, shuffle=False
    )
    print(
        f"Train samples: {len(dl_train.dataset)}, "
        f"Val samples: {len(dl_val.dataset)}"
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = Stage1Model(
        n_mels=cfg.n_mels,
        cond_dim=cfg.pretrain_cond_dim,
        gate_d_model=cfg.gate_d_model,
        use_difficulty_gate=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage1 pretrain params: {n_params / 1e3:.1f} k")

    # ── Optimiser ────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(), lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.pretrain_epochs * len(dl_train)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── AMP ──────────────────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using AMP")

    # ── Checkpoint dir ───────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    pause = _PauseController()
    signal.signal(signal.SIGINT, pause.handler)

    for epoch in range(1, cfg.pretrain_epochs + 1):
        if epoch > 1:
            dl_train = _build_dataloader(
                cfg, mel_stats, "train", batch_size=cfg.pretrain_batch_size, epoch=epoch
            )

        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        accum_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Pretrain epoch {epoch}/{cfg.pretrain_epochs}")

        for batch_idx, batch in enumerate(pbar):
            mel = batch["spectrogram"].to(device)
            bpm = batch["bpm_signal"].to(device)
            spec_mask = batch["spec_mask"].to(device)
            onset_labels = batch["onset_labels"].to(device)

            cond = bpm.unsqueeze(1) if cfg.pretrain_cond_dim > 0 else None

            with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                logits = model.detector(mel, cond)
                valid = ~spec_mask
                loss = F.binary_cross_entropy_with_logits(
                    logits[valid], onset_labels[valid]
                )

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()
            epoch_loss += loss.item()
            epoch_batches += 1

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
                with torch.no_grad():
                    preds = (logits > 0.0).float()
                    f1 = _compute_f1(preds, onset_labels, valid)
                pbar.set_postfix(
                    loss=f"{accum_loss:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    f1=f"{f1:.3f}",
                )
                accum_loss = 0.0

            if pause.pause_requested:
                _save_checkpoint(
                    model, optimizer, epoch, global_step, best_loss,
                    ckpt_dir / "pretrain_latest.pt", "onset_pretrain",
                    batch_idx=batch_idx + 1,
                )
                print(
                    f"\nTraining paused at epoch {epoch}, step {global_step}. "
                    f"Run with --resume to continue."
                )
                return str(ckpt_dir / "pretrain_latest.pt")

            if global_step % cfg.val_interval == 0:
                val_loss = _validate_onset(model, dl_val, device, cfg)
                print(f"  Step {global_step}: val_loss={val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    _save_checkpoint(
                        model, optimizer, epoch, global_step, best_loss,
                        ckpt_dir / "pretrain_best.pt", "onset_pretrain",
                    )
                model.train()

        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"Pretrain epoch {epoch} finished: loss={avg_loss:.4f}")

        _val_report(model, dl_val, device)

    # Save final
    _save_checkpoint(
        model, optimizer, cfg.pretrain_epochs, global_step, best_loss,
        ckpt_dir / "pretrain_best.pt", "onset_pretrain",
    )
    print("Phase 1 complete — best model saved to pretrain_best.pt")
    return str(ckpt_dir / "pretrain_best.pt")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Train with difficulty gate
# ═══════════════════════════════════════════════════════════════════════════════


def _train_with_gate(cfg: Stage1TrainConfig, pretrain_path: str) -> str:
    """Load pretrained detector, add DifficultyGate, and train jointly.

    Returns the path to the best gate checkpoint.
    """
    print("=" * 60)
    print("Phase 2: Training with difficulty gate")
    print("=" * 60)

    device_str = cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    # ── Mel stats ────────────────────────────────────────────────────────
    stats = torch.load(cfg.stats_file, map_location="cpu", weights_only=True)
    mel_stats = (stats["mean"], stats["std"])

    # ── Data ─────────────────────────────────────────────────────────────
    dl_train = _build_dataloader(
        cfg, mel_stats, "train", batch_size=cfg.gate_batch_size, epoch=1
    )
    dl_val = _build_dataloader(
        cfg, mel_stats, "val", batch_size=cfg.gate_batch_size, shuffle=False
    )
    print(
        f"Train samples: {len(dl_train.dataset)}, "
        f"Val samples: {len(dl_val.dataset)}"
    )

    # ── Model ────────────────────────────────────────────────────────────
    # Create full model and load pretrained detector weights
    model = Stage1Model(
        n_mels=cfg.n_mels,
        cond_dim=cfg.pretrain_cond_dim,  # same cond dim as pretraining
        gate_d_model=cfg.gate_d_model,
        use_difficulty_gate=True,
    ).to(device)

    pretrain_ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=True)
    detector_state = {
        k.removeprefix("detector."): v
        for k, v in pretrain_ckpt["model_state_dict"].items()
        if k.startswith("detector.")
    }
    model.detector.load_state_dict(detector_state)
    print(f"Loaded pretrained detector from {pretrain_path}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage1 with gate params: {n_params / 1e3:.1f} k")

    # ── Optimiser ────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(), lr=cfg.gate_lr, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.gate_epochs * len(dl_train)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── AMP ──────────────────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Checkpointing ────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    pause = _PauseController()
    signal.signal(signal.SIGINT, pause.handler)

    # Gumbel temperature annealing: linear from start → end
    anneal_total = cfg.temperature_anneal_epochs * len(dl_train)

    for epoch in range(1, cfg.gate_epochs + 1):
        if epoch > 1:
            dl_train = _build_dataloader(
                cfg, mel_stats, "train", batch_size=cfg.gate_batch_size, epoch=epoch
            )

        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        accum_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Gate epoch {epoch}/{cfg.gate_epochs}")

        for batch_idx, batch in enumerate(pbar):
            mel = batch["spectrogram"].to(device)
            bpm = batch["bpm_signal"].to(device)
            chart_const = batch["chart_constant"].to(device)
            spec_mask = batch["spec_mask"].to(device)
            onset_labels = batch["onset_labels"].to(device)

            # Linear temperature anneal
            t_frac = min(1.0, global_step / anneal_total)
            temperature = (
                cfg.temperature_start
                + (cfg.temperature_end - cfg.temperature_start) * t_frac
            )

            with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                logits = model(mel, chart_const, bpm)
                valid = ~spec_mask

                # BCE loss (main supervision)
                bce_loss = F.binary_cross_entropy_with_logits(
                    logits[valid], onset_labels[valid]
                )

                # Optional: density regularisation via Gumbel-Softmax
                y_mask = model.sample_binary(mel, chart_const, bpm, temperature)
                pred_onsets = y_mask.sum(dim=1)  # (B,)
                target_onsets = onset_labels.sum(dim=1)  # (B,)
                density_loss = ((pred_onsets - target_onsets) ** 2).mean()

                loss = bce_loss + 0.01 * density_loss

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()
            epoch_loss += loss.item()
            epoch_batches += 1

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
                with torch.no_grad():
                    preds = (logits > 0.0).float()
                    f1 = _compute_f1(preds, onset_labels, valid)
                pbar.set_postfix(
                    loss=f"{accum_loss:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    f1=f"{f1:.3f}",
                    temp=f"{temperature:.2f}",
                )
                accum_loss = 0.0

            if pause.pause_requested:
                _save_checkpoint(
                    model, optimizer, epoch, global_step, best_loss,
                    ckpt_dir / "gate_latest.pt", "onset_with_gate",
                    batch_idx=batch_idx + 1,
                )
                print(
                    f"\nTraining paused at epoch {epoch}, step {global_step}."
                )
                return str(ckpt_dir / "gate_latest.pt")

            if global_step % cfg.val_interval == 0:
                val_loss = _validate_onset_with_gate(model, dl_val, device)
                print(f"  Step {global_step}: val_loss={val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    _save_checkpoint(
                        model, optimizer, epoch, global_step, best_loss,
                        ckpt_dir / "gate_best.pt", "onset_with_gate",
                    )
                model.train()

        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"Gate epoch {epoch} finished: loss={avg_loss:.4f}")

        _val_density_report(model, dl_val, device)

    _save_checkpoint(
        model, optimizer, cfg.gate_epochs, global_step, best_loss,
        ckpt_dir / "gate_best.pt", "onset_with_gate",
    )
    print("Phase 2 complete — best model saved to gate_best.pt")
    return str(ckpt_dir / "gate_best.pt")


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def _validate_onset(
    model: Stage1Model,
    dl_val: DataLoader,
    device: torch.device,
    cfg: Stage1TrainConfig,
    max_batches: int = 20,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    use_amp = device.type == "cuda"
    for batch in dl_val:
        if n >= max_batches:
            break
        mel = batch["spectrogram"].to(device)
        bpm = batch["bpm_signal"].to(device)
        spec_mask = batch["spec_mask"].to(device)
        onset_labels = batch["onset_labels"].to(device)
        cond = bpm.unsqueeze(1) if cfg.pretrain_cond_dim > 0 else None
        with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
            logits = model.detector(mel, cond)
            valid = ~spec_mask
            loss = F.binary_cross_entropy_with_logits(
                logits[valid], onset_labels[valid]
            )
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def _validate_onset_with_gate(
    model: Stage1Model,
    dl_val: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    use_amp = device.type == "cuda"
    for batch in dl_val:
        if n >= max_batches:
            break
        mel = batch["spectrogram"].to(device)
        bpm = batch["bpm_signal"].to(device)
        chart_const = batch["chart_constant"].to(device)
        spec_mask = batch["spec_mask"].to(device)
        onset_labels = batch["onset_labels"].to(device)
        with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
            logits = model(mel, chart_const, bpm)
            valid = ~spec_mask
            loss = F.binary_cross_entropy_with_logits(
                logits[valid], onset_labels[valid]
            )
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def _val_report(
    model: Stage1Model,
    dl_val: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> None:
    """Print F1/precision/recall on validation set."""
    model.eval()
    tp = fp = fn = 0
    n = 0
    for batch in dl_val:
        if n >= max_batches:
            break
        mel = batch["spectrogram"].to(device)
        bpm = batch["bpm_signal"].to(device)
        spec_mask = batch["spec_mask"].to(device)
        onset_labels = batch["onset_labels"].to(device)
        cond = bpm.unsqueeze(1) if model.cond_dim > 0 else None
        logits = model.detector(mel, cond)
        preds = (logits > 0.0).float()
        valid = ~spec_mask
        p = preds[valid].long()
        t = onset_labels[valid].long()
        tp += (p & t).sum().item()
        fp += (p & ~t).sum().item()
        fn += (~p & t).sum().item()
        n += 1
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"  [val] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")


@torch.no_grad()
def _val_density_report(
    model: Stage1Model,
    dl_val: DataLoader,
    device: torch.device,
    max_batches: int = 40,
) -> None:
    """Log onset density per chart-constant bin to verify gate learning."""
    model.eval()
    bins: Dict[int, list] = {}
    n = 0
    for batch in dl_val:
        if n >= max_batches:
            break
        mel = batch["spectrogram"].to(device)
        bpm = batch["bpm_signal"].to(device)
        chart_const = batch["chart_constant"].to(device)
        onset_labels = batch["onset_labels"].to(device)
        spec_mask = batch["spec_mask"].to(device)
        logits = model(mel, chart_const, bpm)
        preds = (logits > 0.0).float()
        valid = ~spec_mask
        for b in range(mel.shape[0]):
            cc = chart_const[b].item()
            pred_count = preds[b][valid[b]].sum().item()
            target_count = onset_labels[b][valid[b]].sum().item()
            bins.setdefault(cc, []).append((pred_count, target_count))
        n += 1
    if not bins:
        return
    lines = []
    for cc in sorted(bins):
        pairs = bins[cc]
        preds = [p for p, _ in pairs]
        targets = [t for _, t in pairs]
        lines.append(
            f"  const {cc/10:.1f}: pred={sum(preds)/len(preds):.0f}±…, "
            f"target={sum(targets)/len(targets):.0f}±…"
        )
    print("  [val density by constant]")
    for line in lines[:12]:
        print(line)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def train_stage1(cfg: Stage1TrainConfig) -> None:
    _set_seed(cfg.seed)

    if cfg.resume:
        ckpt_dir = Path(cfg.checkpoint_dir)
        gate_latest = ckpt_dir / "gate_latest.pt"
        pretrain_latest = ckpt_dir / "pretrain_latest.pt"
        if gate_latest.exists():
            print(f"Resuming from {gate_latest}")
            resume_ckpt = torch.load(gate_latest, map_location="cpu", weights_only=True)
            stage = resume_ckpt.get("stage", "onset_pretrain")
            if stage == "onset_with_gate":
                # Jump straight to gate phase
                _resume_gate(cfg, gate_latest)
                return
        if pretrain_latest.exists():
            print(f"Resuming pretrain from {pretrain_latest}")
            # Resume pretrain — for simplicity, restart pretrain
            # (detector state can be loaded, but step resume is complex)
            print("  (restarting pretrain phase)")
            pretrain_path = _pretrain_onset(cfg)
            _train_with_gate(cfg, pretrain_path)
            return

    pretrain_path = _pretrain_onset(cfg)
    _train_with_gate(cfg, pretrain_path)
    print("Stage 1 training complete.")


def _resume_gate(cfg: Stage1TrainConfig, path: Path) -> None:
    """Resume gate training from a checkpoint."""
    # Simplified: restart with loaded model
    print("  Restarting gate phase with latest model state")
    _train_with_gate(cfg, str(path))


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 onset detector")
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--stats_file", default="./dataset/mel_stats.pt")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/stage1")
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--gate_epochs", type=int, default=30)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--gate_lr", type=float, default=1e-4)
    parser.add_argument("--pretrain_batch_size", type=int, default=8)
    parser.add_argument("--gate_batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = Stage1TrainConfig(
        data_dir=args.data_dir,
        stats_file=args.stats_file,
        checkpoint_dir=args.checkpoint_dir,
        pretrain_epochs=args.pretrain_epochs,
        gate_epochs=args.gate_epochs,
        pretrain_lr=args.pretrain_lr,
        gate_lr=args.gate_lr,
        pretrain_batch_size=args.pretrain_batch_size,
        gate_batch_size=args.gate_batch_size,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
    )
    train_stage1(cfg)


if __name__ == "__main__":
    main()
