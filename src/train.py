from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloader import (
    MaiMaiDataset,
    collate_fn,
    compute_mel_stats,
)
from src.model import ChartGPT, ChartGPTConfig
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


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataloaders(
    cfg: TrainConfig, mel_stats: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[DataLoader, DataLoader]:
    ds_train = MaiMaiDataset(
        cfg.data_dir,
        split="train",
        val_fraction=cfg.val_fraction,
        mel_stats=mel_stats,
        max_tokens=cfg.max_tokens,
    )
    ds_val = MaiMaiDataset(
        cfg.data_dir,
        split="val",
        val_fraction=cfg.val_fraction,
        mel_stats=mel_stats,
        max_tokens=cfg.max_tokens,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dl_train, dl_val


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

    # ── Data ─────────────────────────────────────────────────────────────
    dl_train, dl_val = _build_dataloaders(cfg, (mel_mean, mel_std))
    print(f"Train charts: {len(dl_train.dataset)}, Val charts: {len(dl_val.dataset)}")

    # ── Model ────────────────────────────────────────────────────────────
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
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.warmup_steps, T_mult=2)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # AMP scaler (CUDA only – no-op on CPU / MPS)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using AMP (automatic mixed precision)")

    # ── Checkpoint dir ───────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training state ───────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_batches = 0
        accum_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{cfg.epochs}")

        for batch_idx, batch in enumerate(pbar):
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
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
                    )
                    accum_loss = 0.0

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
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_val_loss": best_loss,
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
    )
    train(cfg)


if __name__ == "__main__":
    main()
