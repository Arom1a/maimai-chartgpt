"""Validate difficulty‑density correlation across the dataset.

Computes the Pearson correlation between chart constant and note count
(ground truth), then optionally evaluates a trained model to compare.

Usage::

    python -m src.validate_density --data_dir ./dataset
    python -m src.validate_density --data_dir ./dataset \\
        --stage2_checkpoint checkpoints/stage2/best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _compute_gt_stats(
    data_dir: str, max_files: int = 500
) -> Tuple[List[float], List[int], List[float], List[int]]:
    """Return (constants, note_counts, constants, onset_counts)."""
    consts_note: List[float] = []
    counts_note: List[int] = []
    consts_onset: List[float] = []
    counts_onset: List[int] = []

    json_files = sorted(Path(data_dir).rglob("processed.json"))[:max_files]
    for path in json_files:
        with open(path) as f:
            data = json.load(f)
        for chart in data["charts"]:
            if not chart["notes"]:
                continue
            major, minor = chart["constant"]
            cc = major + minor / 10.0
            consts_note.append(cc)
            counts_note.append(len(chart["notes"]))
            # Count unique rounded timestamps as onsets
            onsets = len(
                set(round(n["timestamp_ms"] / 10.0) * 10 for n in chart["notes"])
            )
            consts_onset.append(cc)
            counts_onset.append(onsets)

    return consts_note, counts_note, consts_onset, counts_onset


def _bin_stats(
    xs: List[float], ys: List[int]
) -> Dict[float, Tuple[float, float]]:
    """Group *ys* by *xs* (rounded to .1) and return mean ± std."""
    bins: Dict[float, List[int]] = defaultdict(list)
    for x, y in zip(xs, ys):
        bins[round(x, 1)].append(y)
    result = {}
    for k, v in sorted(bins.items()):
        result[k] = (np.mean(v), np.std(v))
    return result


def _evaluate_model(
    model_ckpt: str,
    data_dir: str,
    device: str,
    stats_file: str,
    max_samples: int = 50,
) -> Optional[Tuple[List[float], List[int]]]:
    """Run greedy generation on *max_samples* val charts, return (consts, note_counts)."""
    try:
        import torch.nn.functional as F
        from src.dataloader import (
            MaiMaiDataset,
            collate_fn_stage2,
        )
        from src.stage2_model import Stage2Config, Stage2Model
        from src.token_validator import Stage2Validator
        from torch.utils.data import DataLoader

        # Load stats
        stats = torch.load(stats_file, map_location="cpu", weights_only=True)
        mel_stats = (stats["mean"], stats["std"])

        # Load model
        ckpt = torch.load(model_ckpt, map_location="cpu", weights_only=True)
        if "config" in ckpt:
            cfg = Stage2Config(**ckpt["config"])
        else:
            cfg = Stage2Config()
        model = Stage2Model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Data
        ds = MaiMaiDataset(data_dir, split="val", mel_stats=mel_stats, max_tokens=4096)
        dl = DataLoader(
            ds, batch_size=1, shuffle=False, collate_fn=collate_fn_stage2
        )

        consts: List[float] = []
        note_counts: List[int] = []

        for i, batch in enumerate(dl):
            if i >= max_samples:
                break
            spectrogram = batch["spectrogram"].to(device)
            bpm_signal = batch["bpm_signal"].to(device)
            chart_constant = batch["chart_constant"].to(device)
            onset_times_ms = batch["onset_times_ms_stage2"]

            cc_val = chart_constant.item() / 10.0
            oms = [round(t.item()) for t in onset_times_ms[0]]
            if not oms:
                continue

            try:
                gen = model.generate(
                    spectrogram,
                    bpm_signal,
                    chart_constant,
                    onset_times_ms=oms,
                    max_notes_per_onset=16,
                    temperature=0.0,
                    validator=Stage2Validator(),
                )
                note_count = sum(
                    1 for t in gen
                    if t not in (1, 13267, 13268, 13269, 2, 0)  # SOS, CC, ONSET, END_ONSET, EOS, PAD
                )
                consts.append(cc_val)
                note_counts.append(note_count)
            except Exception:
                pass

        if len(consts) < 3:
            return None
        return consts, note_counts

    except Exception as e:
        print(f"  Model evaluation failed: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate difficulty-density correlation"
    )
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--max_files", type=int, default=500)
    parser.add_argument(
        "--stage2_checkpoint",
        help="Evaluate a trained Stage 2 model",
    )
    parser.add_argument(
        "--stats_file",
        default="./dataset/mel_stats.pt",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # ── Ground truth stats ──────────────────────────────────────────────
    c_n, n_n, c_o, n_o = _compute_gt_stats(args.data_dir, args.max_files)
    if len(c_n) < 3:
        print("Not enough data")
        return

    print(f"Dataset: {len(c_n)} charts from up to {args.max_files} songs\n")

    r_note = float(np.corrcoef(c_n, n_n)[0, 1])
    r_onset = float(np.corrcoef(c_o, n_o)[0, 1])

    print("Ground‑truth correlation with chart constant:")
    print(f"  Note count:    ρ = {r_note:.4f}")
    print(f"  Onset count:   ρ = {r_onset:.4f}")

    # Per-bin breakdown
    note_bins = _bin_stats(c_n, n_n)
    onset_bins = _bin_stats(c_o, n_o)
    print(f"\nPer‑bin means (first 12 bins):")
    print(f"  {'Const':>6}  {'Notes':>6}  {'Onsets':>6}")
    for k in sorted(note_bins)[:12]:
        nm, ns = note_bins[k]
        om, os = onset_bins.get(k, (0, 0))
        print(f"  {k:>5.1f}  {nm:>6.0f}  {om:>6.0f}")

    # ── Model evaluation (optional) ─────────────────────────────────────
    if args.stage2_checkpoint:
        print(f"\nEvaluating model: {args.stage2_checkpoint}")
        result = _evaluate_model(
            args.stage2_checkpoint,
            args.data_dir,
            args.device,
            args.stats_file,
        )
        if result is not None:
            c_m, n_m = result
            r_model = float(np.corrcoef(c_m, n_m)[0, 1])
            print(f"  Model note‑token correlation: ρ = {r_model:.4f}")
            print(f"  (n={len(c_m)} samples)")
        else:
            print("  Model evaluation returned no results")


if __name__ == "__main__":
    main()
