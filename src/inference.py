"""Two-stage inference: Stage 1 onset detection → Stage 2 note generation.

Usage::

    python -m src.inference \\
        --stage1_checkpoint checkpoints/stage1/gate_best.pt \\
        --stage2_checkpoint checkpoints/stage2/best.pt \\
        --audio track.mp3 \\
        --bpm 175.0 \\
        --constant "12,5" \\
        --title "My Song"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio

from src.dataloader import (
    MEL_HOP_LENGTH,
    MEL_WIN_LENGTH,
    N_MELS,
    TARGET_SAMPLE_RATE,
    _load_audio_ffmpeg,
)
from src.stage1_model import Stage1Model
from src.stage2_model import Stage2Config, Stage2Model
from src.token_validator import Stage2Validator
from src.tokenizer import ChartTokenizer


def _build_bpm_signal(
    bpm10_list: list[dict],
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """BPM at each mel frame (100 Hz)."""
    signal = torch.zeros(num_frames, dtype=torch.float32, device=device)
    frame_time_ms = 0.0
    frame_step_ms = MEL_HOP_LENGTH / TARGET_SAMPLE_RATE * 1000.0
    change_pairs = sorted(
        [(r["change_timestamp_ms"], r["bpm10"] / 10.0) for r in bpm10_list],
        key=lambda x: x[0],
    )
    ci = 0
    cur_bpm = change_pairs[0][1] if change_pairs else 150.0
    for fi in range(num_frames):
        while ci < len(change_pairs) and change_pairs[ci][0] <= frame_time_ms:
            cur_bpm = change_pairs[ci][1]
            ci += 1
        signal[fi] = cur_bpm
        frame_time_ms += frame_step_ms
    return signal


def _resolve_bpm(args) -> tuple[list[dict], Optional[str], Optional[str]]:
    if args.chart_metadata:
        with open(args.chart_metadata) as f:
            meta = json.load(f)
        bpm10 = meta["bpm10_list"]
        if not bpm10:
            raise ValueError("chart_metadata JSON must contain 'bpm10_list'")
        return bpm10, meta.get("title"), meta.get("artist")
    if args.bpm is not None:
        return (
            [{"bpm10": int(args.bpm * 10), "change_timestamp_ms": 0}],
            args.title,
            args.artist,
        )
    raise ValueError("One of --bpm or --chart_metadata is required")


def _load_mel(
    audio_path: str,
    device: torch.device,
    mel_mean: Optional[torch.Tensor],
    mel_std: Optional[torch.Tensor],
) -> torch.Tensor:
    waveform = _load_audio_ffmpeg(audio_path, TARGET_SAMPLE_RATE).to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=MEL_WIN_LENGTH,
        hop_length=MEL_HOP_LENGTH,
        n_mels=N_MELS,
    ).to(device)
    mel = mel_transform(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-6))
    if mel_mean is not None:
        mel = (mel - mel_mean.unsqueeze(1)) / mel_std.unsqueeze(1)
    else:
        mel = (mel - mel.mean(dim=1, keepdim=True)) / mel.std(dim=1, keepdim=True).clamp(min=1e-6)
    return mel.unsqueeze(0)  # (1, n_mels, T)


def _predict_onsets(
    model: Stage1Model,
    mel: torch.Tensor,
    bpm_signal: torch.Tensor,
    chart_constant: torch.Tensor,
    device: torch.device,
    threshold: float,
    min_gap_ms: int,
) -> List[int]:
    """Run Stage 1 and return list of onset timestamps (ms)."""
    model.eval()
    with torch.no_grad():
        logits = model(mel, chart_constant, bpm_signal)
        probs = torch.sigmoid(logits).squeeze(0)  # (T,)

    # Binary threshold
    binary = (probs > threshold).float()

    # Apply NMS with minimum gap
    if min_gap_ms > 0:
        min_gap_frames = min_gap_ms // 10
        indices = torch.nonzero(binary).squeeze(1)
        if indices.numel() > 1:
            keep = [indices[0].item()]
            for idx in indices[1:]:
                if idx.item() - keep[-1] >= min_gap_frames:
                    keep.append(idx.item())
            binary = torch.zeros_like(binary)
            for k in keep:
                binary[k] = 1.0

    onset_frames = torch.nonzero(binary).squeeze(1)
    onset_times = [f.item() * 10 for f in onset_frames]  # frame → ms
    return onset_times


def _load_stage1(checkpoint_path: str, device: torch.device) -> Stage1Model:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", {})
    model = Stage1Model(
        n_mels=cfg.get("n_mels", 80),
        cond_dim=cfg.get("cond_dim", 1),
        gate_d_model=cfg.get("gate_d_model", 512),
        use_difficulty_gate=cfg.get("use_difficulty_gate", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded Stage 1 model ({n / 1e3:.0f}k params)")
    if "epoch" in ckpt:
        print(f"  from epoch {ckpt['epoch']}, stage={ckpt.get('stage', '?')}")
    return model


def _load_stage2(checkpoint_path: str, device: torch.device) -> Stage2Model:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "config" in ckpt:
        model_cfg = Stage2Config(**ckpt["config"])
    else:
        model_cfg = Stage2Config()
    model = Stage2Model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded Stage 2 model ({n / 1e6:.1f}M params)")
    if "epoch" in ckpt:
        print(f"  from epoch {ckpt['epoch']}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage chart generation: Stage 1 onsets → Stage 2 notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Checkpoints
    parser.add_argument(
        "--stage1_checkpoint", required=True,
        help="Stage 1 onset-detector checkpoint (.pt)",
    )
    parser.add_argument(
        "--stage2_checkpoint", required=True,
        help="Stage 2 note-generator checkpoint (.pt)",
    )
    parser.add_argument("--audio", required=True, help="Input audio file (mp3/wav)")

    # BPM (mutually exclusive)
    bpm_group = parser.add_mutually_exclusive_group(required=True)
    bpm_group.add_argument("--bpm", type=float, help="Constant BPM (e.g. 175.0)")
    bpm_group.add_argument(
        "--chart_metadata",
        help="JSON with title/artist/bpm10_list for variable BPM",
    )

    parser.add_argument(
        "--constant", required=True,
        help='Target difficulty, e.g. "12,5"',
    )
    parser.add_argument("--title", help="Song title")
    parser.add_argument("--artist", help="Song artist")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", help="Save path for generated chart JSON")
    parser.add_argument(
        "--mel_stats", default="./dataset/mel_stats.pt",
        help="Precomputed mel mean/std",
    )

    # Stage 1 parameters
    parser.add_argument(
        "--onset_threshold", type=float, default=0.5,
        help="Sigmoid threshold for onset detection",
    )
    parser.add_argument(
        "--onset_min_gap", type=int, default=30,
        help="Minimum gap between onsets in ms (NMS)",
    )

    # Stage 2 parameters
    parser.add_argument(
        "--max_notes_per_onset", type=int, default=32,
        help="Max notes per onset block",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (0=greedy)",
    )
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Mel stats ──────────────────────────────────────────────────────
    stats_path = Path(args.mel_stats)
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        mel_mean, mel_std = stats["mean"].to(device), stats["std"].to(device)
        print(f"Loaded mel stats from {stats_path}")
    else:
        print("Warning: mel stats not found, using per-sample normalization")
        mel_mean = mel_std = None

    # ── Resolve BPM ────────────────────────────────────────────────────
    bpm10_list, title_meta, artist_meta = _resolve_bpm(args)
    title = args.title or title_meta or "Untitled"
    artist = args.artist or artist_meta or ""
    print(f"Title: {title}")
    if artist:
        print(f"Artist: {artist}")
    print(
        f"BPM: {bpm10_list[0]['bpm10'] / 10:.1f} "
        f"({len(bpm10_list)} change{'s' if len(bpm10_list) > 1 else ''})"
    )

    # ── Chart constant ─────────────────────────────────────────────────
    major, minor = map(int, args.constant.split(","))
    chart_const = major * 10 + minor
    print(f"Chart constant: {major}.{minor}")

    # ── Load audio → mel ───────────────────────────────────────────────
    t_load = time.perf_counter()
    mel_spec = _load_mel(args.audio, device, mel_mean, mel_std)
    print(
        f"Mel spectrogram: {mel_spec.shape}, "
        f"{mel_spec.shape[2] * 10 / 1000:.1f}s audio"
    )

    # ── BPM signal ─────────────────────────────────────────────────────
    bpm_signal = _build_bpm_signal(
        bpm10_list, mel_spec.shape[2], device
    ).unsqueeze(0)
    const_tensor = torch.tensor([chart_const], dtype=torch.long, device=device)

    # ── Load models ────────────────────────────────────────────────────
    s1 = _load_stage1(args.stage1_checkpoint, device)
    s2 = _load_stage2(args.stage2_checkpoint, device)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: onset detection
    # ══════════════════════════════════════════════════════════════════════
    print("Stage 1: detecting onsets …")
    t1 = time.perf_counter()
    onset_times = _predict_onsets(
        s1, mel_spec, bpm_signal, const_tensor, device,
        threshold=args.onset_threshold,
        min_gap_ms=args.onset_min_gap,
    )
    print(f"  Detected {len(onset_times)} onsets in {time.perf_counter() - t1:.1f}s")
    if onset_times:
        print(f"  Range: {onset_times[0]}–{onset_times[-1]} ms")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: note generation
    # ══════════════════════════════════════════════════════════════════════
    if not onset_times:
        print("Warning: no onsets detected, skipping Stage 2")
        notes = []
    else:
        print(f"Stage 2: generating notes (temperature={args.temperature}) …")
        t2 = time.perf_counter()
        validator = Stage2Validator(bpm10_list)
        with torch.no_grad():
            tokens = s2.generate(
                mel_spec,
                bpm_signal,
                const_tensor,
                onset_times_ms=onset_times,
                max_notes_per_onset=args.max_notes_per_onset,
                temperature=args.temperature,
                validator=validator,
            )
        elapsed = time.perf_counter() - t2
        print(
            f"  Generated {len(tokens)} tokens in {elapsed:.1f}s "
            f"({len(tokens) / elapsed:.0f} tok/s)"
        )

        # Decode
        tokenizer = ChartTokenizer(bpm10_list)
        try:
            notes = tokenizer.decode_tokens_stage2(tokens, onset_times)
            print(f"  Decoded {len(notes)} notes")
        except Exception as e:
            print(f"  Decode failed: {e}")
            notes = []

    # ── Total time ─────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_load
    print(f"Total: {total_elapsed:.1f}s")

    # ── Note distribution ──────────────────────────────────────────────
    if notes:
        kind_counts: dict[str, int] = {}
        for n in notes:
            kind_counts[n["kind"]] = kind_counts.get(n["kind"], 0) + 1
        print("Note distribution:")
        for kind in ("Tap", "Hold", "Slide", "Touch", "TouchHold"):
            if kind in kind_counts:
                print(f"  {kind}: {kind_counts[kind]}")

    # ── Output ─────────────────────────────────────────────────────────
    output = {
        "title": title,
        "artist": artist,
        "cabinet": "DX",
        "version": "Generated",
        "charts": [
            {
                "constant": [major, minor],
                "designer": "ChartGPT-Stage2",
                "bpm10_list": bpm10_list,
                "notes": notes,
            }
        ],
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved chart to {args.output}")
    elif notes:
        print(f"\nFirst 5 notes (of {len(notes)}):")
        for n in notes[:5]:
            print(json.dumps(n, indent=2))


if __name__ == "__main__":
    main()
