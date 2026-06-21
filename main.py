"""Inference script — generate a maimai chart from an audio file.

Two ways to supply BPM information (pick one):

1. Constant BPM::

    python main.py --checkpoint model.pt --audio track.mp3 \\
        --bpm 175.0 --constant "12,5" --title "My Song"

2. Chart metadata file (supports variable BPM)::

    python main.py --checkpoint model.pt --audio track.mp3 \\
        --chart_metadata meta.json --constant "12,5"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torchaudio

from src.dataloader import (
    MEL_HOP_LENGTH,
    MEL_WIN_LENGTH,
    N_MELS,
    TARGET_SAMPLE_RATE,
    _load_audio_ffmpeg,
)
from src.model import ChartGPT, ChartGPTConfig
from src.token_validator import ChartValidator
from src.tokenizer import ChartTokenizer


def _build_bpm_signal(
    bpm10_list: list[dict], num_frames: int, device: torch.device
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


def _load_bpm10_list(args) -> tuple[list[dict], str | None, str | None]:
    """Resolve the bpm10_list from CLI flags.

    Returns ``(bpm10_list, title, artist)``.
    """
    # ── From chart-metadata JSON ─────────────────────────────────────
    if args.chart_metadata:
        with open(args.chart_metadata) as f:
            meta = json.load(f)
        bpm10 = meta["bpm10_list"]
        title = meta.get("title")
        artist = meta.get("artist")
        if not bpm10:
            raise ValueError("chart_metadata JSON must contain 'bpm10_list'")
        return bpm10, title, artist

    # ── Constant BPM ─────────────────────────────────────────────────
    if args.bpm is not None:
        bpm10 = [{"bpm10": int(args.bpm * 10), "change_timestamp_ms": 0}]
        return bpm10, args.title, args.artist

    raise ValueError("One of --bpm or --chart_metadata is required")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a maimai chart from audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # constant BPM
  python main.py --checkpoint model.pt --audio track.mp3 \\
      --bpm 175.0 --constant "12,5" --title "My Song"

  # variable BPM from metadata file
  python main.py --checkpoint model.pt --audio track.mp3 \\
      --chart_metadata meta.json --constant "13,2\"""",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Model checkpoint (.pt)"
    )
    parser.add_argument(
        "--audio", required=True, help="Input audio file (mp3/wav)"
    )

    # BPM source (mutually exclusive)
    bpm_group = parser.add_mutually_exclusive_group(required=True)
    bpm_group.add_argument(
        "--bpm",
        type=float,
        help="Constant BPM for the whole song (e.g. 175.0)",
    )
    bpm_group.add_argument(
        "--chart_metadata",
        help="JSON file with title/artist/bpm10_list for variable BPM",
    )

    parser.add_argument(
        "--constant",
        required=True,
        help='Desired difficulty constant, e.g. "12,5"',
    )
    parser.add_argument(
        "--title",
        help="Song title (embedded in output JSON)",
    )
    parser.add_argument(
        "--artist",
        help="Song artist (embedded in output JSON)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8000,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (cuda / cpu / mps)",
    )
    parser.add_argument(
        "--output",
        help="Path to save generated chart JSON",
    )
    parser.add_argument(
        "--mel_stats",
        default="./dataset/mel_stats.pt",
        help="Path to precomputed mel mean/std",
    )
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load mel stats ──────────────────────────────────────────────────
    stats_path = Path(args.mel_stats)
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        mel_mean = stats["mean"].to(device)
        mel_std = stats["std"].to(device)
        print(f"Loaded mel stats from {stats_path}")
    else:
        print("Warning: mel stats not found, using naive normalization")
        mel_mean = None
        mel_std = None

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "config" in checkpoint:
        cfg_dict = checkpoint["config"]
        model_cfg = ChartGPTConfig(**cfg_dict)
        print(
            f"Model config: d_model={cfg_dict['d_model']}, "
            f"nhead={cfg_dict['nhead']}, layers={cfg_dict['num_decoder_layers']}"
        )
    else:
        model_cfg = ChartGPTConfig()
        print("Warning: no config in checkpoint, using defaults")
    model = ChartGPT(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({n_params / 1e6:.1f} M params)")
    if "epoch" in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs, "
              f"step {checkpoint.get('step', '?')}")

    # ── Resolve BPM & metadata ──────────────────────────────────────────
    bpm10_list, title_from_meta, artist_from_meta = _load_bpm10_list(args)
    title = args.title or title_from_meta or "Untitled"
    artist = args.artist or artist_from_meta or ""
    print(f"Title: {title}")
    if artist:
        print(f"Artist: {artist}")
    print(f"BPM: {bpm10_list[0]['bpm10']/10:.1f} "
          f"({len(bpm10_list)} change{'s' if len(bpm10_list) > 1 else ''})")

    # ── Chart constant ──────────────────────────────────────────────────
    major, minor = map(int, args.constant.split(","))
    chart_const = major * 10 + minor
    print(f"Chart constant: {major}.{minor}")

    # ── Load audio → mel spectrogram ────────────────────────────────────
    waveform = _load_audio_ffmpeg(args.audio, TARGET_SAMPLE_RATE).to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=MEL_WIN_LENGTH,
        hop_length=MEL_HOP_LENGTH,
        n_mels=N_MELS,
    ).to(device)
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-6))

    if mel_mean is not None:
        mel_spec = (mel_spec - mel_mean.unsqueeze(1)) / mel_std.unsqueeze(1)
    else:
        mel_spec = (
            (mel_spec - mel_spec.mean(dim=1, keepdim=True))
            / mel_spec.std(dim=1, keepdim=True).clamp(min=1e-6)
        )

    mel_spec = mel_spec.unsqueeze(0)  # (1, n_mels, T_spec)
    print(f"Mel spectrogram: {mel_spec.shape}, "
          f"{mel_spec.shape[2] * 10 / 1000:.1f}s audio")

    # ── BPM signal ──────────────────────────────────────────────────────
    bpm_signal = _build_bpm_signal(
        bpm10_list, mel_spec.shape[2], device
    ).unsqueeze(0)

    # ── Generate ────────────────────────────────────────────────────────
    validator = ChartValidator(bpm10_list)
    const_tensor = torch.tensor([chart_const], dtype=torch.long, device=device)

    print(f"Generating (max {args.max_tokens} tokens, "
          f"temperature={args.temperature})…")
    t0 = time.perf_counter()

    with torch.no_grad():
        tokens = model.generate(
            mel_spec,
            bpm_signal,
            const_tensor,
            max_len=args.max_tokens,
            temperature=args.temperature,
            validator=validator,
        )

    elapsed = time.perf_counter() - t0
    print(f"Generated {len(tokens)} tokens in {elapsed:.1f}s "
          f"({len(tokens) / elapsed:.0f} tok/s)")

    # ── Decode ──────────────────────────────────────────────────────────
    tokenizer = ChartTokenizer(bpm10_list)
    try:
        notes = tokenizer.decode_tokens(tokens)
        print(f"Decoded {len(notes)} notes")
    except Exception as e:
        print(f"Decode failed (sequence may be truncated or invalid): {e}")
        notes = []

    # ── Print summary ───────────────────────────────────────────────────
    if notes:
        kind_counts: dict[str, int] = {}
        for n in notes:
            kind_counts[n["kind"]] = kind_counts.get(n["kind"], 0) + 1
        print("Note distribution:")
        for kind in ("Tap", "Hold", "Slide", "Touch", "TouchHold"):
            if kind in kind_counts:
                print(f"  {kind}: {kind_counts[kind]}")
    else:
        print("No notes decoded (truncated or invalid generation)")

    # ── Save output ─────────────────────────────────────────────────────
    output = {
        "title": title,
        "artist": artist,
        "cabinet": "DX",
        "version": "Generated",
        "charts": [
            {
                "constant": [major, minor],
                "designer": "ChartGPT",
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
