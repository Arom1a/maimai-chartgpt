"""Inference — generate a maimai chart from an audio file.

Two modes are available:

Single‑stage (original ChartGPT)::

    python main.py --checkpoint model.pt --audio track.mp3 \\
        --bpm 175.0 --constant "12,5" --title "My Song"

Two‑stage (onset detector → note generator)::

    python main.py --checkpoint checkpoints/stage2/best.pt \\
        --stage1_checkpoint checkpoints/stage1/gate_best.pt \\
        --audio track.mp3 --bpm 175.0 --constant "12,5"

When ``--stage1_checkpoint`` is provided, two‑stage mode is used:
Stage 1 detects onset timestamps; Stage 2 generates notes within each
onset block.  Otherwise the legacy single‑stage model is used.
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


def _infer_two_stage(
    args,
    device: torch.device,
    mel_mean,
    mel_std,
    bpm10_list: list[dict],
    title: str,
    artist: str,
    major: int,
    minor: int,
    chart_const: int,
) -> None:
    """Two‑stage inference: Stage 1 onsets → Stage 2 notes."""
    from src.stage1_model import Stage1Model
    from src.stage2_model import Stage2Config, Stage2Model
    from src.token_validator import Stage2Validator

    # ── Load Stage 2 model ────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "config" in checkpoint:
        cfg_d = checkpoint["config"]
        model_cfg = Stage2Config(**cfg_d)
        print(
            f"Stage2 config: d_model={cfg_d['d_model']}, "
            f"nhead={cfg_d['nhead']}, layers={cfg_d['num_decoder_layers']}"
        )
    else:
        model_cfg = Stage2Config()
        print("Warning: no config in checkpoint, using defaults")
    s2 = Stage2Model(model_cfg).to(device)
    s2.load_state_dict(checkpoint["model_state_dict"])
    s2.eval()
    n2 = sum(p.numel() for p in s2.parameters())
    print(f"Loaded Stage 2 model ({n2 / 1e6:.1f}M params)")
    if "epoch" in checkpoint:
        print(f"  epoch {checkpoint['epoch']}")

    # ── Load Stage 1 model ────────────────────────────────────────────
    s1_ckpt = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    s1_cfg = s1_ckpt.get("config", {})
    s1 = Stage1Model(
        n_mels=s1_cfg.get("n_mels", 80),
        cond_dim=s1_cfg.get("cond_dim", 1),
        gate_d_model=s1_cfg.get("gate_d_model", 512),
        use_difficulty_gate=s1_cfg.get("use_difficulty_gate", True),
    ).to(device)
    s1.load_state_dict(s1_ckpt["model_state_dict"])
    s1.eval()
    n1 = sum(p.numel() for p in s1.parameters())
    print(f"Loaded Stage 1 model ({n1 / 1e3:.0f}k params)")
    if "epoch" in s1_ckpt:
        print(f"  epoch {s1_ckpt['epoch']}, stage={s1_ckpt.get('stage', '?')}")

    # ── Audio → mel ───────────────────────────────────────────────────
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
        mel_spec = (
            (mel_spec - mel_mean.unsqueeze(1))
            / mel_std.unsqueeze(1).clamp(min=1e-6)
        )
    else:
        mel_spec = (
            (mel_spec - mel_spec.mean(dim=1, keepdim=True))
            / mel_spec.std(dim=1, keepdim=True).clamp(min=1e-6)
        )
    mel_spec = mel_spec.unsqueeze(0)
    print(
        f"Mel spectrogram: {mel_spec.shape}, "
        f"{mel_spec.shape[2] * 10 / 1000:.1f}s audio"
    )

    # ── BPM signal ─────────────────────────────────────────────────────
    bpm_signal = _build_bpm_signal(
        bpm10_list, mel_spec.shape[2], device
    ).unsqueeze(0)
    const_t = torch.tensor([chart_const], dtype=torch.long, device=device)

    # ═══════════════════════════════════════════════════════
    # Stage 1: onset detection
    # ═══════════════════════════════════════════════════════
    print("Stage 1: detecting onsets …")
    t1 = time.perf_counter()
    with torch.no_grad():
        logits = s1(mel_spec, const_t, bpm_signal)
        probs = torch.sigmoid(logits).squeeze(0)

    binary = (probs > args.onset_threshold).float()
    if args.onset_min_gap > 0:
        gap_frames = args.onset_min_gap // 10
        indices = torch.nonzero(binary).squeeze(1)
        if indices.numel() > 1:
            keep = [indices[0].item()]
            for idx in indices[1:]:
                if idx.item() - keep[-1] >= gap_frames:
                    keep.append(idx.item())
            binary = torch.zeros_like(binary)
            for k in keep:
                binary[k] = 1.0

    onset_frames = torch.nonzero(binary).squeeze(1)
    onset_times = [f.item() * 10 for f in onset_frames]
    print(
        f"  Detected {len(onset_times)} onsets "
        f"in {time.perf_counter() - t1:.1f}s"
    )
    if onset_times:
        print(f"  Range: {onset_times[0]}–{onset_times[-1]} ms")

    # ═══════════════════════════════════════════════════════
    # Stage 2: note generation
    # ═══════════════════════════════════════════════════════
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
                const_t,
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
        tokenizer = ChartTokenizer(bpm10_list)
        try:
            notes = tokenizer.decode_tokens_stage2(tokens, onset_times)
            print(f"  Decoded {len(notes)} notes")
        except Exception as e:
            print(f"  Decode failed: {e}")
            notes = []

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
        "--stage1_checkpoint",
        help="Stage 1 onset-detector checkpoint (.pt).  "
             "When provided, two‑stage inference is used.",
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
        "--onset_threshold",
        type=float,
        default=0.5,
        help="Stage 1 onset threshold (two‑stage only)",
    )
    parser.add_argument(
        "--onset_min_gap",
        type=int,
        default=30,
        help="Min gap between onsets in ms (two‑stage only)",
    )
    parser.add_argument(
        "--max_notes_per_onset",
        type=int,
        default=32,
        help="Max notes per onset block (two‑stage only)",
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

    # ── Branch: two‑stage or single‑stage ───────────────────────────────
    if args.stage1_checkpoint:
        _infer_two_stage(args, device, mel_mean, mel_std,
                         bpm10_list, title, artist,
                         major, minor, chart_const)
        return

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
    song_end_ms = mel_spec.shape[2] * 10  # each mel frame = 10 ms

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
            song_end_ms=song_end_ms,
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
