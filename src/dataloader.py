from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.tokenizer import (
    PAD,
    VOCAB_SIZE,
    ChartTokenizer,
    decode_time_token,
    encode_time_token,
    is_time_token,
)


# ── Audio processing constants ────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 16000
MEL_HOP_LENGTH = 160  # 10 ms at 16 kHz → 100 Hz
MEL_WIN_LENGTH = 512  # 32 ms window
N_MELS = 80
ENC_STRIDE = 2             # 1 × stride-2 conv block → 50 Hz


def _load_audio_ffmpeg(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    """Decode an audio file to mono PCM via ffmpeg, returning a 1-D tensor."""
    cmd = [
        "ffmpeg",
        "-v",
        "quiet",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(samples.copy())


def compute_abs_times(tokens: List[int]) -> List[float]:
    """Compute absolute song time (seconds) for each decoder-input position.

    Returns a list of the same length as *tokens* where ``times[i]`` is
    the cumulative song time **after** processing ``tokens[i]`` (i.e. the
    time that the decoder will use as the absolute-time embedding for the
    **next** step).
    """
    times: List[float] = []
    cur = 0.0
    for tok in tokens:
        if is_time_token(tok):
            cur += decode_time_token(tok) / 1000.0
        times.append(cur)
    return times


# ═══════════════════════════════════════════════════════════════════════════════
# MaiMaiDataset
# ═══════════════════════════════════════════════════════════════════════════════


class MaiMaiDataset(Dataset):
    """Loads audio + chart pairs and produces model-ready tensors.

    Parameters
    ----------
    data_dir : str or Path
        Root of the ``dataset/`` tree containing ``processed.json`` and
        ``track.mp3`` files.
    split : str
        ``"train"`` or ``"val"``.
    val_fraction : float
        Fraction of songs reserved for validation (default 0.1).
    mel_stats : Optional[tuple]
        ``(mean, std)`` tensors of shape ``(N_MELS,)`` for per-channel
        standardisation.  If ``None`` the values are computed lazily
        (slower on first epoch).
    max_tokens : int or None
        If set, charts that tokenize to more than *max_tokens* tokens are
        skipped.  This keeps outlier charts from blowing up batch memory.
        Default ``8192`` covers ~99 % of the dataset.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        mel_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_tokens: Optional[int] = 8192,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.val_fraction = val_fraction

        # Gather all (processed_json_path, chart_index) samples
        self.samples: List[Tuple[Path, int]] = []
        for json_path in sorted(self.data_dir.rglob("processed.json")):
            with open(json_path, "rb") as f:
                data = json.load(f)
            title_hash = hash(data["title"])  # deterministic across runs
            for ci in range(len(data["charts"])):
                if data["charts"][ci]["notes"]:
                    self.samples.append((json_path, ci))

        # Split by song (hash mod) so charts from the same song stay together
        train_samples: List[Tuple[Path, int]] = []
        val_samples: List[Tuple[Path, int]] = []
        for json_path, ci in self.samples:
            with open(json_path, "rb") as f:
                data = json.load(f)
            bucket = abs(hash(data["title"])) % 100
            if bucket < int(self.val_fraction * 100):
                val_samples.append((json_path, ci))
            else:
                train_samples.append((json_path, ci))

        if split == "train":
            self.samples = train_samples
        else:
            self.samples = val_samples

        # Filter by estimated token length (fast pre-check using note count)
        if max_tokens is not None:
            filtered: List[Tuple[Path, int]] = []
            for json_path, ci in self.samples:
                with open(json_path, "rb") as f:
                    data = json.load(f)
                # Rough estimate: ~6.4 tokens per note + 2 for SOS/EOS
                est_tokens = len(data["charts"][ci]["notes"]) * 7 + 2
                if est_tokens <= max_tokens:
                    filtered.append((json_path, ci))
            self.samples = filtered

        # Mel stats
        if mel_stats is not None:
            self._mel_mean, self._mel_std = mel_stats
        else:
            self._mel_mean: Optional[torch.Tensor] = None
            self._mel_std: Optional[torch.Tensor] = None

        # Lazy-init the mel transform (resample cache)
        self._mel_transform: Optional[torchaudio.transforms.MelSpectrogram] = None

    def _get_mel_transform(
        self, device: torch.device
    ) -> torchaudio.transforms.MelSpectrogram:
        if self._mel_transform is None:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=TARGET_SAMPLE_RATE,
                n_fft=MEL_WIN_LENGTH,
                hop_length=MEL_HOP_LENGTH,
                n_mels=N_MELS,
            ).to(device)
        return self._mel_transform

    @staticmethod
    def load_audio(path: str | Path) -> torch.Tensor:
        """Return mono 16 kHz waveform as 1-D tensor."""
        return _load_audio_ffmpeg(path, TARGET_SAMPLE_RATE)

    def _compute_bpm_signal(
        self, bpm10_list: List[Dict], num_frames: int
    ) -> torch.Tensor:
        """BPM (not bpm10) at each mel frame (100 Hz)."""
        signal = torch.zeros(num_frames, dtype=torch.float32)
        change_pairs = sorted(
            [(r["change_timestamp_ms"], r["bpm10"] / 10.0) for r in bpm10_list],
            key=lambda x: x[0],
        )
        frame_time_ms = 0.0
        frame_step_ms = MEL_HOP_LENGTH / TARGET_SAMPLE_RATE * 1000.0  # 10 ms

        ci = 0
        cur_bpm = change_pairs[0][1] if change_pairs else 150.0
        for fi in range(num_frames):
            while ci < len(change_pairs) and change_pairs[ci][0] <= frame_time_ms:
                cur_bpm = change_pairs[ci][1]
                ci += 1
            signal[fi] = cur_bpm
            frame_time_ms += frame_step_ms
        return signal

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        json_path, chart_idx = self.samples[idx]
        mp3_path = json_path.parent / "track.mp3"

        if not mp3_path.exists():
            raise FileNotFoundError(f"Audio not found: {mp3_path}")

        # Device defaults to CPU for DataLoader workers
        device = torch.device("cpu")

        with open(json_path, "rb") as f:
            song_data = json.load(f)
        chart = song_data["charts"][chart_idx]

        # ── Audio → mel spectrogram ────────────────────────────────────────
        waveform = self.load_audio(str(mp3_path))
        waveform = waveform.to(device)
        mel_transform = self._get_mel_transform(device)
        mel_spec = mel_transform(waveform)  # (n_mels, T_spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-6))  # log-mel

        # Standardise per channel
        if self._mel_mean is None:
            self._mel_mean = mel_spec.mean(dim=1)
            self._mel_std = mel_spec.std(dim=1).clamp(min=1e-6)
        mel_spec = (mel_spec - self._mel_mean.unsqueeze(1)) / self._mel_std.unsqueeze(1)
        T_spec = mel_spec.shape[1]

        # ── BPM signal ─────────────────────────────────────────────────────
        bpm_signal = self._compute_bpm_signal(chart["bpm10_list"], T_spec)

        # ── Chart constant ─────────────────────────────────────────────────
        major, minor = chart["constant"]
        chart_const = major * 10 + minor

        # ── Tokenize ───────────────────────────────────────────────────────
        tokenizer = ChartTokenizer(chart["bpm10_list"])
        tokens = tokenizer.encode_notes(chart["notes"])

        # ── Absolute times for decoder ─────────────────────────────────────
        abs_times_raw = compute_abs_times(tokens)
        abs_times = torch.tensor(abs_times_raw, dtype=torch.float32)

        # ── Onset labels (Stage 1 training target) ────────────────────────
        onset_labels = torch.zeros(T_spec, dtype=torch.float32)
        for note in chart["notes"]:
            frame = round(note["timestamp_ms"] / 10.0)
            for f in range(max(0, frame - 1), min(T_spec, frame + 2)):
                onset_labels[f] = 1.0

        return {
            "spectrogram": mel_spec,  # (n_mels, T_spec)
            "bpm_signal": bpm_signal,  # (T_spec,)
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "chart_constant": torch.tensor(chart_const, dtype=torch.long),
            "abs_times": abs_times,  # (len(tokens),)
            "onset_labels": onset_labels,  # (T_spec,)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Collate
# ═══════════════════════════════════════════════════════════════════════════════


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad all sequences in *batch* to the maximum length in the batch."""
    n_mels = batch[0]["spectrogram"].shape[0]

    max_spec_len = max(item["spectrogram"].shape[1] for item in batch)
    max_tok_len = max(item["tokens"].shape[0] for item in batch)

    specs = []
    bpms = []
    tokens = []
    consts = []
    ts = []
    onsets = []
    spec_masks = []
    tok_masks = []

    for item in batch:
        sl = item["spectrogram"].shape[1]
        tl = item["tokens"].shape[0]

        spec_pad = max_spec_len - sl
        tok_pad = max_tok_len - tl

        specs.append(F.pad(item["spectrogram"], (0, spec_pad)))
        bpms.append(F.pad(item["bpm_signal"], (0, spec_pad)))
        tokens.append(F.pad(item["tokens"], (0, tok_pad), value=PAD))
        ts.append(F.pad(item["abs_times"], (0, tok_pad)))
        onsets.append(F.pad(item["onset_labels"], (0, spec_pad)))
        consts.append(item["chart_constant"])

        spec_masks.append(
            torch.cat(
                [
                    torch.zeros(sl, dtype=torch.bool),
                    torch.ones(spec_pad, dtype=torch.bool),
                ]
            )
        )
        tok_masks.append(
            torch.cat(
                [
                    torch.zeros(tl, dtype=torch.bool),
                    torch.ones(tok_pad, dtype=torch.bool),
                ]
            )
        )

    return {
        "spectrogram": torch.stack(specs),  # (B, n_mels, T_spec_max)
        "bpm_signal": torch.stack(bpms),  # (B, T_spec_max)
        "tokens": torch.stack(tokens),  # (B, L_max)
        "chart_constant": torch.stack(consts),  # (B,)
        "abs_times": torch.stack(ts),  # (B, L_max)
        "onset_labels": torch.stack(onsets),  # (B, T_spec_max)
        "spec_mask": torch.stack(spec_masks),  # (B, T_spec_max)
        "tok_mask": torch.stack(tok_masks),  # (B, L_max)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_mel_stats(
    data_dir: str | Path, max_files: int = 200, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean and std of log-mel spectrograms.

    Parameters
    ----------
    data_dir : str or Path
        Root of the dataset tree.
    max_files : int
        Maximum number of audio files to process (enough for stable stats).
    device : str
        Torch device.

    Returns
    -------
    (mean, std) : each shape ``(N_MELS,)``
    """
    data_dir = Path(data_dir)
    mp3_files = sorted(data_dir.rglob("track.mp3"))[:max_files]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=MEL_WIN_LENGTH,
        hop_length=MEL_HOP_LENGTH,
        n_mels=N_MELS,
    ).to(device)

    sum_mel = torch.zeros(N_MELS, device=device)
    sum_mel_sq = torch.zeros(N_MELS, device=device)
    total_frames = 0

    for mp3_path in tqdm(mp3_files, desc="Computing mel stats"):
        waveform = _load_audio_ffmpeg(str(mp3_path), TARGET_SAMPLE_RATE).to(device)
        mel = mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-6))
        sum_mel += mel.sum(dim=1)
        sum_mel_sq += (mel**2).sum(dim=1)
        total_frames += mel.shape[1]

    mean = sum_mel / total_frames
    std = torch.sqrt(sum_mel_sq / total_frames - mean**2).clamp(min=1e-6)
    return mean.cpu(), std.cpu()
