from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tokenizer import (
    EOS,
    PAD,
    SOS,
    VOCAB_SIZE,
    decode_time_token,
    encode_time_token,
    is_time_token,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ChartGPTConfig:
    d_model: int = 512
    nhead: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    n_mels: int = 80
    max_abs_time: float = 300.0  # max song duration for time embedding scale
    enable_checkpointing: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Sinusoidal time embedding
# ═══════════════════════════════════════════════════════════════════════════════


class SinusoidalTimeEmbedding(nn.Module):
    """Maps a continuous scalar time (seconds) to a ``d_model``-dim vector."""

    def __init__(self, d_model: int, max_time: float = 300.0):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        # Frequencies spaced from 1/max_time to 20 Hz so they cover the
        # typical range of musical events (0–~200 Hz)
        freqs = torch.logspace(math.log10(1.0 / max_time), math.log10(20.0), half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """*t*: ``(..., 1)``  →  ``(..., d_model)``"""
        angles = t * self.freqs  # (..., half)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal position encoding."""

    def __init__(self, d_model: int, max_len: int = 16384):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """*positions*: ``(...,)`` → ``(..., d_model)``"""
        return self.pe[positions]


# ═══════════════════════════════════════════════════════════════════════════════
# Audio encoder
# ═══════════════════════════════════════════════════════════════════════════════


class AudioEncoder(nn.Module):
    """Conv1d stack that downsamples mel spectrograms to ~12.5 Hz."""

    def __init__(self, n_mels: int = 128, d_model: int = 512):
        super().__init__()
        self.blocks = nn.Sequential(
            # Block 1: 100 Hz, 128 → 256
            nn.Conv1d(n_mels, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # Block 2: 50 Hz, 256 → 512
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # Block 3: 25 Hz, 512 → 512
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # Block 4: 12.5 Hz, 512 → d_model
            nn.Conv1d(512, d_model, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """*x*: ``(B, n_mels, T)`` → ``(B, T_enc, d_model)``"""
        x = self.blocks(x)
        return x.transpose(1, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# ChartGPT
# ═══════════════════════════════════════════════════════════════════════════════


class ChartGPT(nn.Module):
    def __init__(self, config: Optional[ChartGPTConfig] = None):
        super().__init__()
        config = config or ChartGPTConfig()
        self.config = config

        # ── Encoder ────────────────────────────────────────────────────
        self.audio_encoder = AudioEncoder(n_mels=config.n_mels, d_model=config.d_model)
        self.bpm_proj = nn.Linear(1, config.d_model)
        self.const_emb = nn.Embedding(200, config.d_model)  # 0..199 covers 0.0-19.9

        # ── Decoder ────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.abs_time_emb = SinusoidalTimeEmbedding(
            config.d_model, max_time=config.max_abs_time
        )
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers
        )

        self.output_head = nn.Linear(config.d_model, VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode(
        self,
        spectrogram: torch.Tensor,
        bpm_signal: torch.Tensor,
        chart_constant: torch.Tensor,
    ) -> torch.Tensor:
        """Encode audio + conditioning into a memory tensor.

        Returns
        -------
        ``(B, T_enc, d_model)``
        """
        # Audio features
        enc_out = self.audio_encoder(spectrogram)  # (B, T_enc, d_model)
        T_enc = enc_out.shape[1]

        # BPM conditioning – interpolate from 100 Hz → encoder frame rate
        bpm_cond = self.bpm_proj(bpm_signal.unsqueeze(-1))  # (B, T_spec, d_model)
        bpm_cond = F.interpolate(
            bpm_cond.transpose(1, 2),
            size=T_enc,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        enc_out = enc_out + bpm_cond

        # Chart constant
        const_emb = self.const_emb(chart_constant)  # (B, d_model)
        enc_out = enc_out + const_emb.unsqueeze(1)

        return enc_out

    def forward(
        self,
        spectrogram: torch.Tensor,
        bpm_signal: torch.Tensor,
        chart_constant: torch.Tensor,
        target_tokens: torch.Tensor,
        target_abs_times: torch.Tensor,
        *,
        spec_mask: Optional[torch.Tensor] = None,
        tok_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing forward pass.

        Parameters
        ----------
        spectrogram : ``(B, n_mels, T_spec)``
        bpm_signal : ``(B, T_spec)``
        chart_constant : ``(B,)``
        target_tokens : ``(B, L)`` — full token sequence including SOS/EOS.
        target_abs_times : ``(B, L)`` — absolute song time (seconds) per token.
        spec_mask : ``(B, T_spec)`` — ``True`` where padded.
        tok_mask : ``(B, L)`` — ``True`` where padded.

        Returns
        -------
        ``(dec_out, targets)``
            dec_out: ``(B, L-1, d_model)`` — raw decoder output (no projection).
            targets: ``(B, L-1)`` — shifted right by 1 from *target_tokens*.

        Notes
        -----
        The output projection to vocabulary size is **not** applied here.
        Call :meth:`compute_logits_chunk` or iterate over
        ``model.output_head(dec_out[:, start:end])`` in chunks to avoid
        materialising the giant ``(B, L, 13267)`` tensor on GPU.
        """
        B = spectrogram.shape[0]

        # ── Encode ─────────────────────────────────────────────────────
        memory = self._encode(spectrogram, bpm_signal, chart_constant)
        # (B, T_enc, d_model)

        # Build encoder padding mask from spec_mask (if provided)
        memory_key_padding_mask: Optional[torch.Tensor] = None
        if spec_mask is not None:
            T_enc = memory.shape[1]
            mask_float = (~spec_mask).float().unsqueeze(1)  # (B, 1, T_spec)
            mask_float = F.interpolate(mask_float, size=T_enc, mode="nearest")
            memory_key_padding_mask = mask_float.squeeze(1) < 0.5  # (B, T_enc)

        # ── Decoder input ──────────────────────────────────────────────
        dec_input = target_tokens[:, :-1]  # (B, L-1)
        dec_target = target_tokens[:, 1:]  # (B, L-1)
        dec_abs_times = target_abs_times[:, :-1]  # (B, L-1)

        # Embeddings
        tok_emb = self.token_emb(dec_input)  # (B, L-1, d_model)
        time_emb = self.abs_time_emb(dec_abs_times.unsqueeze(-1))  # (B, L-1, d_model)
        positions = torch.arange(dec_input.shape[1], device=dec_input.device).unsqueeze(0)
        pos_emb = self.pos_enc(positions)  # (1, L-1, d_model)

        tgt_emb = tok_emb + time_emb + pos_emb  # (B, L-1, d_model)

        # Decoder padding mask
        tgt_key_padding_mask: Optional[torch.Tensor] = None
        if tok_mask is not None:
            tgt_key_padding_mask = tok_mask[:, :-1]

        # Explicit causal bool mask.  PyTorch SDPA detects the triangular
        # pattern and dispatches to Flash Attention / Memory-Efficient
        # Attention on CUDA without materialising the full (L, L) matrix.
        tgt_mask = torch.triu(
            torch.ones(dec_input.shape[1], dec_input.shape[1],
                       device=dec_input.device, dtype=torch.bool),
            diagonal=1,
        )

        # ── Decode ─────────────────────────────────────────────────────
        if self.training and self.config.enable_checkpointing:
            dec_out = torch.utils.checkpoint.checkpoint(
                self.decoder,
                tgt_emb,
                memory,
                tgt_mask,
                None,  # memory_mask
                tgt_key_padding_mask,
                memory_key_padding_mask,
                None,  # tgt_is_causal
                False,  # memory_is_causal
                use_reentrant=False,
            )
        else:
            dec_out = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        # dec_out: (B, L-1, d_model)

        return dec_out, dec_target

    @torch.no_grad()
    def generate(
        self,
        spectrogram: torch.Tensor,
        bpm_signal: torch.Tensor,
        chart_constant: torch.Tensor,
        *,
        max_len: int = 8000,
        temperature: float = 1.0,
    ) -> List[int]:
        """Autoregressively generate a token sequence for one song.

        Parameters
        ----------
        spectrogram : ``(1, n_mels, T_spec)``  (batch size 1)
        bpm_signal : ``(1, T_spec)``
        chart_constant : ``(1,)``
        max_len : int
            Maximum number of tokens to generate.
        temperature : float
            Softmax temperature.  ``0.0`` → greedy argmax.

        Returns
        -------
        ``list[int]`` — token IDs including ``SOS`` and ``EOS``.
        """
        device = spectrogram.device
        if spectrogram.shape[0] != 1:
            raise ValueError("generate() supports batch-size 1 only")

        # Encode once
        memory = self._encode(spectrogram, bpm_signal, chart_constant)
        # (1, T_enc, d_model) → (T_enc, 1, d_model) for decoder

        generated = [SOS]

        for _ in range(max_len):
            L = len(generated)
            tgt_tensor = torch.tensor([generated], device=device, dtype=torch.long)

            # Compute absolute times for all positions
            cur = 0.0
            abs_times = torch.zeros(1, L, 1, device=device)
            for i, tok in enumerate(generated):
                if is_time_token(tok):
                    cur += decode_time_token(tok) / 1000.0
                abs_times[0, i, 0] = cur

            # Embeddings
            tok_emb = self.token_emb(tgt_tensor)
            time_emb = self.abs_time_emb(abs_times)
            positions = torch.arange(L, device=device).unsqueeze(0)
            pos_emb = self.pos_enc(positions)

            tgt_emb = tok_emb + time_emb + pos_emb

            tgt_mask = torch.triu(
                torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
            )
            dec_out = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
            )

            # Last position logits
            logits = self.output_head(dec_out[:, -1, :]).squeeze(0)  # (V,)

            if temperature <= 0.0:
                next_token = torch.argmax(logits).item()
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)

            if next_token == EOS:
                break

        return generated
