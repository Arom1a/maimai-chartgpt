from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tokenizer import (
    CC,
    END_ONSET,
    EOS,
    ONSET,
    PAD,
    SOS,
    STAGE2_VOCAB_SIZE,
    compute_abs_times_stage2,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Stage2Config:
    d_model: int = 512
    nhead: int = 8
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    n_mels: int = 80
    max_abs_time: float = 300.0
    enable_checkpointing: bool = True
    encoder_num_layers: int = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Sinusoidal embeddings
# ═══════════════════════════════════════════════════════════════════════════════


class SinusoidalTimeEmbedding(nn.Module):
    """Maps a continuous scalar time (seconds) to a *d_model*-dim vector."""

    def __init__(self, d_model: int, max_time: float = 300.0):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        freqs = torch.logspace(
            math.log10(1.0 / max_time), math.log10(20.0), half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t * self.freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SinusoidalPositionalEncoding(nn.Module):
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
        return self.pe[positions]


# ═══════════════════════════════════════════════════════════════════════════════
# Stage2Encoder — CNN (100→50 Hz) + FiLM + Transformer
# ═══════════════════════════════════════════════════════════════════════════════


class Stage2Encoder(nn.Module):
    """CNN downsampler + FiLM difficulty conditioning + Transformer encoder."""

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

        # FiLM: γ = 1 + tanh(Wg·cc), β = Wb·cc
        self.cc_embed = nn.Embedding(200, d_model)
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # FiLM gamma should start near 1 (identity)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(
        self,
        x: torch.Tensor,
        chart_constant: torch.Tensor,
    ) -> torch.Tensor:
        """*x*: ``(B, n_mels, T_100Hz)`` → ``(B, T_50Hz, d_model)``."""
        x = self.cnn(x)  # (B, d_model, T_50Hz)
        x = x.transpose(1, 2)  # (B, T_50Hz, d_model)

        # FiLM modulation
        cc = self.cc_embed(chart_constant)  # (B, d_model)
        gamma = 1.0 + torch.tanh(self.film_gamma(cc)).unsqueeze(1)  # (B,1,d_model)
        beta = self.film_beta(cc).unsqueeze(1)  # (B,1,d_model)
        x = gamma * x + beta

        x = self.transformer(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Stage2Decoder — Transformer decoder with ONSET time embedding
# ═══════════════════════════════════════════════════════════════════════════════


class Stage2Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = STAGE2_VOCAB_SIZE,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_abs_time: float = 300.0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.time_emb = SinusoidalTimeEmbedding(d_model, max_time=max_abs_time)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.output_head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt_tokens: torch.Tensor,  # (B, L)
        abs_times: torch.Tensor,  # (B, L, 1)
        memory: torch.Tensor,  # (B, T_enc, d_model)
        *,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``(B, L, d_model)`` decoder hidden states."""
        L = tgt_tokens.shape[1]

        tok_emb = self.token_emb(tgt_tokens)  # (B, L, d_model)
        positions = torch.arange(L, device=tgt_tokens.device).unsqueeze(0)
        pos_emb = self.pos_enc(positions)  # (1, L, d_model)
        time_emb = self.time_emb(abs_times)  # (B, L, d_model)

        tgt_emb = tok_emb + pos_emb + time_emb  # (B, L, d_model)

        return self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage2Model
# ═══════════════════════════════════════════════════════════════════════════════


class Stage2Model(nn.Module):
    def __init__(self, config: Optional[Stage2Config] = None):
        super().__init__()
        config = config or Stage2Config()
        self.config = config

        self.encoder = Stage2Encoder(
            n_mels=config.n_mels,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.encoder_num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
        self.bpm_proj = nn.Linear(1, config.d_model)

        self.decoder = Stage2Decoder(
            vocab_size=STAGE2_VOCAB_SIZE,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_abs_time=config.max_abs_time,
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _decode(
        self,
        dec_input: torch.Tensor,
        dec_abs_times: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        _unused: None,
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Positional‑arg wrapper so ``torch.utils.checkpoint`` can call it."""
        return self.decoder(
            tgt_tokens=dec_input,
            abs_times=dec_abs_times,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def _encode(
        self,
        spectrogram: torch.Tensor,
        bpm_signal: torch.Tensor,
        chart_constant: torch.Tensor,
    ) -> torch.Tensor:
        enc_out = self.encoder(spectrogram, chart_constant)  # (B, T_enc, d_model)
        T_enc = enc_out.shape[1]

        bpm_cond = self.bpm_proj(bpm_signal.unsqueeze(-1))  # (B, T_spec, d_model)
        bpm_cond = F.interpolate(
            bpm_cond.transpose(1, 2),
            size=T_enc,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        enc_out = enc_out + bpm_cond
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

        Returns ``(dec_out, targets)``
        where *dec_out* is ``(B, L-1, d_model)`` unprojected decoder
        output and *targets* is ``(B, L-1)`` shifted one position right.
        """
        memory = self._encode(spectrogram, bpm_signal, chart_constant)

        memory_key_padding_mask: Optional[torch.Tensor] = None
        if spec_mask is not None:
            T_enc = memory.shape[1]
            mask_float = (~spec_mask).float().unsqueeze(1)
            mask_float = F.interpolate(mask_float, size=T_enc, mode="nearest")
            memory_key_padding_mask = mask_float.squeeze(1) < 0.5

        dec_input = target_tokens[:, :-1]  # (B, L-1)
        dec_target = target_tokens[:, 1:]  # (B, L-1)
        dec_abs_times = target_abs_times[:, :-1]  # (B, L-1)

        tgt_key_padding_mask: Optional[torch.Tensor] = None
        if tok_mask is not None:
            tgt_key_padding_mask = tok_mask[:, :-1]

        tgt_mask = torch.triu(
            torch.ones(
                dec_input.shape[1],
                dec_input.shape[1],
                device=dec_input.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        if self.training and self.config.enable_checkpointing:
            dec_out = torch.utils.checkpoint.checkpoint(
                self._decode,
                dec_input,
                dec_abs_times.unsqueeze(-1),
                memory,
                tgt_mask,
                None,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                use_reentrant=False,
            )
        else:
            dec_out = self.decoder(
                tgt_tokens=dec_input,
                abs_times=dec_abs_times.unsqueeze(-1),
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return dec_out, dec_target

    @torch.no_grad()
    def generate(
        self,
        spectrogram: torch.Tensor,
        bpm_signal: torch.Tensor,
        chart_constant: torch.Tensor,
        *,
        onset_times_ms: List[int],
        max_notes_per_onset: int = 32,
        temperature: float = 1.0,
        validator=None,
        song_end_ms: Optional[float] = None,
    ) -> List[int]:
        """Autoregressively generate a stage‑2 token sequence.

        The onset timestamps are injected from Stage 1.  The model generates
        note tokens within each onset block until ``END_ONSET``, then
        terminates with ``EOS`` after the last block.

        Parameters
        ----------
        onset_times_ms : list[int]
            Absolute onset timestamps (ms) from Stage 1.
        max_notes_per_onset : int
            Safety cap on notes produced per onset block.
        """
        device = spectrogram.device
        if spectrogram.shape[0] != 1:
            raise ValueError("generate() supports batch-size 1 only")

        memory = self._encode(spectrogram, bpm_signal, chart_constant)

        generated: List[int] = [SOS, CC]

        if validator is not None:
            validator.advance(SOS)
            validator.advance(CC)

        for onset_ms in onset_times_ms:
            generated.append(ONSET)
            if validator is not None:
                validator.advance(ONSET)

            note_count = 0
            while note_count < max_notes_per_onset:
                L = len(generated)
                abs_times_raw = compute_abs_times_stage2(generated, onset_times_ms)
                abs_times = torch.tensor(
                    abs_times_raw, device=device, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)

                tgt_tensor = torch.tensor(
                    [generated], device=device, dtype=torch.long
                )
                tgt_mask = torch.triu(
                    torch.ones(L, L, device=device, dtype=torch.bool),
                    diagonal=1,
                )

                dec_out = self.decoder(
                    tgt_tokens=tgt_tensor,
                    abs_times=abs_times,
                    memory=memory,
                    tgt_mask=tgt_mask,
                )
                logits = self.decoder.output_head(
                    dec_out[:, -1, :]
                ).squeeze(0)  # (V,)

                # Apply validator mask
                if validator is not None:
                    mask = validator.valid_mask(device=device)
                    if song_end_ms is not None:
                        cur_time = (
                            onset_ms if abs_times_raw[-1] >= onset_ms / 1000.0 else 0.0
                        )
                        if onset_ms >= song_end_ms and mask[EOS]:
                            mask = torch.zeros_like(mask)
                            mask[EOS] = True
                    if mask.any():
                        logits[~mask] = float("-inf")

                if temperature <= 0.0:
                    next_token = torch.argmax(logits).item()
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                generated.append(next_token)
                if validator is not None:
                    validator.advance(next_token)
                note_count += 1

                if next_token == END_ONSET:
                    break
                if next_token == EOS:
                    if validator is not None:
                        validator.advance(EOS)
                    return generated

        # All onsets processed — force EOS
        if generated[-1] != EOS:
            generated.append(EOS)
            if validator is not None:
                validator.advance(EOS)

        return generated
