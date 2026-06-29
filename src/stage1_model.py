from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# OnsetDetector — difficulty-agnostic CNN
# ═══════════════════════════════════════════════════════════════════════════════


class OnsetDetector(nn.Module):
    """Lightweight 1-D CNN that predicts per-frame onset logits from mel.

    Parameters
    ----------
    n_mels : int
        Number of mel bands (default 80).
    cond_dim : int
        Extra conditioning channels appended to mel input (e.g. 1 for BPM).
    """

    def __init__(self, n_mels: int = 80, cond_dim: int = 0):
        super().__init__()
        in_ch = n_mels + cond_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mel: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """*mel*: ``(B, n_mels, T)``, *cond*: ``(B, cond_dim, T)`` or None.
        Returns ``(B, T)`` per-frame onset logits.
        """
        if cond is not None:
            mel = torch.cat([mel, cond], dim=1)
        return self.cnn(mel).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# DifficultyGate — learned threshold conditioned on chart constant
# ═══════════════════════════════════════════════════════════════════════════════


class DifficultyGate(nn.Module):
    """Maps chart-constant index to a scalar bias added to onset logits.

    Parameters
    ----------
    d_model : int
        Embedding dimension (default 512).
    max_const : int
        Maximum chart constant index (default 200 → 0..199).
    """

    def __init__(self, d_model: int = 512, max_const: int = 200):
        super().__init__()
        self.cc_embed = nn.Embedding(max_const, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, chart_constant: torch.Tensor) -> torch.Tensor:
        """*chart_constant*: ``(B,)`` int indices.
        Returns ``(B,)`` scalar bias.
        """
        emb = self.cc_embed(chart_constant)  # (B, d_model)
        bias = self.mlp(emb) * 5.0
        return bias.squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage1Model — combined onset detector with optional difficulty gating
# ═══════════════════════════════════════════════════════════════════════════════


class Stage1Model(nn.Module):
    """Onset detection model with learned difficulty-dependent threshold.

    Wraps an ``OnsetDetector`` and an optional ``DifficultyGate``.
    Provides both deterministic ``prob()`` and straight-through Gumbel-Softmax
    ``sample_binary()`` for training.
    """

    def __init__(
        self,
        n_mels: int = 80,
        cond_dim: int = 1,
        gate_d_model: int = 512,
        use_difficulty_gate: bool = True,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.cond_dim = cond_dim
        self.use_difficulty_gate = use_difficulty_gate

        self.detector = OnsetDetector(n_mels=n_mels, cond_dim=cond_dim)
        if use_difficulty_gate:
            self.gate = DifficultyGate(d_model=gate_d_model)

    def _make_cond(self, bpm_signal: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Reshape *bpm_signal* to ``(B, 1, T)`` if ``cond_dim`` > 0."""
        if bpm_signal is not None and self.cond_dim > 0:
            return bpm_signal.unsqueeze(1)
        return None

    def forward(
        self,
        mel: torch.Tensor,
        chart_constant: Optional[torch.Tensor] = None,
        bpm_signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-frame onset logits ``(B, T)``.

        If *use_difficulty_gate* is True and *chart_constant* is provided,
        a scalar bias is added to every frame.
        """
        cond = self._make_cond(bpm_signal)
        logits = self.detector(mel, cond)  # (B, T)
        if self.use_difficulty_gate and chart_constant is not None:
            bias = self.gate(chart_constant)  # (B,)
            logits = logits + bias.unsqueeze(1)
        return logits

    @torch.no_grad()
    def prob(
        self,
        mel: torch.Tensor,
        chart_constant: Optional[torch.Tensor] = None,
        bpm_signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deterministic onset probability ``(B, T)`` — for eval / inference."""
        return torch.sigmoid(self.forward(mel, chart_constant, bpm_signal))

    def sample_binary(
        self,
        mel: torch.Tensor,
        chart_constant: Optional[torch.Tensor] = None,
        bpm_signal: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Straight-through Gumbel-Softmax binary mask ``(B, T)``.

        Gradients flow through the soft sigmoid (backward) while the forward
        pass uses a hard 0/1 threshold.  This makes onset selection
        differentiable for end-to-end training with Stage 2.
        """
        logits = self.forward(mel, chart_constant, bpm_signal)
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y_soft = torch.sigmoid((logits + gumbels) / temperature)
        y_hard = (y_soft > 0.5).float()
        return (y_hard - y_soft).detach() + y_soft
