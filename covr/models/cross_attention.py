"""Cross-attention fusion model for video-text retrieval."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


def _maybe_proj(in_dim: int, out_dim: int) -> nn.Module:
    """Identity when dims match (no compression needed), Linear otherwise."""
    return nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)


def _masked_mean(x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean pool over dim=1, ignoring padded positions.

    x:        [B, N, D]
    pad_mask: [B, N] bool, True = padding
    returns:  [B, D]
    """
    if pad_mask is None:
        return x.mean(dim=1)
    valid = (~pad_mask).float().unsqueeze(-1)  # [B, N, 1]
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


class _SelfAttentionBlock(nn.Module):
    """Pre-LN transformer block, video tokens attend to each other."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, key_padding_mask=pad_mask, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class _CrossAttentionBlock(nn.Module):
    """Pre-LN transformer block: Q=video patches, K/V=text tokens."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv = self.norm_kv(context)
        x = (
            x
            + self.attn(
                self.norm_q(x), kv, kv, key_padding_mask=kv_mask, need_weights=False
            )[0]
        )
        x = x + self.ffn(self.norm2(x))
        return x


@dataclass
class FusionConfig:
    video_dim: int = 768
    text_dim: int = 1024
    d_model: int = 768
    nhead: int = 8
    ffn_dim: int = 2048
    embed_dim: int = 768
    dropout: float = 0.1
    num_self_attn: int = 2
    num_cross_attn: int = 2


class CrossAttentionFusion(nn.Module):
    def __init__(self, config: FusionConfig):
        super().__init__()
        cfg = config
        self.proj_video = _maybe_proj(cfg.video_dim, cfg.d_model)
        self.proj_text = _maybe_proj(cfg.text_dim, cfg.d_model)

        self.self_attn = nn.ModuleList(
            [
                _SelfAttentionBlock(cfg.d_model, cfg.nhead, cfg.ffn_dim, cfg.dropout)
                for _ in range(cfg.num_self_attn)
            ]
        )
        self.cross_attn = nn.ModuleList(
            [
                _CrossAttentionBlock(cfg.d_model, cfg.nhead, cfg.ffn_dim, cfg.dropout)
                for _ in range(cfg.num_cross_attn)
            ]
        )

        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.proj_out = _maybe_proj(cfg.d_model, cfg.embed_dim)

    def forward(
        self,
        video_tokens: torch.Tensor,
        text: torch.Tensor,
        video_pad_mask: torch.Tensor | None = None,
        text_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # video_tokens: [B, N, video_dim]
        # text:         [B, T, text_dim]
        v = self.proj_video(video_tokens)  # [B, N, d_model]
        t = self.proj_text(text)  # [B, T, d_model]

        for block in self.self_attn:
            v = block(v, video_pad_mask)
        for block in self.cross_attn:
            v = block(v, t, text_pad_mask)

        v = _masked_mean(self.norm_out(v), video_pad_mask)  # [B, d_model]
        return F.normalize(self.proj_out(v), dim=-1)  # [B, embed_dim]


class GalleryEncoder(nn.Module):
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.proj_in = _maybe_proj(config.video_dim, config.d_model)
        self.norm_out = nn.LayerNorm(config.d_model)
        self.proj_out = _maybe_proj(config.d_model, config.embed_dim)

    def forward(
        self,
        video_tokens: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # video_tokens: [B, N, video_dim]
        v = self.proj_in(video_tokens)  # [B, N, d_model]
        v = _masked_mean(self.norm_out(v), pad_mask)  # [B, d_model]
        return F.normalize(self.proj_out(v), dim=-1)  # [B, embed_dim]


class InfoNCELoss(nn.Module):
    """Symmetric in-batch InfoNCE.

    L = -log[ exp(sim(q_i, t_i) / τ) / Σ_j exp(sim(q_i, t_j) / τ) ]

    Both embeddings must be L2-normalised (cosine sim = dot product).
    Temperature τ is learnable, initialised to 0.07 (same as CLIP).
    """

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(init_temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=1e-4)

    def forward(
        self, query_emb: torch.Tensor, target_emb: torch.Tensor
    ) -> torch.Tensor:
        sim = query_emb @ target_emb.T / self.temperature  # [B, B]
        labels = torch.arange(len(sim), device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss
