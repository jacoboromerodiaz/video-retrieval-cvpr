"""Collation utilities for variable-length patch embeddings."""

import torch


def _pad_and_mask(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(tensors)
    N_max = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[1]
    padded = tensors[0].new_zeros(B, N_max, D)
    key_padding_mask = torch.ones(B, N_max, dtype=torch.bool)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        padded[i, :n] = t
        key_padding_mask[i, :n] = False
    return padded, key_padding_mask


def collate_fn(samples: list[dict]) -> dict:
    """Pad variable-length patch tensors and produce key-padding masks."""
    src_padded, src_mask = _pad_and_mask([s["source_patches"] for s in samples])
    tgt_padded, tgt_mask = _pad_and_mask([s["target_patches"] for s in samples])
    query_padded, query_mask = _pad_and_mask([s["query_emb"] for s in samples])
    return {
        "source_patches": src_padded,  # [B, N_max, vid_dim]
        "source_mask": src_mask,  # [B, N_max]
        "target_patches": tgt_padded,  # [B, N_max', vid_dim]
        "target_mask": tgt_mask,  # [B, N_max']
        "query_emb": query_padded,  # [B, T_max, text_dim]
        "query_mask": query_mask,  # [B, T_max]
    }
