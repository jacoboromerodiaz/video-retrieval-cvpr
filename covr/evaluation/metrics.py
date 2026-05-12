import torch
from typing import Union, Sequence


def _ranks(scores: torch.Tensor, gt_indices: torch.Tensor) -> torch.Tensor:
    """0-indexed rank of the ground-truth gallery item for each query."""
    ranked = scores.argsort(dim=1, descending=True)  # [Q, G]
    return (ranked == gt_indices.unsqueeze(1)).int().argmax(dim=1)  # [Q]


def recall_at_k(
    scores: torch.Tensor,
    gt_indices: torch.Tensor,
    k: Union[int, Sequence[int]],
) -> Union[float, list[float]]:
    """Fraction of queries where the GT item appears in the top-K results.
    If k is an int, returns a float.
    If k is a list/sequence, returns a list of floats.
    """
    ranks = _ranks(scores, gt_indices)
    if isinstance(k, int):
        return (ranks < k).float().mean().item()
    return [(ranks < k_value).float().mean().item() for k_value in k]


def median_rank(scores: torch.Tensor, gt_indices: torch.Tensor) -> float:
    """Median 1-indexed rank of the GT item across all queries."""
    return _ranks(scores, gt_indices).float().median().item() + 1.0


def mean_rank(scores: torch.Tensor, gt_indices: torch.Tensor) -> float:
    """Mean 1-indexed rank of the GT item across all queries."""
    return _ranks(scores, gt_indices).float().mean().item() + 1.0
