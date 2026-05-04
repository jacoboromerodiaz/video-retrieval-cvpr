import torch


def _ranks(scores: torch.Tensor, gt_indices: torch.Tensor) -> torch.Tensor:
    """0-indexed rank of the ground-truth gallery item for each query."""
    ranked = scores.argsort(dim=1, descending=True)  # [Q, G]
    return (ranked == gt_indices.unsqueeze(1)).int().argmax(dim=1)  # [Q]


def recall_at_k(scores: torch.Tensor, gt_indices: torch.Tensor, k: int) -> float:
    """Fraction of queries where the GT item appears in the top-K results."""
    return (_ranks(scores, gt_indices) < k).float().mean().item()


def median_rank(scores: torch.Tensor, gt_indices: torch.Tensor) -> float:
    """Median 1-indexed rank of the GT item across all queries."""
    return _ranks(scores, gt_indices).float().median().item() + 1.0


def mean_rank(scores: torch.Tensor, gt_indices: torch.Tensor) -> float:
    """Mean 1-indexed rank of the GT item across all queries."""
    return _ranks(scores, gt_indices).float().mean().item() + 1.0
