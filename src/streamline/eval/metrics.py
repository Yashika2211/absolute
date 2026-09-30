"""Binary-relevance ranking metrics for a single user.

`recommended` is a ranked list (best first); `relevant` is the set of items the
user actually interacted with in the target window.
"""

from __future__ import annotations

import math
from collections.abc import Sequence, Set


def _check(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def recall_at_k(recommended: Sequence[int], relevant: Set[int], k: int) -> float:
    """Fraction of relevant items that appear in the top-k."""
    _check(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: Set[int], k: int) -> float:
    """DCG of the top-k divided by the best achievable DCG for this user."""
    _check(k)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal


def mrr_at_k(recommended: Sequence[int], relevant: Set[int], k: int) -> float:
    """Reciprocal rank of the first relevant item in the top-k (0 if none)."""
    _check(k)
    for rank, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1.0 / (rank + 1)
    return 0.0
