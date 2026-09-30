import numpy as np
import pytest
import torch

from conftest import make_events
from streamline.eval.harness import evaluate
from streamline.training.two_tower import (
    PAD,
    ItemVocab,
    Sequences,
    TwoTowerConfig,
    TwoTowerRecommender,
    in_batch_loss,
)

EVENTS = make_events(
    [
        (1, 1, 10, "view"),
        (2, 1, 11, "view"),
        (3, 1, 12, "addtocart"),
        (4, 2, 10, "view"),
        (5, 2, 13, "view"),
        (6, 3, 99, "view"),  # item 99 appears once -> OOV with min_count=2
        (7, 3, 11, "view"),
        (8, 3, 12, "view"),
        (9, 2, 12, "view"),
        (10, 1, 13, "view"),
    ]
)


def test_vocab_roundtrip() -> None:
    vocab = ItemVocab.from_events(EVENTS, min_count=2)
    assert len(vocab) == 4 + 2
    encoded = vocab.encode([10, 99, 13])
    assert encoded[1] == 1  # OOV
    assert vocab.decode(encoded[[0, 2]]).tolist() == [10, 13]


def test_history_has_no_leakage() -> None:
    """History for position p holds only the same user's events strictly before p."""
    vocab = ItemVocab.from_events(EVENTS, min_count=1)
    seqs = Sequences.from_events(EVENTS, vocab)
    ordered = EVENTS.sort(["user_id", "ts_ms"])
    ts, users = ordered["ts_ms"].to_numpy(), ordered["user_id"].to_numpy()

    positions = np.arange(len(ts) + 1)
    items, _ = seqs.history(positions, max_len=3)
    for p, row in zip(positions, items, strict=True):
        hist = [i for i in row if i != PAD]
        lo = p - len(hist)
        assert lo >= 0
        # the true previous events, same user, strictly earlier in time
        assert hist == seqs.items[lo:p].tolist()
        if hist:
            assert (users[lo:p] == users[p - 1]).all()
            if p < len(ts) and users[p] == users[p - 1]:
                assert (ts[lo:p] < ts[p]).all()


def test_inference_history_does_not_bleed_into_next_user() -> None:
    vocab = ItemVocab.from_events(EVENTS, min_count=1)
    seqs = Sequences.from_events(EVENTS, vocab)
    ends = seqs.last_position_per_user()
    items, _ = seqs.history(np.array([ends[1]]), max_len=10)
    assert vocab.decode(items[items != PAD]).tolist() == [10, 11, 12, 13]


def test_training_positions_skip_first_event_and_oov() -> None:
    vocab = ItemVocab.from_events(EVENTS, min_count=2)
    seqs = Sequences.from_events(EVENTS, vocab)
    pos = seqs.training_positions()
    assert (pos > seqs.user_start[pos]).all()
    assert (seqs.items[pos] >= 2).all()


def test_in_batch_loss_masks_duplicates() -> None:
    user = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    item = user.clone()
    idx = torch.tensor([2, 3, 2, 4])
    log_q = torch.zeros(5)
    loss = in_batch_loss(user, item, idx, log_q, temperature=0.1)
    assert torch.isfinite(loss)


def test_fit_and_recommend_learns_cooccurrence() -> None:
    # training users view A then B; new users who have only viewed A should get B first
    rows = []
    for u in range(200):
        a, b = (1, 2) if u % 2 == 0 else (3, 4)
        rows += [(u * 10, u, a, "view"), (u * 10 + 1, u, b, "view")]
    eval_users = range(1000, 1020)
    rows += [(5000 + u, u, 1 if u % 2 == 0 else 3, "view") for u in eval_users]
    history = make_events(rows)
    target = make_events([(10_000, u, 2 if u % 2 == 0 else 4, "view") for u in eval_users])
    cfg = TwoTowerConfig(dim=16, batch_size=32, epochs=30, lr=1e-2, min_item_count=1, patience=100)
    metrics = evaluate(TwoTowerRecommender(cfg), history, target, ks=[1])
    assert metrics["recall@1"] == pytest.approx(1.0)


def test_unknown_user_gets_recommendations() -> None:
    model = TwoTowerRecommender(TwoTowerConfig(dim=8, batch_size=2, epochs=1, min_item_count=1))
    model.fit(EVENTS)
    recs = model.recommend([12345], k=3)
    assert len(recs[0]) == 3
    assert set(recs[0]) <= set(EVENTS["item_id"].to_list())


def test_recommend_before_fit_fails() -> None:
    with pytest.raises(AssertionError):
        TwoTowerRecommender().recommend([1], k=1)
