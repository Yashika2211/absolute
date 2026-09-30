"""Two-tower retrieval model.

User tower: the user's last `max_history` events (item + event type + recency
position) -> mean pool and last-event vector -> MLP -> L2-normalised vector.
Item tower: item embedding (shared with the user tower) + residual MLP -> L2 norm.

Training examples are next-event prediction: for each event, the target is the
item and the input is the user's events *strictly before it* in the fit data,
so training is point-in-time correct by construction. Loss is in-batch sampled
softmax with logQ correction for item popularity and accidental-hit masking.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from streamline.ingest.events import EVENT_TYPES

PAD, OOV = 0, 1
TYPE_INDEX = {t: i + 1 for i, t in enumerate(EVENT_TYPES)}  # 0 is padding


@dataclass
class TwoTowerConfig:
    dim: int = 64
    max_history: int = 50
    min_item_count: int = 2
    batch_size: int = 2048
    lr: float = 2e-3
    weight_decay: float = 0.0
    epochs: int = 10
    patience: int = 2
    temperature: float = 0.05
    seed: int = 42
    device: str = "cpu"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ItemVocab:
    """Raw item id <-> dense index. 0 = padding, 1 = out-of-vocabulary."""

    def __init__(self, item_ids: Sequence[int]) -> None:
        self.ids = np.asarray(item_ids, dtype=np.int64)
        self.index = {int(item): i + 2 for i, item in enumerate(self.ids)}

    @classmethod
    def from_events(cls, events: pl.DataFrame, min_count: int) -> ItemVocab:
        counts = events.group_by("item_id").len().filter(pl.col("len") >= min_count)
        return cls(counts.sort("item_id")["item_id"].to_list())

    def __len__(self) -> int:
        return len(self.ids) + 2

    def encode(self, items: Sequence[int] | np.ndarray) -> np.ndarray:
        return np.fromiter((self.index.get(int(i), OOV) for i in items), np.int64, len(items))

    def decode(self, idx: np.ndarray) -> np.ndarray:
        decoded: np.ndarray = self.ids[idx - 2]
        return decoded


@dataclass
class Sequences:
    """All events of all users, flattened in (user, time) order."""

    items: np.ndarray  # encoded item index per event
    types: np.ndarray  # event type index per event
    user_start: np.ndarray  # index of the user's first event, per event
    user_ids: np.ndarray  # raw user id per event

    @classmethod
    def from_events(cls, events: pl.DataFrame, vocab: ItemVocab) -> Sequences:
        df = events.sort(["user_id", "ts_ms", "item_id"], maintain_order=True)
        user_ids = df["user_id"].to_numpy()
        is_start = np.ones(len(user_ids), dtype=bool)
        is_start[1:] = user_ids[1:] != user_ids[:-1]
        start_idx = np.maximum.accumulate(np.where(is_start, np.arange(len(user_ids)), 0))
        types = df["event"].cast(pl.Utf8).replace_strict(TYPE_INDEX, return_dtype=pl.Int64)
        return cls(
            items=vocab.encode(df["item_id"].to_numpy()),
            types=types.to_numpy(),
            user_start=start_idx,
            user_ids=user_ids,
        )

    def history(self, end: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Events [end - max_len, end) of the same user, right-aligned and zero padded.

        `end` is exclusive, so the event at position `end` itself is never included.
        """
        idx = end[:, None] - max_len + np.arange(max_len)[None, :]
        owner_start = self.user_start[np.maximum(end - 1, 0)]  # user of the last included event
        valid = (idx >= owner_start[:, None]) & (idx >= 0)
        idx = np.clip(idx, 0, len(self.items) - 1)
        return np.where(valid, self.items[idx], PAD), np.where(valid, self.types[idx], PAD)

    def training_positions(self) -> np.ndarray:
        """Events that have at least one earlier event and an in-vocabulary target."""
        pos = np.arange(len(self.items))
        return pos[(pos > self.user_start) & (self.items >= 2)]

    def last_position_per_user(self) -> dict[int, int]:
        """One-past-the-end index of each user's events (for inference)."""
        is_end = np.ones(len(self.user_ids), dtype=bool)
        is_end[:-1] = self.user_ids[:-1] != self.user_ids[1:]
        ends = np.flatnonzero(is_end)
        return {int(u): int(e) + 1 for u, e in zip(self.user_ids[ends], ends, strict=True)}


class TwoTowerNet(nn.Module):
    def __init__(self, n_items: int, dim: int, max_history: int) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(n_items, dim, padding_idx=PAD)
        self.type_emb = nn.Embedding(len(EVENT_TYPES) + 1, dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_history, dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.item_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def user_vectors(self, hist_items: Tensor, hist_types: Tensor) -> Tensor:
        mask = (hist_items != PAD).unsqueeze(-1).float()
        positions = torch.arange(hist_items.shape[1], device=hist_items.device)
        tokens = self.item_emb(hist_items) + self.type_emb(hist_types) + self.pos_emb(positions)
        tokens = tokens * mask
        mean = tokens.sum(1) / mask.sum(1).clamp(min=1.0)
        last = tokens[:, -1]  # right-aligned: most recent event
        return F.normalize(self.user_mlp(torch.cat([mean, last], dim=-1)), dim=-1)

    def item_vectors(self, items: Tensor) -> Tensor:
        emb = self.item_emb(items)
        return F.normalize(emb + self.item_mlp(emb), dim=-1)


def in_batch_loss(
    user: Tensor, item: Tensor, item_idx: Tensor, log_q: Tensor, temperature: float
) -> Tensor:
    """Sampled softmax over in-batch items with logQ correction and duplicate masking."""
    logits = user @ item.T / temperature - log_q[item_idx][None, :]
    duplicate = item_idx[:, None] == item_idx[None, :]
    duplicate.fill_diagonal_(False)
    logits = logits.masked_fill(duplicate, float("-inf"))
    labels = torch.arange(len(user), device=user.device)
    return F.cross_entropy(logits, labels)


EpochCallback = Callable[[int, float, "TwoTowerRecommender"], float | None]


class TwoTowerRecommender:
    name = "two_tower"

    def __init__(self, config: TwoTowerConfig | None = None) -> None:
        self.config = config or TwoTowerConfig()
        self.vocab: ItemVocab | None = None
        self.net: TwoTowerNet | None = None
        self.seqs: Sequences | None = None
        self.user_end: dict[int, int] = {}
        self.history_log: list[dict[str, float]] = []
        self.best_epoch = 0

    def fit(self, history: pl.DataFrame, on_epoch: EpochCallback | None = None) -> None:
        """Train on `history`. If `on_epoch` returns a score, keep the best epoch
        and stop after `patience` epochs without improvement."""
        cfg = self.config
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        device = torch.device(cfg.device)

        self.vocab = ItemVocab.from_events(history, cfg.min_item_count)
        self.seqs = Sequences.from_events(history, self.vocab)
        self.user_end = self.seqs.last_position_per_user()
        self.net = TwoTowerNet(len(self.vocab), cfg.dim, cfg.max_history).to(device)

        counts = np.bincount(self.seqs.items, minlength=len(self.vocab)).astype(np.float64)
        log_q = torch.tensor(np.log((counts + 1) / (counts + 1).sum()), dtype=torch.float32)
        log_q = log_q.to(device)

        positions = self.seqs.training_positions()
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best_score, best_state, bad_epochs = -np.inf, None, 0

        for epoch in range(1, cfg.epochs + 1):
            self.net.train()
            start, total = time.perf_counter(), 0.0
            order = rng.permutation(positions)
            n_batches = len(order) // cfg.batch_size  # drop last partial batch
            for b in range(n_batches):
                batch = order[b * cfg.batch_size : (b + 1) * cfg.batch_size]
                h_items, h_types = self.seqs.history(batch, cfg.max_history)
                target = torch.from_numpy(self.seqs.items[batch]).to(device)
                user = self.net.user_vectors(
                    torch.from_numpy(h_items).to(device), torch.from_numpy(h_types).to(device)
                )
                loss = in_batch_loss(
                    user, self.net.item_vectors(target), target, log_q, cfg.temperature
                )
                opt.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                opt.step()
                total += loss.item()
            record = {
                "epoch": epoch,
                "train_loss": total / max(n_batches, 1),
                "epoch_seconds": time.perf_counter() - start,
            }
            score = on_epoch(epoch, record["train_loss"], self) if on_epoch else None
            if score is not None:
                record["val_score"] = score
            self.history_log.append(record)
            if score is None:
                self.best_epoch = epoch
                continue
            if score > best_score:
                best_score, bad_epochs, self.best_epoch = score, 0, epoch
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    @torch.no_grad()
    def recommend(self, user_ids: Sequence[int], k: int, chunk: int = 4096) -> list[list[int]]:
        assert self.net is not None and self.seqs is not None and self.vocab is not None
        self.net.eval()
        device = torch.device(self.config.device)
        n = len(self.vocab)
        item_vecs = self.net.item_vectors(torch.arange(2, n, device=device))
        k_eff = min(k, n - 2)

        out: list[list[int]] = []
        for s in range(0, len(user_ids), chunk):
            users = user_ids[s : s + chunk]
            # users unseen in fit data get an empty history (end=0 with no valid events)
            ends = np.array([self.user_end.get(int(u), 0) for u in users], dtype=np.int64)
            h_items, h_types = self.seqs.history(ends, self.config.max_history)
            h_items[ends == 0] = PAD
            h_types[ends == 0] = PAD
            vecs = self.net.user_vectors(
                torch.from_numpy(h_items).to(device), torch.from_numpy(h_types).to(device)
            )
            top = torch.topk(vecs @ item_vecs.T, k_eff, dim=1).indices.cpu().numpy() + 2
            out.extend(self.vocab.decode(row).tolist() for row in top)
        return out
