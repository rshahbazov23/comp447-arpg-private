"""
Per-step logging and analysis for the confidence-guided rejection mechanism.

Tracks accept/reject decisions, per-step confidence statistics, wall-clock time,
and per-position deferral counts. Produces a JSON log and a spatial rejection
heatmap on the 16x16 token grid (proposal §4, Analysis).

Supports multi-batch runs:
  - `defer_counts` and `commit_step` accumulate across ALL batches so the heatmap
    reflects aggregate behavior over the full run.
  - `steps` (the per-step detail list) records only the FIRST batch to keep
    JSON sizes manageable on 50k-sample runs (which can involve thousands of
    batches).  First-batch detail is enough for debugging; aggregate analysis
    comes from the cumulative defer_counts / commit_step arrays.
"""

import json
import os
import time

import numpy as np
import torch


class RejectionTracker:
    def __init__(self, num_samples: int, seq_len: int = 256):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.steps = []                                                  # per-step detail, first batch only
        self.defer_counts = np.zeros(seq_len, dtype=np.int64)            # cumulative across batches
        self.commit_step_sum = np.zeros(seq_len, dtype=np.int64)         # sum of commit steps across batches
        self.commit_step_count = np.zeros(seq_len, dtype=np.int64)       # number of batches that committed each position
        self.num_batches_tracked = 0
        self._step_start_time = None

    def step_begin(self):
        self._step_start_time = time.perf_counter()

    def log_step(
        self,
        step: int,
        next_range: torch.Tensor,
        accept_mask_col: torch.Tensor,
        conf_mean: torch.Tensor,
    ):
        """Record one decoding step.

        next_range: (num_pred,) positions attempted this step (1-indexed spatial).
        accept_mask_col: (num_pred,) bool, True = committed, False = deferred.
        conf_mean: (num_pred,) batch-averaged confidence for each position.
        """
        wall_ms = None
        if self._step_start_time is not None:
            wall_ms = (time.perf_counter() - self._step_start_time) * 1000.0
            self._step_start_time = None

        next_range_np = next_range.detach().cpu().numpy()
        accept_np = accept_mask_col.detach().cpu().numpy().astype(bool)
        conf_np = conf_mean.detach().float().cpu().numpy()

        accepted_positions = next_range_np[accept_np].tolist()
        deferred_positions = next_range_np[~accept_np].tolist()

        # `orders` uses 1-indexed positions (1..seq_len); position 0 is the class
        # token. Our arrays are 0-indexed (0..seq_len-1 maps to the image grid).
        for pos in deferred_positions:
            self.defer_counts[pos - 1] += 1
        for pos in accepted_positions:
            # Only record the FIRST step at which this position was committed
            # within the current batch. Tracked per-batch via a sentinel (-1) in
            # the batch-local commit_step view maintained below.
            idx = pos - 1
            if self._batch_commit_step[idx] == -1:
                self._batch_commit_step[idx] = step

        # Per-step detail only for the first batch (keeps JSON small on 50k runs).
        if self.num_batches_tracked == 0:
            accepted_conf = conf_np[accept_np]
            rejected_conf = conf_np[~accept_np]

            def stats(arr):
                if arr.size == 0:
                    return {"min": None, "mean": None, "max": None}
                return {
                    "min": float(arr.min()),
                    "mean": float(arr.mean()),
                    "max": float(arr.max()),
                }

            self.steps.append({
                "step": step,
                "num_pred": int(next_range_np.size),
                "num_accepted": int(accept_np.sum()),
                "num_rejected": int((~accept_np).sum()),
                "accepted_conf": stats(accepted_conf),
                "rejected_conf": stats(rejected_conf),
                "accepted_positions": accepted_positions,
                "deferred_positions": deferred_positions,
                "wall_time_ms": wall_ms,
            })

    def begin_batch(self):
        """Reset per-batch state before running a new batch through the model."""
        self._batch_commit_step = np.full(self.seq_len, -1, dtype=np.int64)

    def end_batch(self):
        """Fold this batch's commit_step into the cumulative aggregate."""
        committed = self._batch_commit_step >= 0
        self.commit_step_sum[committed] += self._batch_commit_step[committed]
        self.commit_step_count[committed] += 1
        self.num_batches_tracked += 1

    def finalize(self):
        """Compute summary statistics after the run completes."""
        total_rejections = int(self.defer_counts.sum())
        total_attempts = sum(s["num_pred"] for s in self.steps) * self.num_batches_tracked
        mean_commit_step = np.where(
            self.commit_step_count > 0,
            self.commit_step_sum / np.maximum(self.commit_step_count, 1),
            -1.0,
        )
        positions_never_committed = int((self.commit_step_count == 0).sum())
        self.summary = {
            "num_samples_per_batch": self.num_samples,
            "num_batches_tracked": self.num_batches_tracked,
            "seq_len": self.seq_len,
            "steps_recorded_first_batch": len(self.steps),
            "first_batch_total_attempts": sum(s["num_pred"] for s in self.steps),
            "aggregate_total_rejections": total_rejections,
            "aggregate_rejection_rate": total_rejections / max(1, total_attempts),
            "first_batch_wall_time_ms": sum(
                s["wall_time_ms"] or 0.0 for s in self.steps
            ),
            "positions_never_committed": positions_never_committed,
            "max_defer_count_per_position": int(self.defer_counts.max()),
            "mean_defer_count_per_position": float(self.defer_counts.mean()),
            "mean_commit_step_per_position": mean_commit_step.tolist(),
        }

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "summary": getattr(self, "summary", {}),
            "steps_first_batch": self.steps,
            "defer_counts_per_position": self.defer_counts.tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def make_heatmap(self, path: str, grid_size: int = 16):
        """Save a spatial heatmap of defer counts on the grid_size x grid_size grid."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        assert grid_size * grid_size == self.seq_len, (
            f"grid_size^2 ({grid_size**2}) must equal seq_len ({self.seq_len})"
        )

        heat = self.defer_counts.reshape(grid_size, grid_size)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(heat, cmap="hot", interpolation="nearest")
        ax.set_title(
            f"Rejection heatmap ({grid_size}x{grid_size})\n"
            f"total rejections={int(self.defer_counts.sum())}, "
            f"batches={self.num_batches_tracked}"
        )
        ax.set_xlabel("x (column)")
        ax.set_ylabel("y (row)")
        fig.colorbar(im, ax=ax, label="times deferred")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
