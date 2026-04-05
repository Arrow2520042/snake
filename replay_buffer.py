"""Replay and n-step utilities used by CNN training.

This module is intentionally model-agnostic and contains only:
- SumTree-based prioritized replay buffer (PER)
- N-step transition accumulator
- Optional Cython backend loading helpers
"""

import glob
import importlib
import os
import random
import sys
from typing import Any

import numpy as np


def _load_per_cython_backend():
    """Load optional Cython PER backend from in-place or build output paths."""
    try:
        return importlib.import_module('per_cython_backend'), True
    except Exception:
        pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    build_glob = os.path.join(base_dir, 'build', 'lib.*')
    for build_dir in sorted(glob.glob(build_glob)):
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)
        try:
            return importlib.import_module('per_cython_backend'), True
        except Exception:
            continue

    return None, False


_tmp_per_cy, _PER_CYTHON_AVAILABLE = _load_per_cython_backend()
_per_cy: Any = _tmp_per_cy


class SumTree:
    """Binary sum tree for proportional prioritized replay."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx, change):
        idx = (idx - 1) // 2
        while idx >= 0:
            self.tree[idx] += change
            if idx == 0:
                break
            idx = (idx - 1) // 2

    def _retrieve(self, idx, value):
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value):
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Proportional PER with importance sampling."""

    PER_E = 0.01
    PER_A = 0.6
    PER_B_START = 0.4
    PER_B_END = 1.0
    PER_B_FRAMES = 100_000

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self._beta_step = 0
        self._cython_enabled = bool(_PER_CYTHON_AVAILABLE)
        self._max_priority = 1.0

    @property
    def cython_enabled(self):
        return self._cython_enabled

    def _get_priority(self, error):
        return (np.abs(error) + self.PER_E) ** self.PER_A

    def push(self, state, action, reward, next_state, done, steps=1):
        s = np.asarray(state, dtype=np.float32).copy()
        ns = np.asarray(next_state, dtype=np.float32).copy()
        self.tree.add(self._max_priority, (s, int(action), float(reward), ns, bool(done), int(steps)))

    def sample(self, batch_size):
        batch, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size
        beta = min(
            self.PER_B_END,
            self.PER_B_START + (self.PER_B_END - self.PER_B_START) * self._beta_step / self.PER_B_FRAMES,
        )
        self._beta_step += 1

        if self._cython_enabled and segment > 0.0:
            offsets = np.random.uniform(0.0, segment, size=batch_size).astype(np.float64)
            values = offsets + segment * np.arange(batch_size, dtype=np.float64)
            idx_arr = _per_cy.sample_indices(self.tree.tree, self.tree.capacity, values)
            for idx in idx_arr:
                idx_i = int(idx)
                data_idx = idx_i - self.tree.capacity + 1
                data = self.tree.data[data_idx] if 0 <= data_idx < self.tree.capacity else None
                if data is None:
                    value = random.uniform(0, self.tree.total())
                    idx_i, priority, data = self.tree.get(value)
                    if data is None:
                        continue
                else:
                    priority = self.tree.tree[idx_i]
                batch.append(data)
                indices.append(idx_i)
                priorities.append(priority)
        else:
            for i in range(batch_size):
                lo = segment * i
                hi = segment * (i + 1)
                value = random.uniform(lo, hi)
                idx, priority, data = self.tree.get(value)
                if data is None:
                    value = random.uniform(0, self.tree.total())
                    idx, priority, data = self.tree.get(value)
                if data is None:
                    continue
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)

        if not batch:
            return None

        s, a, r, ns, d, steps = zip(*batch)
        probs = np.array(priorities, dtype=np.float64) / (self.tree.total() + 1e-8)
        weights = (self.tree.size * probs + 1e-8) ** (-beta)
        weights /= weights.max()

        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.uint8),
            np.array(steps, dtype=np.int32),
            np.array(indices, dtype=np.int64),
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, errors):
        idx_np = np.asarray(indices, dtype=np.int64)
        if idx_np.size == 0:
            return
        pri_np = np.asarray(self._get_priority(np.asarray(errors, dtype=np.float64)), dtype=np.float64)
        if pri_np.size:
            self._max_priority = max(self._max_priority, float(np.max(pri_np)))
        if self._cython_enabled:
            _per_cy.batch_update(self.tree.tree, idx_np, pri_np)
        else:
            for idx, priority in zip(idx_np, pri_np):
                self.tree.update(int(idx), float(priority))

    def clear(self):
        self.tree = SumTree(self.capacity)
        self._beta_step = 0
        self._max_priority = 1.0

    def __len__(self):
        return self.tree.size


class NStepBuffer:
    """Per-environment n-step return accumulator."""

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if done:
            return self._flush_all()
        if len(self.buffer) >= self.n:
            return [self._make_nstep()]
        return []

    def _make_nstep(self):
        """Build one n-step transition from the front of the buffer."""
        reward_sum = 0.0
        for i in range(len(self.buffer)):
            reward_sum += (self.gamma ** i) * self.buffer[i][2]
        s0 = self.buffer[0][0]
        a0 = self.buffer[0][1]
        last = self.buffer[-1]
        self.buffer.pop(0)
        return (s0, a0, reward_sum, last[3], last[4], len(self.buffer) + 1)

    def _flush_all(self):
        transitions = []
        while self.buffer:
            transitions.append(self._make_nstep())
        return transitions

    def reset(self):
        self.buffer.clear()


__all__ = [
    'SumTree',
    'PrioritizedReplayBuffer',
    'NStepBuffer',
]
