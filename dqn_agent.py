"""Core DQN agent components for Snake training.

This module contains:
- Prioritized Experience Replay (PER) with SumTree.
- N-step return buffering.
- Dueling Double DQN policy/target networks.
- Adam optimization and ReduceLROnPlateau scheduling.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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

    def _get_priority(self, error):
        return (np.abs(error) + self.PER_E) ** self.PER_A

    def push(self, state, action, reward, next_state, done, steps=1):
        s = np.asarray(state, dtype=np.float32).copy()
        ns = np.asarray(next_state, dtype=np.float32).copy()
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, (s, int(action), float(reward), ns, bool(done), int(steps)))

    def sample(self, batch_size):
        batch, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size
        beta = min(
            self.PER_B_END,
            self.PER_B_START + (self.PER_B_END - self.PER_B_START) * self._beta_step / self.PER_B_FRAMES,
        )
        self._beta_step += 1

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
        for idx, error in zip(indices, errors):
            self.tree.update(idx, self._get_priority(error))

    def __len__(self):
        return self.tree.size


class DuelingNet(nn.Module):
    """Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A)."""

    def __init__(self, input_dim, output_dim, hidden=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class NStepBuffer:
    """Per-environment n-step return accumulator."""

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if done:
            # Flush all remaining transitions on episode end
            return self._flush_all()
        if len(self.buffer) >= self.n:
            return [self._make_nstep()]
        return []

    def _make_nstep(self):
        """Build one n-step transition from the front of the buffer."""
        R = 0.0
        for i in range(len(self.buffer)):
            R += (self.gamma ** i) * self.buffer[i][2]
        s0 = self.buffer[0][0]
        a0 = self.buffer[0][1]
        last = self.buffer[-1]
        self.buffer.pop(0)
        return (s0, a0, R, last[3], last[4], len(self.buffer) + 1)

    def _flush_all(self):
        """Flush remaining transitions at episode end."""
        transitions = []
        while self.buffer:
            transitions.append(self._make_nstep())
        return transitions

    def reset(self):
        self.buffer.clear()


class DQNAgent:
    """Dueling Double DQN with n-step returns, soft target update, PER, and CUDA."""

    STATE_DIM = 26

    def __init__(self, state_dim=26, n_actions=3, lr=1e-3, gamma=0.99,
                 batch_size=256, capacity=500_000, tau=0.005, max_grad_norm=1.0,
                 n_step=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.n_step = n_step

        self.policy_net = DuelingNet(state_dim, n_actions).to(self.device)
        self.target_net = DuelingNet(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Adam performs adaptive first/second-moment gradient scaling.
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # Scheduler lowers LR when score metric plateaus for many evaluations.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-5)
        self.replay = PrioritizedReplayBuffer(capacity)
        self.steps = 0
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.9997
        self._nstep_buffers = {}  # env_id -> NStepBuffer

    def _normalize_action_mask(self, action_mask):
        if action_mask is None:
            return None
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.n_actions:
            return None
        if not mask.any():
            return np.ones(self.n_actions, dtype=bool)
        return mask

    def _normalize_action_masks(self, action_masks, batch_size):
        if action_masks is None:
            return None
        masks = np.asarray(action_masks, dtype=bool)
        if masks.shape != (batch_size, self.n_actions):
            return None
        dead_rows = ~masks.any(axis=1)
        if dead_rows.any():
            masks = masks.copy()
            masks[dead_rows] = True
        return masks

    def act(self, state, action_mask=None):
        self.steps += 1
        mask = self._normalize_action_mask(action_mask)
        if random.random() < self.eps:
            if mask is not None:
                valid = np.flatnonzero(mask)
                return int(np.random.choice(valid))
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(s).squeeze(0).detach().cpu().numpy()
            if mask is not None:
                q_vals = np.where(mask, q_vals, -1e9)
            return int(np.argmax(q_vals))

    def act_batch(self, states, action_masks=None):
        """Select actions for a batch of states in a single forward pass."""
        n = len(states)
        self.steps += n
        masks = self._normalize_action_masks(action_masks, n)
        explore_mask = np.random.random(n) < self.eps
        actions = np.empty(n, dtype=np.int64)

        # Sample random actions only for exploring environments.
        if explore_mask.any():
            explore_indices = np.flatnonzero(explore_mask)
            if masks is None:
                actions[explore_indices] = np.random.randint(0, self.n_actions, size=explore_indices.size)
            else:
                for i in explore_indices:
                    valid = np.flatnonzero(masks[i])
                    actions[i] = int(np.random.choice(valid))

        if explore_mask.all():
            return actions

        with torch.no_grad():
            batch = torch.as_tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32, device=self.device)
            q_values = self.policy_net(batch)
            if masks is not None:
                mask_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
                q_values = q_values.masked_fill(~mask_t, torch.finfo(q_values.dtype).min)
            greedy_actions = q_values.argmax(dim=1).detach().cpu().numpy()

        actions[~explore_mask] = greedy_actions[~explore_mask]
        return actions

    def push(self, env_id, state, action, reward, next_state, done, snake_length=None):
        """Push a transition through the n-step buffer for env_id."""
        if env_id not in self._nstep_buffers:
            self._nstep_buffers[env_id] = NStepBuffer(self.n_step, self.gamma)
        # Dynamic n-step: short snake → small n (fast learning), long snake → large n (better planning)
        if snake_length is not None:
            self._nstep_buffers[env_id].n = max(3, min(self.n_step, snake_length))
        transitions = self._nstep_buffers[env_id].push(state, action, reward, next_state, done)
        for s0, a0, R, s_n, d_n, steps_used in transitions:
            self.replay.push(s0, a0, R, s_n, d_n, steps_used)


    def update(self):
        """Run one DQN optimization step using PER-weighted Double DQN targets."""
        if len(self.replay) < self.batch_size:
            return None, None
        sample = self.replay.sample(self.batch_size)
        if sample is None:
            return None, None

        s, a, r, ns, d, steps, indices, weights = sample
        dev = self.device
        s_t = torch.as_tensor(s, dtype=torch.float32).to(dev)
        a_t = torch.as_tensor(a, dtype=torch.int64).unsqueeze(1).to(dev)
        r_t = torch.as_tensor(r, dtype=torch.float32).unsqueeze(1).to(dev)
        ns_t = torch.as_tensor(ns, dtype=torch.float32).to(dev)
        d_t = torch.as_tensor(d, dtype=torch.float32).unsqueeze(1).to(dev)
        w_t = torch.as_tensor(weights, dtype=torch.float32).unsqueeze(1).to(dev)
        steps_t = torch.as_tensor(steps, dtype=torch.float32).unsqueeze(1).to(dev)

        q_values = self.policy_net(s_t).gather(1, a_t)

        # Double DQN with per-transition n-step discount
        with torch.no_grad():
            best_actions = self.policy_net(ns_t).argmax(1, keepdim=True)
            next_q = self.target_net(ns_t).gather(1, best_actions)
            gamma_n = self.gamma ** steps_t
            target = r_t + gamma_n * next_q * (1.0 - d_t)

        td_errors = (q_values - target).detach().cpu().squeeze(1).numpy()
        loss = (w_t * nn.functional.mse_loss(q_values, target, reduction='none')).mean()

        # Standard Adam update: zero grad -> backprop -> gradient clipping -> optimizer step.
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.replay.update_priorities(indices, td_errors)
        self._soft_update()

        return loss.item(), q_values.detach().mean().item()

    def decay_epsilon(self):
        """Call once per round from training loop (not per gradient update)."""
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
            if self.eps < self.eps_min:
                self.eps = self.eps_min

    def _soft_update(self):
        """Polyak averaging: target <- tau * policy + (1 - tau) * target."""
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    def step_scheduler(self, metric):
        """Step ReduceLROnPlateau with a score metric (for example avg200)."""
        self.scheduler.step(metric)

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'eps': self.eps,
            'steps': self.steps,
            'n_step': self.n_step,
        }, path)

    def load(self, path, weights_only=False):
        data = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(data, dict) and 'policy_net' in data:
            self.policy_net.load_state_dict(data['policy_net'])
            self.target_net.load_state_dict(data['target_net'])
            if not weights_only:
                if 'optimizer' in data:
                    self.optimizer.load_state_dict(data['optimizer'])
                if 'scheduler' in data:
                    self.scheduler.load_state_dict(data['scheduler'])
                if 'eps' in data:
                    self.eps = data['eps']
                if 'steps' in data:
                    self.steps = data['steps']
        else:
            # Legacy: plain state_dict
            self.policy_net.load_state_dict(data)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
