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
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, value):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])

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

    def push(self, state, action, reward, next_state, done):
        s = np.asarray(state, dtype=np.float32).copy()
        ns = np.asarray(next_state, dtype=np.float32).copy()
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, (s, int(action), float(reward), ns, bool(done)))

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

        s, a, r, ns, d = zip(*batch)
        probs = np.array(priorities, dtype=np.float64) / (self.tree.total() + 1e-8)
        weights = (self.tree.size * probs + 1e-8) ** (-beta)
        weights /= weights.max()

        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.uint8),
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
    """Dueling Double DQN with n-step returns, soft target update, and PER.
    CPU-only.  State vector has 23 features by default.
    """

    STATE_DIM = 26

    def __init__(self, state_dim=26, n_actions=3, lr=1e-3, gamma=0.99,
                 batch_size=256, capacity=500_000, tau=0.005, max_grad_norm=1.0,
                 n_step=5):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.n_step = n_step

        self.policy_net = DuelingNet(state_dim, n_actions)
        self.target_net = DuelingNet(state_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-5)
        self.replay = PrioritizedReplayBuffer(capacity)
        self.steps = 0
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.9999
        self._nstep_buffers = {}  # env_id -> NStepBuffer

    def act(self, state):
        self.steps += 1
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(torch.argmax(self.policy_net(s)).item())

    def push(self, env_id, state, action, reward, next_state, done):
        """Push a transition through the n-step buffer for env_id."""
        if env_id not in self._nstep_buffers:
            self._nstep_buffers[env_id] = NStepBuffer(self.n_step, self.gamma)
        transitions = self._nstep_buffers[env_id].push(state, action, reward, next_state, done)
        for s0, a0, R, s_n, d_n, steps_used in transitions:
            self.replay.push(s0, a0, R, s_n, d_n)

    def reset_nstep(self, env_id):
        """Reset n-step buffer for an environment (on episode boundary)."""
        if env_id in self._nstep_buffers:
            self._nstep_buffers[env_id].reset()

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        sample = self.replay.sample(self.batch_size)
        if sample is None:
            return

        s, a, r, ns, d, indices, weights = sample
        s_t = torch.as_tensor(s, dtype=torch.float32)
        a_t = torch.as_tensor(a, dtype=torch.int64).unsqueeze(1)
        r_t = torch.as_tensor(r, dtype=torch.float32).unsqueeze(1)
        ns_t = torch.as_tensor(ns, dtype=torch.float32)
        d_t = torch.as_tensor(d, dtype=torch.float32).unsqueeze(1)
        w_t = torch.as_tensor(weights, dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_net(s_t).gather(1, a_t)

        # Double DQN with n-step discount
        with torch.no_grad():
            best_actions = self.policy_net(ns_t).argmax(1, keepdim=True)
            next_q = self.target_net(ns_t).gather(1, best_actions)
            gamma_n = self.gamma ** self.n_step
            target = r_t + gamma_n * next_q * (1.0 - d_t)

        td_errors = (q_values - target).detach().squeeze(1).numpy()
        loss = (w_t * nn.functional.mse_loss(q_values, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.replay.update_priorities(indices, td_errors)
        self._soft_update()

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
            if self.eps < self.eps_min:
                self.eps = self.eps_min

    def _soft_update(self):
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    def step_scheduler(self, metric):
        """Step the LR scheduler with a score metric (e.g. avg50 score)."""
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
        data = torch.load(path, map_location='cpu')
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
