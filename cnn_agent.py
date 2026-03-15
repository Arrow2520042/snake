"""Dueling CNN DQN agent with PER, n-step returns, and CUDA support.

Uses a grid observation (body_age, head, food, walls) + auxiliary vector
(direction one-hot, normalized length) instead of hand-crafted features.
Shares PER and NStepBuffer infrastructure with dqn_agent.py.

Training stack highlights:
- Adam optimizer for adaptive gradient updates.
- ReduceLROnPlateau scheduler for score-based LR decay.
- Soft target network updates (Polyak averaging).
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_agent import PrioritizedReplayBuffer, NStepBuffer


class DuelingCNN(nn.Module):
    """Dueling DQN with convolutional backbone for grid input.

    Input tensor layout (flat): [grid_channels * board^2 | aux_features]
    The forward pass reshapes the grid portion into (C, H, W) for conv layers,
    then concatenates the flattened conv output with auxiliary features before
    the dueling value/advantage heads.
    """

    def __init__(self, board_size, n_channels=4, n_aux=5, n_actions=3):
        super().__init__()
        self.board_size = board_size
        self.n_channels = n_channels
        self.n_aux = n_aux
        self.grid_size = n_channels * board_size * board_size

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        dummy = torch.zeros(1, n_channels, board_size, board_size)
        conv_flat = self.conv(dummy).view(1, -1).size(1)
        combined_dim = conv_flat + n_aux

        self.value_stream = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        grid = x[:, :self.grid_size].view(-1, self.n_channels, self.board_size, self.board_size)
        aux = x[:, self.grid_size:]

        conv_out = self.conv(grid).view(grid.size(0), -1)
        combined = torch.cat([conv_out, aux], dim=1)

        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class CNNAgent:
    """Dueling Double DQN with CNN backbone, n-step returns, PER, and CUDA.

    Grid state: 4 channels (body_age, head, food, walls) + 5 aux features
    (direction one-hot + normalized length) = 4*board^2 + 5 flat floats.
    """

    def __init__(self, board_size=10, n_channels=4, n_aux=5, n_actions=3,
                 lr=5e-4, gamma=0.99, batch_size=256, capacity=200_000,
                 tau=0.005, max_grad_norm=1.0, n_step=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.n_step = n_step
        self.board_size = board_size
        self.n_channels = n_channels
        self.n_aux = n_aux

        self.policy_net = DuelingCNN(board_size, n_channels, n_aux, n_actions).to(self.device)
        self.target_net = DuelingCNN(board_size, n_channels, n_aux, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Adam optimizer drives gradient-based learning of the policy network.
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # Scheduler decays learning rate when validation metric (avg200) stalls.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)
        self.replay = PrioritizedReplayBuffer(capacity)

        self.steps = 0
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.9997
        self._nstep_buffers = {}

    # -- action selection ------------------------------------------------
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
        # Decide which envs explore vs exploit
        rands = np.random.random(n)
        explore_mask = rands < self.eps
        if masks is None:
            random_actions = np.random.randint(0, self.n_actions, size=n)
        else:
            random_actions = np.empty(n, dtype=np.int64)
            for i in range(n):
                valid = np.flatnonzero(masks[i])
                random_actions[i] = int(np.random.choice(valid))

        # If all exploring, skip the forward pass entirely
        if explore_mask.all():
            return random_actions

        with torch.no_grad():
            batch = torch.as_tensor(np.array(states), dtype=torch.float32).to(self.device)
            q_values = self.policy_net(batch)
            if masks is None:
                greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            else:
                q_np = q_values.detach().cpu().numpy()
                masked_q = np.where(masks, q_np, -1e9)
                greedy_actions = masked_q.argmax(axis=1)

        actions = np.where(explore_mask, random_actions, greedy_actions)
        return actions

    # -- experience storage ----------------------------------------------
    def push(self, env_id, state, action, reward, next_state, done, snake_length=None):
        if env_id not in self._nstep_buffers:
            self._nstep_buffers[env_id] = NStepBuffer(self.n_step, self.gamma)
        if snake_length is not None:
            self._nstep_buffers[env_id].n = max(3, min(self.n_step, snake_length))
        transitions = self._nstep_buffers[env_id].push(state, action, reward, next_state, done)
        for s0, a0, R, s_n, d_n, steps_used in transitions:
            self.replay.push(s0, a0, R, s_n, d_n, steps_used)


    # -- gradient update -------------------------------------------------
    def update(self):
        """Run one CNN-DQN optimization step with PER weighting and n-step targets."""
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

        with torch.no_grad():
            best_actions = self.policy_net(ns_t).argmax(1, keepdim=True)
            next_q = self.target_net(ns_t).gather(1, best_actions)
            gamma_n = self.gamma ** steps_t
            target = r_t + gamma_n * next_q * (1.0 - d_t)

        td_errors = (q_values - target).detach().cpu().squeeze(1).numpy()
        loss = (w_t * nn.functional.mse_loss(q_values, target, reduction='none')).mean()

        # Adam update step with optional gradient clipping for stability.
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.replay.update_priorities(indices, td_errors)
        self._soft_update()

        return loss.item(), q_values.detach().mean().item()

    # -- epsilon / scheduler ---------------------------------------------
    def decay_epsilon(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
            if self.eps < self.eps_min:
                self.eps = self.eps_min

    def step_scheduler(self, metric):
        self.scheduler.step(metric)

    # -- soft target update ----------------------------------------------
    def _soft_update(self):
        """Polyak averaging: target <- tau * policy + (1 - tau) * target."""
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    # -- persistence -----------------------------------------------------
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'eps': self.eps,
            'steps': self.steps,
            'n_step': self.n_step,
            'board_size': self.board_size,
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
            self.policy_net.load_state_dict(data)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
