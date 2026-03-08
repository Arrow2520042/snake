import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_agent import PrioritizedReplayBuffer


class CNNNet(nn.Module):
    def __init__(self, board_size, n_channels=4, n_actions=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        dummy = torch.zeros(1, n_channels, board_size, board_size)
        flat_size = self.conv(dummy).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNAgent:
    """CNN-based Double DQN agent using 4-channel grid observation.
    Channels: [head, body, food, walls].  CPU-only.
    """

    def __init__(self, board_size=20, n_actions=3, lr=5e-4, gamma=0.99,
                 batch_size=64, capacity=50_000, tau=0.005, max_grad_norm=1.0):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.board_size = board_size

        self.policy_net = CNNNet(board_size, n_channels=4, n_actions=n_actions)
        self.target_net = CNNNet(board_size, n_channels=4, n_actions=n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = PrioritizedReplayBuffer(capacity)
        self.steps = 0
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.9995

    def act(self, state):
        """state: np.ndarray of shape (4, board_size, board_size)."""
        self.steps += 1
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(torch.argmax(self.policy_net(s)).item())

    def push(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        sample = self.replay.sample(self.batch_size)
        if sample is None:
            return

        s, a, r, ns, d, indices, weights = sample
        bs = self.board_size
        s_t = torch.as_tensor(s, dtype=torch.float32).view(-1, 4, bs, bs)
        ns_t = torch.as_tensor(ns, dtype=torch.float32).view(-1, 4, bs, bs)
        a_t = torch.as_tensor(a, dtype=torch.int64).unsqueeze(1)
        r_t = torch.as_tensor(r, dtype=torch.float32).unsqueeze(1)
        d_t = torch.as_tensor(d, dtype=torch.float32).unsqueeze(1)
        w_t = torch.as_tensor(weights, dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_net(s_t).gather(1, a_t)
        with torch.no_grad():
            best_actions = self.policy_net(ns_t).argmax(1, keepdim=True)
            next_q = self.target_net(ns_t).gather(1, best_actions)
            target = r_t + self.gamma * next_q * (1.0 - d_t)

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

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location='cpu'))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
