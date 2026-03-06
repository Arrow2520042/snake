import random
import collections
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None


class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim=9, n_actions=3, lr=1e-3, gamma=0.99, batch_size=64, capacity=10000, device=None):
        if torch is None:
            raise RuntimeError('PyTorch is required for DQNAgent')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_net = SimpleNet(state_dim, n_actions).to(self.device)
        self.target_net = SimpleNet(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = collections.deque(maxlen=capacity)
        self.steps = 0
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.9995
        # optimize cuDNN when using CUDA
        try:
            if self.device.startswith('cuda'):
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def act(self, state):
        self.steps += 1
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(torch.argmax(q).item())

    def push(self, state, action, reward, next_state, done):
        # use np.asarray to avoid ValueError with copy=False in NumPy 2.0
        s = np.asarray(state, dtype=np.float32).copy()
        ns = np.asarray(next_state, dtype=np.float32).copy()
        self.replay.append((s, int(action), float(reward), ns, bool(done)))

    def sample(self):
        batch = random.sample(self.replay, self.batch_size)
        s, a, r, ns, d = zip(*batch)
        # return numpy arrays (will be converted to tensors in update)
        return np.array(s, dtype=np.float32), np.array(a, dtype=np.int64), np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32), np.array(d, dtype=np.uint8)

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        s, a, r, ns, d = self.sample()
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(s_t).gather(1, a_t)
        with torch.no_grad():
            next_q = self.target_net(ns_t).max(1)[0].unsqueeze(1)
            target = r_t + self.gamma * next_q * (1.0 - d_t)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # eps decay
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
            if self.eps < self.eps_min:
                self.eps = self.eps_min

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.sync_target()
