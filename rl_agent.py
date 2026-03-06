import numpy as np

class QLearningAgent:
    """Tabular Q-learning agent for the Snake environment.
    State representation expected: [danger_s,danger_r,danger_l,dir_up,dir_down,dir_left,dir_right,food_dx,food_dy]
    We discretize food_dx/food_dy into bins and pack the rest into a small discrete index.
    """

    def __init__(self, n_actions=3, bins=11, lr=0.1, gamma=0.99, eps_start=1.0, eps_min=0.01, eps_decay=0.9995):
        self.n_actions = n_actions
        self.bins = bins
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # compute number of discrete states: dangers (8) * directions (4) * bins * bins
        self.n_dangers = 8
        self.n_dirs = 4
        self.n_states = self.n_dangers * self.n_dirs * self.bins * self.bins
        self.q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def _state_to_idx(self, state):
        # state: list/array length 9 as documented
        ds = int(state[0])  # danger_straight
        dr = int(state[1])  # danger_right
        dl = int(state[2])  # danger_left
        # pack dangers into 0..7
        danger_idx = (ds << 2) | (dr << 1) | (dl << 0)

        # directions: state[3]=up,4=down,5=left,6=right
        # map to indices consistently: up=0, right=1, down=2, left=3
        up = int(bool(state[3]))
        down = int(bool(state[4]))
        left = int(bool(state[5]))
        right = int(bool(state[6]))
        if up:
            dir_idx = 0
        elif right:
            dir_idx = 1
        elif down:
            dir_idx = 2
        elif left:
            dir_idx = 3
        else:
            dir_idx = 0

        # food_dx, food_dy in approx range [-1,1]
        fx = float(state[7])
        fy = float(state[8])
        bx = int(((fx + 1.0) / 2.0) * (self.bins - 1))
        by = int(((fy + 1.0) / 2.0) * (self.bins - 1))
        bx = max(0, min(self.bins - 1, bx))
        by = max(0, min(self.bins - 1, by))

        idx = (((danger_idx * self.n_dirs) + dir_idx) * self.bins + bx) * self.bins + by
        return int(idx)

    def act(self, state):
        idx = self._state_to_idx(state)
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.n_actions)
        return int(np.argmax(self.q[idx]))

    def learn(self, state, action, reward, next_state, done):
        s = self._state_to_idx(state)
        ns = self._state_to_idx(next_state)
        qsa = self.q[s, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q[ns])
        self.q[s, action] = qsa + self.lr * (target - qsa)

        # decay eps
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
            if self.eps < self.eps_min:
                self.eps = self.eps_min

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(path)
        if self.q.shape[1] != self.n_actions:
            raise ValueError('Loaded Q-table action dimension mismatch')
