# Snake AI Project - TODOs

## Completed (v2 refactor)

- [x] Cell-based internal state `(cx, cy)` tuples (eliminates pixel↔cell conversion)
- [x] Double DQN (action selection from policy_net, evaluation from target_net)
- [x] Soft target update (τ=0.005) instead of hard sync every 100 steps
- [x] Prioritized Experience Replay (SumTree-based, α=0.6, β annealing 0.4→1.0)
- [x] Extended state vector: 18 features (was 9) - ray distances, snake length
- [x] Distance reward shaping (+0.1 closer, -0.1 farther) + anti-loop penalty (-0.3)
- [x] O(1) collision detection via `snake_body_set = set(snake[1:])`
- [x] Efficient food placement via free cells list + `random.choice()`
- [x] CNN agent alternative (`cnn_agent.py`) with 4-channel grid observation
- [x] Remove all CUDA/device code (CPU-only)
- [x] Remove tabular Q-learning mode
- [x] Replay viewer (`replay_viewer.py`)
- [x] Hyperparameter sweep (`sweep.py`)
- [x] Curriculum learning support (`--board-size` parameter)
- [x] Fixed `choose_action_with_debug` dead code bug
- [x] Cleaned up unnecessary try/except pass blocks

## Breaking changes

- Old `.pth` checkpoints are incompatible (state_dim 9→18, hidden 128→256)
- `Point` namedtuple removed from game logic (now uses plain tuples)
- `resize_window` no longer takes `preserve_state` parameter
- `rl_agent.py` (tabular) deleted

## Remaining roadmap

- [ ] Extract `live_trainer.py` from `game.py` __main__ block
- [ ] Add training metrics panel: avg reward/N episodes, episode length, win-rate
- [ ] Save checkpoint metadata (eps, gamma, level, date) alongside `.pth`
- [ ] Add unit tests: collision, state encoding, action mapping, level load/save
- [ ] Add seed support and deterministic mode for reproducible experiments
- [ ] Add FPS/step profiler and DQN benchmark
- [ ] Implement curriculum learning scheduler (auto board-size progression)
- [ ] Add Dueling DQN architecture option
