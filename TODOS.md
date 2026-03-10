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

## Completed (v3 improvements)

- [x] Parallel vectorized environments (N envs round-robin) in train.py
- [x] Checkpoint save/load with full training state (eps, optimizer, steps, target_net)
- [x] Periodic checkpoint saves every N episodes (configurable `--save-every`)
- [x] Save checkpoint metadata (eps, level, date) alongside .pth + run_info.txt
- [x] Snake head green color for visibility
- [x] Anti-loop improvements: hunger timer, per-food timeout, deque 16→32
- [x] Freeze-on-death/win screen with Resume/Stop buttons
- [x] Stop button in live training
- [x] Mid-episode pause with step-through
- [x] Max score display accounting for walls (`Score: X/MAX`)
- [x] Resume training toggle (ON/OFF) in Train on Level submenu
- [x] Cursor-based text input (arrow keys, Home/End, Delete) in submenu & level designer
- [x] food_timeout end-of-episode label in live demo
- [x] Removed classic snake / manual play mode
- [x] `--num-envs 8` default in headless GUI training subprocess

## Completed (v4 reward fixes)

- [x] Food reward always positive (was -5 for tight spaces → now min +2)
- [x] Reduced space-awareness penalty (removed body_ratio scaling, capped at -0.3)
- [x] Slower epsilon decay (0.9995 → 0.9999) for more exploration with parallel envs
- [x] Cached flood_fill in play_step, reused in get_state (saves 1 BFS/step)
- [x] Tail-chase Manhattan distance feature added (state_dim 22→23)

## Completed (v5 architecture upgrade)

- [x] Dueling DQN architecture (value/advantage stream separation)
- [x] N-step returns (n=5, per-env accumulator, compatible with PER)
- [x] Breaking: new architecture incompatible with v4 checkpoints

## Completed (v6 look-ahead & trap-food)

- [x] Per-action flood fill features (simulate each action, compute reachable space)
- [x] State vector 23→26: 3 new action_flood features (straight, right, left)
- [x] Trap-food penalty: eating food with post_eat_ratio < 0.10 gives -5 reward
- [x] Breaking: state_dim 23→26, old checkpoints incompatible

## Breaking changes

- Old `.pth` checkpoints are incompatible (state_dim 9→18, hidden 128→256)
- `Point` namedtuple removed from game logic (now uses plain tuples)
- `resize_window` no longer takes `preserve_state` parameter
- `rl_agent.py` (tabular) deleted
- v4 reward changes: retrain or use `--fresh` with existing weights
- v4 state_dim 22→23: all old checkpoints incompatible
- v5 Dueling DQN + N-step: all old checkpoints incompatible (new network class)
- v6 state_dim 23→26: all old checkpoints incompatible

## Remaining roadmap

### High priority (training quality)
- [x] LR scheduling (ReduceLROnPlateau after score stagnation)
- [x] Add Dueling DQN architecture option (better value/advantage separation)
- [x] Add training graphs (matplotlib plots via analyze_logs.py --plot)", "oldString": "### High priority (training quality)\n- [x] LR scheduling (ReduceLROnPlateau after score stagnation)\n- [ ] Add Dueling DQN architecture option (better value/advantage separation)\n- [x] Add training graphs (matplotlib plots via analyze_logs.py --plot)

### Medium priority (tooling & experiments)
- [ ] Implement curriculum learning scheduler (auto board-size progression)
- [ ] Add seed support and deterministic mode for reproducible experiments
- [ ] Implement CNN agent training pipeline (full-board vision alternative)
- [x] Add training metrics panel: avg reward/N episodes, episode length, win-rate

### Low priority (code quality)
- [ ] Extract `live_trainer.py` from `game.py` __main__ block
- [ ] Add unit tests: collision, state encoding, action mapping, level load/save
- [ ] Add FPS/step profiler and DQN benchmark
