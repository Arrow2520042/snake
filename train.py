"""Headless training script for Snake DQN/CNN agent.

Supports N parallel environments (single-threaded, round-robin stepping)
for higher throughput.  CUDA used automatically when available.

Usage:
    python train.py --episodes 1000
    python train.py --simple-rewards --board-size 10 --num-envs 64
    python train.py --v12-from-scratch --episodes 300000
    python train.py --level levels/mymap.json --board-size 20
    python train.py --init-checkpoint model.pth --episodes 500
"""

import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import torch
from game import SnakeGameAI


def _load_walls(level_path, board_blocks):
    """Load wall set from a level JSON file."""
    with open(level_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    walls = set()
    for item in data:
        cx, cy = int(item[0]), int(item[1])
        if 0 <= cx < board_blocks and 0 <= cy < board_blocks:
            walls.add((cx, cy))
    return walls


def train(episodes=1000, max_steps=15000, save_every=200, level_path=None,
          init_checkpoint=None, log_name=None, save_path='model.pth',
          board_size=20, num_envs=1, fresh=False,
          agent_type='cnn', simple_rewards=False,
          action_masking=True,
          reward_mode='blend', reward_switch_start=0.25,
          reward_switch_end=0.35,
          eps_start=None, eps_min=None, eps_decay=None,
          resume_eps_boost=False, resume_eps_mult=2.5,
          resume_eps_min=0.04, resume_eps_max=0.12,
          rollback_eps_mult=2.5, rollback_eps_min=0.04,
          rollback_eps_max=0.12, rollback_eps_hold_episodes=3000,
          eval_every=5000, eval_episodes=20,
          eval_max_steps=None, eval_seed=12345,
          rollback_source='auto', scheduler_source='auto',
          walls_mode=False,
          instant_eval_drop_ratio=0.50,
          npz_save_every=2000):
    """Main training orchestrator.

    Responsibilities in this function:
    - Build environments and choose agent architecture.
    - Drive the parallel episode loop and checkpoint policy.
    - Handle exploration schedules and rollback logic.

    Optimizer details (Adam), target updates, and LR scheduler internals live
    inside agent.update()/agent.step_scheduler() in dqn_agent.py/cnn_agent.py.
    """

    state_mode = 'grid' if agent_type == 'cnn' else 'features'

    # 1) Select policy architecture (feature MLP or convolutional grid model).
    if agent_type == 'cnn':
        from cnn_agent import CNNAgent
        agent = CNNAgent(board_size=board_size)
    else:
        from dqn_agent import DQNAgent
        agent = DQNAgent()

    effective_reward_mode = 'simple' if simple_rewards else str(reward_mode).lower()
    if effective_reward_mode not in ('simple', 'complex', 'blend'):
        raise ValueError('reward_mode must be one of: simple, complex, blend')
    action_masking = bool(action_masking)

    reward_switch_start = max(0.0, min(0.99, float(reward_switch_start)))
    reward_switch_end = max(reward_switch_start + 1e-6, min(1.0, float(reward_switch_end)))
    resume_eps_boost = bool(resume_eps_boost)
    resume_eps_mult = float(resume_eps_mult)
    resume_eps_min = float(resume_eps_min)
    resume_eps_max = max(resume_eps_min, float(resume_eps_max))
    npz_save_every = max(1, int(npz_save_every))

    # 2) Optional checkpoint restore for resume/fine-tuning workflows.
    if init_checkpoint and os.path.isfile(init_checkpoint):
        agent.load(init_checkpoint, weights_only=fresh)
        if fresh:
            print(f'Loaded weights only: {init_checkpoint} (fresh training state: eps={agent.eps:.4f})')
        else:
            print(f'Loaded checkpoint: {init_checkpoint} (eps={agent.eps:.4f}, steps={agent.steps})')

        # Optional epsilon boost right after resume to reintroduce exploration.
        if (not fresh) and resume_eps_boost and eps_start is None:
            old_eps = agent.eps
            agent.eps = min(max(old_eps * resume_eps_mult, resume_eps_min), resume_eps_max)
            print(f'  >> Resume epsilon boost: {old_eps:.4f} -> {agent.eps:.4f}')

    # 3) Explicit epsilon overrides are applied after loading checkpoint state.
    if eps_start is not None:
        agent.eps = float(eps_start)
    if eps_min is not None:
        agent.eps_min = float(eps_min)
    if eps_decay is not None:
        agent.eps_decay = float(eps_decay)

    # Keep rollback LR aligned with the loaded checkpoint state (or explicit override),
    # not with Adam's constructor default.
    base_lr = float(agent.optimizer.param_groups[0]['lr'])
    scheduler_patience = 10 if agent_type == 'cnn' else 50

    walls = None
    wall_count = 0
    wall_ratio = 0.0
    if level_path and os.path.isfile(level_path):
        walls = _load_walls(level_path, board_size)
        wall_count = len(walls)
        total_cells = max(1, board_size * board_size)
        wall_ratio = wall_count / float(total_cells)
        print(f'Loaded level: {level_path} ({len(walls)} walls)')
    if walls_mode and not walls:
        raise ValueError('--walls mode requires a valid --level file')

    # 4) Create parallel environments (single process, round-robin stepping).
    envs = []
    for _ in range(num_envs):
        env = SnakeGameAI(render=False, board_blocks=board_size,
                          max_episode_steps=max_steps,
                          state_mode=state_mode,
                          simple_rewards=(effective_reward_mode == 'simple'),
                          reward_mode=effective_reward_mode,
                          reward_switch_start=reward_switch_start,
                          reward_switch_end=reward_switch_end)
        if walls:
            env.walls = walls
        envs.append(env)

    def _run_deterministic_eval(eval_round_idx):
        """Run greedy evaluation (eps=0, action mask on) on fixed seeds."""
        old_eps = agent.eps
        old_steps = agent.steps
        was_training = agent.policy_net.training

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        agent.eps = 0.0
        agent.policy_net.eval()

        eval_env = SnakeGameAI(render=False, board_blocks=board_size,
                               max_episode_steps=eval_max_steps,
                               state_mode=state_mode,
                               simple_rewards=(effective_reward_mode == 'simple'),
                               reward_mode=effective_reward_mode,
                               reward_switch_start=reward_switch_start,
                               reward_switch_end=reward_switch_end)
        if walls:
            eval_env.walls = set(walls)

        scores = []
        step_counts = []
        try:
            for ep_idx in range(eval_episodes):
                seed_val = eval_seed + eval_round_idx * 10007 + ep_idx
                random.seed(seed_val)
                np.random.seed(seed_val & 0xFFFFFFFF)
                torch.manual_seed(seed_val)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_val)

                state = eval_env.reset()
                done = False
                steps = 0
                while (not done) and steps < eval_max_steps:
                    mask = eval_env.get_safe_action_mask()
                    action = int(agent.act(state, action_mask=mask))
                    state, _, done, _ = eval_env.play_step(action, skip_events=True)
                    steps += 1

                scores.append(float(eval_env.score))
                step_counts.append(float(steps))
        finally:
            agent.eps = old_eps
            agent.steps = old_steps
            if was_training:
                agent.policy_net.train()

            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

        avg_score = float(np.mean(scores)) if scores else 0.0
        avg_steps = float(np.mean(step_counts)) if step_counts else 0.0
        return avg_score, avg_steps

    device_name = getattr(agent, 'device', 'cpu')
    print(f'Agent: {agent_type.upper()} | device: {device_name} | '
          f'reward_mode: {effective_reward_mode} '
          f'({reward_switch_start:.2f}->{reward_switch_end:.2f}) | '
            f'action_masking: {action_masking} | '
            f'n_step: {agent.n_step} | eps: {agent.eps:.4f}/{agent.eps_min:.4f}')

    # 5) Training log setup (metadata + compressed metric snapshots).
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if log_name:
        log_dir = os.path.join('logs', log_name, ts)
    else:
        log_dir = os.path.join('logs', ts)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, 'rewards.npz')
    # Accumulate arrays in memory, flush periodically
    all_scores = []
    all_rewards = []
    all_steps = []

    try:
        info_path = os.path.join(log_dir, 'info.txt')
        with open(info_path, 'w', encoding='utf-8') as fi:
            fi.write(f'agent_type: {agent_type}\n')
            fi.write(f'simple_rewards: {effective_reward_mode == "simple"}\n')
            fi.write(f'reward_mode: {effective_reward_mode}\n')
            fi.write(f'reward_switch_start: {reward_switch_start:.3f}\n')
            fi.write(f'reward_switch_end: {reward_switch_end:.3f}\n')
            fi.write(f'action_masking: {action_masking}\n')
            fi.write(f'device: {getattr(agent, "device", "cpu")}\n')
            fi.write(f'level_path: {level_path}\n')
            fi.write(f'init_checkpoint: {init_checkpoint}\n')
            fi.write(f'episodes: {episodes}\n')
            fi.write(f'max_steps: {max_steps}\n')
            fi.write(f'npz_save_every: {npz_save_every}\n')
            fi.write(f'board_size: {board_size}\n')
            fi.write(f'walls_mode: {bool(walls_mode)}\n')
            fi.write(f'wall_count: {wall_count}\n')
            fi.write(f'wall_ratio: {wall_ratio:.6f}\n')
            fi.write(f'num_envs: {num_envs}\n')
            fi.write(f'n_step: {agent.n_step}\n')
            fi.write(f'eps_start: {agent.eps:.6f}\n')
            fi.write(f'eps_min: {agent.eps_min:.6f}\n')
            fi.write(f'eps_decay: {agent.eps_decay:.7f}\n')
            fi.write(f'resume_eps_boost: {resume_eps_boost}\n')
            fi.write(f'resume_eps_mult: {resume_eps_mult:.3f}\n')
            fi.write(f'resume_eps_min: {resume_eps_min:.3f}\n')
            fi.write(f'resume_eps_max: {resume_eps_max:.3f}\n')
            fi.write(f'rollback_eps_mult: {rollback_eps_mult:.3f}\n')
            fi.write(f'rollback_eps_min: {rollback_eps_min:.3f}\n')
            fi.write(f'rollback_eps_max: {rollback_eps_max:.3f}\n')
            fi.write(f'rollback_eps_hold_episodes: {rollback_eps_hold_episodes}\n')
            fi.write(f'eval_every: {eval_every}\n')
            fi.write(f'eval_episodes: {eval_episodes}\n')
            fi.write(f'eval_max_steps: {eval_max_steps}\n')
            fi.write(f'eval_seed: {eval_seed}\n')
            fi.write(f'instant_eval_drop_ratio: {instant_eval_drop_ratio:.3f}\n')
            fi.write(f'rollback_source: {rollback_source}\n')
            fi.write(f'scheduler_source: {scheduler_source}\n')
            fi.write(f'lr_start: {agent.optimizer.param_groups[0]["lr"]:.1e}\n')
            fi.write(f'started: {ts}\n')
    except Exception:
        pass

    print(f'Training {episodes} episodes | board={board_size} | '
            f'envs={num_envs} | updates/round={max(1, num_envs // 16)} | '
            f'npz_every={npz_save_every} | logs -> {log_dir}')

    best_score = 0
    METRIC_WINDOW = 200
    best_avg200 = 0.0
    best_ckpt_path = os.path.join(log_dir, 'best.pth')
    best_eval_avg = -1.0
    best_eval_ckpt_path = os.path.join(log_dir, 'best_eval.pth')
    eval_points = []
    eval_avg_scores = []
    eval_avg_steps = []
    latest_eval_score = None
    instant_eval_rollbacks = 0
    stagnation_counter = 0
    STAGNATION_THRESHOLD = 15000  # episodes without meaningful improvement before rollback
    STAGNATION_TOLERANCE = 0.9   # tolerate avg200 down to 90% of best before counting as stagnation
    ROLLBACK_EPS_THRESHOLD = 0.3  # don't rollback while still exploring (eps >= 0.3)
    rollback_eps_hold_episodes = max(0, int(rollback_eps_hold_episodes))
    rollback_eps_mult = float(rollback_eps_mult)
    rollback_eps_min = float(rollback_eps_min)
    rollback_eps_max = max(rollback_eps_min, float(rollback_eps_max))
    eval_every = max(0, int(eval_every))
    eval_episodes = max(1, int(eval_episodes))
    eval_max_steps = max_steps if eval_max_steps is None else max(1, int(eval_max_steps))
    eval_seed = int(eval_seed)
    instant_eval_drop_ratio = float(instant_eval_drop_ratio)
    if not (0.0 <= instant_eval_drop_ratio < 1.0):
        raise ValueError('instant_eval_drop_ratio must be in [0.0, 1.0)')
    rollback_source = str(rollback_source).lower()
    if rollback_source not in ('auto', 'avg200', 'eval'):
        raise ValueError('rollback_source must be one of: auto, avg200, eval')
    scheduler_source = str(scheduler_source).lower()
    if scheduler_source not in ('auto', 'avg200', 'eval'):
        raise ValueError('scheduler_source must be one of: auto, avg200, eval')
    scheduler_eps_gate = max(0.02, agent.eps_min * 2.0)
    eps_hold_until_ep = -1
    eps_hold_floor = 0.0
    updates_per_round = max(1, num_envs // 16)  # e.g. 128 envs → 8 updates/round
    t_start = time.time()
    recent_scores = []
    recent_rewards = []
    recent_losses = []
    recent_qvals = []

    # -- parallel episode loop ------------------------------------------
    # Each env runs its own episode concurrently; we round-robin step them.
    states = [env.reset() for env in envs]
    rewards_acc = [0.0] * num_envs
    steps_acc = [0] * num_envs
    ep_counter = 0  # total finished episodes

    while ep_counter < episodes:
        # Keep a temporary epsilon floor after rollback so exploration does not collapse too quickly.
        if ep_counter < eps_hold_until_ep and agent.eps < eps_hold_floor:
            agent.eps = eps_hold_floor

        # Batched inference: one forward pass for all active environments.
        action_masks = None
        if action_masking:
            # Action masking removes one-step certain-death actions before sampling.
            action_masks = [env.get_safe_action_mask() for env in envs]
        actions = agent.act_batch(states, action_masks=action_masks)

        # Step all envs with their actions
        for i, env in enumerate(envs):
            if ep_counter >= episodes:
                break
            action = int(actions[i])
            next_state, reward, done, info = env.play_step(action, skip_events=True)
            agent.push(i, states[i], action, reward, next_state, done,
                       snake_length=len(env.snake))
            rewards_acc[i] += reward
            steps_acc[i] += 1
            states[i] = next_state

            if done or steps_acc[i] >= max_steps:
                ep_counter += 1
                score = env.score
                ep_reward = rewards_acc[i]
                recent_scores.append(score)
                recent_rewards.append(ep_reward)
                # Use 200-episode moving averages for lower-variance progress tracking.
                avg200 = sum(recent_scores[-METRIC_WINDOW:]) / len(recent_scores[-METRIC_WINDOW:])
                avgR200 = sum(recent_rewards[-METRIC_WINDOW:]) / len(recent_rewards[-METRIC_WINDOW:])

                if score > best_score:
                    best_score = score

                # Track best avg200 and save best checkpoint
                if len(recent_scores) >= METRIC_WINDOW:
                    if avg200 > best_avg200:
                        best_avg200 = avg200
                        stagnation_counter = 0
                        agent.save(best_ckpt_path)
                    elif avg200 >= best_avg200 * STAGNATION_TOLERANCE:
                        # Within tolerance band — don't count as stagnation
                        pass
                    else:
                        stagnation_counter += 1

                    if (stagnation_counter >= STAGNATION_THRESHOLD
                            and agent.eps < ROLLBACK_EPS_THRESHOLD):
                        candidates = []
                        if rollback_source == 'avg200':
                            candidates = [
                                (best_ckpt_path, f'avg200={best_avg200:.2f}'),
                            ]
                        elif rollback_source == 'eval':
                            candidates = [
                                (best_eval_ckpt_path, f'eval_avg={best_eval_avg:.2f}'),
                                (best_ckpt_path, f'avg200 fallback={best_avg200:.2f}'),
                            ]
                        else:  # auto
                            candidates = [
                                (best_eval_ckpt_path, f'eval_avg={best_eval_avg:.2f}'),
                                (best_ckpt_path, f'avg200 fallback={best_avg200:.2f}'),
                            ]

                        rollback_path = None
                        rollback_metric = 'n/a'
                        for path, metric_desc in candidates:
                            if os.path.isfile(path):
                                rollback_path = path
                                rollback_metric = metric_desc
                                break

                        if rollback_path is None:
                            stagnation_counter = 0
                            continue

                        old_eps = agent.eps
                        agent.load(rollback_path)

                        # Adaptive exploration boost after rollback.
                        new_eps = min(max(old_eps * rollback_eps_mult, rollback_eps_min), rollback_eps_max)
                        agent.eps = new_eps
                        if rollback_eps_hold_episodes > 0:
                            eps_hold_floor = max(rollback_eps_min, new_eps * 0.60)
                            eps_hold_until_ep = ep_counter + rollback_eps_hold_episodes

                        # Reset LR and scheduler after rollback.
                        for param_group in agent.optimizer.param_groups:
                            param_group['lr'] = base_lr

                        agent.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            agent.optimizer, mode='max', factor=0.5,
                            patience=scheduler_patience, min_lr=1e-5
                        )

                        stagnation_counter = 0
                        print(
                            f'  >> ROLLBACK to {os.path.basename(rollback_path)} '
                            f'({rollback_metric}) at ep {ep_counter} | '
                            f'eps: {old_eps:.4f}->{agent.eps:.4f} | '
                            f'eps_floor={eps_hold_floor:.4f} for next {rollback_eps_hold_episodes} eps'
                        )

                all_scores.append(score)
                all_rewards.append(ep_reward)
                all_steps.append(steps_acc[i])

                if ep_counter % npz_save_every == 0:
                    np.savez_compressed(log_path,
                        scores=np.array(all_scores, dtype=np.int16),
                        rewards=np.array(all_rewards, dtype=np.float16),
                        steps=np.array(all_steps, dtype=np.uint16),
                        losses=np.array(recent_losses, dtype=np.float32),
                        qvals=np.array(recent_qvals, dtype=np.float32),
                        eval_eps=np.array(eval_points, dtype=np.int32),
                        eval_scores=np.array(eval_avg_scores, dtype=np.float32),
                        eval_steps=np.array(eval_avg_steps, dtype=np.float32))

                if ep_counter % 1000 == 0 or ep_counter == 1:
                    elapsed = time.time() - t_start
                    lr = agent.optimizer.param_groups[0]['lr']
                    eps_per_sec = ep_counter / max(1e-6, elapsed)
                    steps_per_sec = sum(all_steps) / max(1e-6, elapsed)
                    avgSteps = (
                        sum(all_steps[-METRIC_WINDOW:])
                        / max(1, len(all_steps[-METRIC_WINDOW:]))
                    )
                    avg_loss = sum(recent_losses[-200:]) / max(1, len(recent_losses[-200:])) if recent_losses else 0.0
                    avg_q = sum(recent_qvals[-200:]) / max(1, len(recent_qvals[-200:])) if recent_qvals else 0.0
                    print(f'ep {ep_counter}/{episodes} | '
                          f'best={best_score} avg200={avg200:.1f} '
                          f'avgR200={avgR200:.1f} avgSteps={avgSteps:.0f} | '
                          f'loss={avg_loss:.4f} Q={avg_q:.2f} | '
                          f'eps/s={eps_per_sec:.2f} steps/s={steps_per_sec:.0f} | '
                          f'eps={agent.eps:.4f} lr={lr:.1e} '
                          f'elapsed={elapsed:.0f}s')
                    # LR scheduler is meaningful only when epsilon is already low.
                    # During high exploration, score variance is too high for reliable LR decisions.
                    if agent.eps <= scheduler_eps_gate:
                        scheduler_metric = best_avg200
                        if scheduler_source in ('eval', 'auto') and latest_eval_score is not None:
                            scheduler_metric = latest_eval_score
                        agent.step_scheduler(scheduler_metric)

                # Deterministic policy check (eps=0 + action mask ON) for deployment-quality selection.
                if eval_every > 0 and ep_counter % eval_every == 0:
                    eval_round_idx = ep_counter // eval_every
                    eval_score, eval_steps = _run_deterministic_eval(eval_round_idx)
                    eval_points.append(ep_counter)
                    eval_avg_scores.append(eval_score)
                    eval_avg_steps.append(eval_steps)
                    latest_eval_score = eval_score

                    if eval_score > best_eval_avg:
                        best_eval_avg = eval_score
                        agent.save(best_eval_ckpt_path)
                        eval_status = 'new best_eval'
                    else:
                        eval_status = f'best_eval={best_eval_avg:.2f}'

                    # Safety fuse: instant rollback if eval collapses by configured ratio.
                    drop_limit = best_eval_avg * (1.0 - instant_eval_drop_ratio)
                    if (
                            instant_eval_drop_ratio > 0.0
                            and best_eval_avg > 0.0
                            and eval_score < drop_limit
                            and os.path.isfile(best_eval_ckpt_path)
                    ):
                        old_eps = agent.eps
                        agent.load(best_eval_ckpt_path)

                        # Keep rollback behavior consistent with stagnation rollback.
                        new_eps = min(max(old_eps * rollback_eps_mult, rollback_eps_min), rollback_eps_max)
                        agent.eps = new_eps
                        if rollback_eps_hold_episodes > 0:
                            eps_hold_floor = max(rollback_eps_min, new_eps * 0.60)
                            eps_hold_until_ep = ep_counter + rollback_eps_hold_episodes

                        for param_group in agent.optimizer.param_groups:
                            param_group['lr'] = base_lr

                        agent.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            agent.optimizer, mode='max', factor=0.5,
                            patience=scheduler_patience, min_lr=1e-5
                        )

                        latest_eval_score = best_eval_avg
                        stagnation_counter = 0
                        instant_eval_rollbacks += 1
                        eval_status = (
                            f'instant rollback (eval {eval_score:.2f} < {drop_limit:.2f}) | '
                            f'best_eval={best_eval_avg:.2f}'
                        )

                    print(
                        f'  >> eval@{ep_counter}: avg_score={eval_score:.2f} '
                        f'avg_steps={eval_steps:.1f} | {eval_status}'
                    )

                if ep_counter % save_every == 0:
                    agent.save(os.path.join(log_dir, f'model_ep{ep_counter}.pth'))

                # Reset this env for next episode
                states[i] = env.reset()
                rewards_acc[i] = 0.0
                steps_acc[i] = 0

        # 6) Optimization stage.
        # Perform several optimizer updates per environment round.
        # This keeps replay consumption and model updates in balance for large num_envs.
        if len(agent.replay) >= 10000:
            for _ in range(updates_per_round):
                loss_val, q_val = agent.update()
                if loss_val is not None:
                    recent_losses.append(loss_val)
                    recent_qvals.append(q_val)
                    
        agent.decay_epsilon()  # once per round, not per gradient update

    # -- save final models ----------------------------------------------
    np.savez_compressed(log_path,
        scores=np.array(all_scores, dtype=np.int16),
        rewards=np.array(all_rewards, dtype=np.float16),
        steps=np.array(all_steps, dtype=np.uint16),
        losses=np.array(recent_losses, dtype=np.float32),
        qvals=np.array(recent_qvals, dtype=np.float32),
        eval_eps=np.array(eval_points, dtype=np.int32),
        eval_scores=np.array(eval_avg_scores, dtype=np.float32),
        eval_steps=np.array(eval_avg_steps, dtype=np.float32))
    agent.save(os.path.join(log_dir, save_path))
    agent.save(save_path)
    if eval_points:
        print(
            f'Done. Best score: {best_score}. '
            f'Best eval avg score: {best_eval_avg:.2f}. '
            f'Model saved to {save_path}'
        )
    else:
        print(f'Done. Best score: {best_score}. Model saved to {save_path}')

    try:
        run_info = os.path.join(log_dir, 'run_info.txt')
        with open(run_info, 'w', encoding='utf-8') as ri:
            ri.write(f'best_score: {best_score}\n')
            ri.write(f'best_avg200: {best_avg200:.6f}\n')
            ri.write(f'instant_eval_drop_ratio: {instant_eval_drop_ratio:.6f}\n')
            ri.write(f'instant_eval_rollbacks: {instant_eval_rollbacks}\n')
            ri.write(f'rollback_source: {rollback_source}\n')
            ri.write(f'scheduler_source: {scheduler_source}\n')
            if eval_points:
                ri.write(f'best_eval_avg_score: {best_eval_avg:.6f}\n')
                ri.write(f'best_eval_path: {best_eval_ckpt_path}\n')
                ri.write(f'eval_points: {len(eval_points)}\n')
            ri.write(f'final_eps: {agent.eps:.6f}\n')
            ri.write(f'final_lr: {agent.optimizer.param_groups[0]["lr"]:.1e}\n')
            ri.write(f'total_time: {time.time() - t_start:.1f}s\n')
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Snake DQN agent (headless)')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=15000)
    parser.add_argument('--save-every', type=int, default=10000)
    parser.add_argument('--npz-save-every', type=int, default=2000,
                        help='Flush rewards.npz every N episodes (higher reduces I/O overhead)')
    parser.add_argument('--level', type=str, default=None, help='Path to level JSON')
    parser.add_argument('--init-checkpoint', type=str, default=None, help='Path to .pth to resume')
    parser.add_argument('--log-name', type=str, default=None, help='Subdirectory name for logs')
    parser.add_argument('--save', type=str, default='model.pth', help='Final model filename')
    parser.add_argument('--board-size', type=int, default=20, help='Board size (curriculum learning)')
    parser.add_argument('--num-envs', type=int, default=128, help='Parallel environments (e.g. 64-128)')
    parser.add_argument('--agent', type=str, default='cnn', choices=['dqn', 'cnn'],
                        help='Agent type: dqn (feature vector) or cnn (grid observation)')
    parser.add_argument('--simple-rewards', action='store_true',
                        help='Use simplified reward: +10 food, -10 death, -0.01 step')
    parser.add_argument('--no-action-masking', action='store_true',
                        help='Disable one-step death action masking (enabled by default)')
    parser.add_argument('--reward-mode', type=str, default='blend',
                        choices=['simple', 'complex', 'blend'],
                        help='Reward strategy: simple, complex, or occupancy-based blend')
    parser.add_argument('--reward-switch-start', type=float, default=0.25,
                        help='Blend start occupancy ratio (e.g. 0.25)')
    parser.add_argument('--reward-switch-end', type=float, default=0.35,
                        help='Blend end occupancy ratio (e.g. 0.35)')
    parser.add_argument('--eps-start', type=float, default=None,
                        help='Override epsilon start value after loading checkpoint (or from scratch)')
    parser.add_argument('--eps-min', type=float, default=None,
                        help='Override epsilon minimum value')
    parser.add_argument('--eps-decay', type=float, default=None,
                        help='Override epsilon decay per training round')
    parser.add_argument('--resume-eps-boost', action='store_true',
                        help='Boost epsilon immediately after loading checkpoint')
    parser.add_argument('--resume-eps-mult', type=float, default=2.5,
                        help='Multiplier for epsilon right after checkpoint resume')
    parser.add_argument('--resume-eps-min', type=float, default=0.04,
                        help='Minimum epsilon right after checkpoint resume boost')
    parser.add_argument('--resume-eps-max', type=float, default=0.12,
                        help='Maximum epsilon right after checkpoint resume boost')
    parser.add_argument('--rollback-eps-mult', type=float, default=2.5,
                        help='Multiplier for epsilon after rollback')
    parser.add_argument('--rollback-eps-min', type=float, default=0.04,
                        help='Minimum epsilon after rollback boost')
    parser.add_argument('--rollback-eps-max', type=float, default=0.12,
                        help='Maximum epsilon after rollback boost')
    parser.add_argument('--rollback-eps-hold', type=int, default=3000,
                        help='Hold temporary epsilon floor for this many episodes after rollback')
    parser.add_argument('--eval-every', type=int, default=5000,
                        help='Run deterministic eval every N episodes (0 disables)')
    parser.add_argument('--eval-episodes', type=int, default=20,
                        help='Episodes per deterministic eval checkpoint test')
    parser.add_argument('--eval-max-steps', type=int, default=None,
                        help='Max steps per deterministic eval episode (default: --max-steps)')
    parser.add_argument('--eval-seed', type=int, default=12345,
                        help='Base seed for deterministic eval reproducibility')
    parser.add_argument('--instant-eval-drop-ratio', type=float, default=0.50,
                        help='Instant rollback when eval drops by this fraction from best_eval (0 disables)')
    parser.add_argument('--walls', action='store_true',
                        help='Enable layout fine-tuning autotuning (requires --level)')
    parser.add_argument('--rollback-source', type=str, default='auto',
                        choices=['auto', 'avg200', 'eval'],
                        help='Checkpoint source used for rollback: avg200, eval, or auto')
    parser.add_argument('--scheduler-source', type=str, default='auto',
                        choices=['auto', 'avg200', 'eval'],
                        help='Metric source for LR scheduler: avg200, eval, or auto')
    parser.add_argument('--v12-from-scratch', action='store_true',
                        help='Apply recommended V12 config for training from scratch')
    parser.add_argument('--fresh', action='store_true',
                        help='Load weights only, reset training state (eps, optimizer, steps)')
    args = parser.parse_args()

    if args.v12_from_scratch:
        default_episodes = parser.get_default('episodes')
        default_save_every = parser.get_default('save_every')
        default_save = parser.get_default('save')
        using_checkpoint = bool(args.init_checkpoint)

        args.agent = 'cnn'
        args.board_size = 10
        args.num_envs = 128
        args.reward_mode = 'blend'
        args.reward_switch_start = 0.25
        args.reward_switch_end = 0.35
        args.simple_rewards = False
        if not using_checkpoint:
            args.fresh = False
        args.log_name = args.log_name or 'v12'
        if args.episodes == default_episodes:
            args.episodes = 300000
        if args.save_every == default_save_every:
            args.save_every = 10000
        if args.save == default_save:
            args.save = 'v12_b10_from_scratch.pth'
        if args.eps_start is None and not using_checkpoint:
            args.eps_start = 1.0
        if args.eps_min is None:
            args.eps_min = 0.01
        if args.eps_decay is None:
            args.eps_decay = 0.99975
        args.rollback_eps_mult = 2.5
        args.rollback_eps_min = 0.04
        args.rollback_eps_max = 0.12
        args.rollback_eps_hold = 3000

        if using_checkpoint and args.eps_start is None:
            args.resume_eps_boost = True
            args.resume_eps_mult = 2.5
            args.resume_eps_min = 0.04
            args.resume_eps_max = 0.12
            print('Note: --v12-from-scratch + --init-checkpoint detected; enabling resume epsilon boost.')

    if args.walls:
        if not args.level:
            parser.error('--walls requires --level <layout.json>.')
        if not os.path.isfile(args.level):
            parser.error(f'--walls level file not found: {args.level}')

        map_walls = _load_walls(args.level, args.board_size)
        wall_count = len(map_walls)
        total_cells = max(1, args.board_size * args.board_size)
        free_ratio = max(1, total_cells - wall_count) / float(total_cells)

        if args.eps_start is None and args.init_checkpoint:
            args.eps_start = 0.05
        if args.eval_episodes == parser.get_default('eval_episodes'):
            args.eval_episodes = 50
        if args.rollback_source == parser.get_default('rollback_source'):
            args.rollback_source = 'eval'
        if args.scheduler_source == parser.get_default('scheduler_source'):
            args.scheduler_source = 'eval'
        if args.log_name is None:
            args.log_name = 'walls'

        if args.reward_switch_start == parser.get_default('reward_switch_start'):
            args.reward_switch_start = max(0.02, min(0.95, args.reward_switch_start * free_ratio))
        if args.reward_switch_end == parser.get_default('reward_switch_end'):
            args.reward_switch_end = max(
                args.reward_switch_start + 0.02,
                min(1.0, args.reward_switch_end * free_ratio),
            )

        print(
            f'Walls mode: {wall_count} walls ({(100.0 * wall_count / total_cells):.1f}% blocked) | '
            f'reward_switch={args.reward_switch_start:.2f}->{args.reward_switch_end:.2f} | '
            f'eval_episodes={args.eval_episodes} | '
            f'rollback={args.rollback_source} scheduler={args.scheduler_source}'
        )

    train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_every=args.save_every,
        npz_save_every=args.npz_save_every,
        level_path=args.level,
        init_checkpoint=args.init_checkpoint,
        log_name=args.log_name,
        save_path=args.save,
        board_size=args.board_size,
        num_envs=args.num_envs,
        fresh=args.fresh,
        agent_type=args.agent,
        simple_rewards=args.simple_rewards,
        action_masking=not args.no_action_masking,
        reward_mode=args.reward_mode,
        reward_switch_start=args.reward_switch_start,
        reward_switch_end=args.reward_switch_end,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        resume_eps_boost=args.resume_eps_boost,
        resume_eps_mult=args.resume_eps_mult,
        resume_eps_min=args.resume_eps_min,
        resume_eps_max=args.resume_eps_max,
        rollback_eps_mult=args.rollback_eps_mult,
        rollback_eps_min=args.rollback_eps_min,
        rollback_eps_max=args.rollback_eps_max,
        rollback_eps_hold_episodes=args.rollback_eps_hold,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        eval_seed=args.eval_seed,
        rollback_source=args.rollback_source,
        scheduler_source=args.scheduler_source,
        walls_mode=args.walls,
        instant_eval_drop_ratio=args.instant_eval_drop_ratio,
    )
