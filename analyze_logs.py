import csv, statistics, sys, os

def analyze(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    scores = [int(r['score']) for r in rows]
    rewards = [float(r['total_reward']) for r in rows]
    steps = [int(r['steps']) for r in rows]

    print(f'Total episodes: {len(scores)}')
    print(f'Score: min={min(scores)} max={max(scores)} mean={statistics.mean(scores):.1f} median={statistics.median(scores):.1f} stdev={statistics.stdev(scores):.1f}')
    print(f'Reward: min={min(rewards):.1f} max={max(rewards):.1f} mean={statistics.mean(rewards):.1f}')
    print(f'Steps: min={min(steps)} max={max(steps)} mean={statistics.mean(steps):.0f}')
    print()

    # Dynamic windows based on total episodes
    n = len(scores)
    block = max(200, n // 10)
    for start in range(0, n, block):
        end = min(start + block, n)
        sl = scores[start:end]
        rl = rewards[start:end]
        stl = steps[start:end]
        print(f'ep{start+1:>6}-{end:<6}: n={len(sl):>4} avgScore={statistics.mean(sl):>5.1f} maxScore={max(sl):>3} avgReward={statistics.mean(rl):>7.1f} avgSteps={statistics.mean(stl):>6.0f}')

    return scores, rewards, steps


def plot_training(path, save_dir=None):
    """Generate training progress plots from a rewards.csv file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed, skipping plots')
        return

    scores, rewards, steps = analyze(path)
    n = len(scores)
    episodes = list(range(1, n + 1))

    # Rolling averages
    window = 50
    avg_scores = [sum(scores[max(0,i-window+1):i+1]) / len(scores[max(0,i-window+1):i+1]) for i in range(n)]
    avg_rewards = [sum(rewards[max(0,i-window+1):i+1]) / len(rewards[max(0,i-window+1):i+1]) for i in range(n)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Training Progress ({n} episodes)', fontsize=14)

    # Score plot
    axes[0].plot(episodes, scores, alpha=0.15, color='steelblue', linewidth=0.5)
    axes[0].plot(episodes, avg_scores, color='navy', linewidth=1.5, label=f'avg{window}')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reward plot
    axes[1].plot(episodes, rewards, alpha=0.15, color='coral', linewidth=0.5)
    axes[1].plot(episodes, avg_rewards, color='darkred', linewidth=1.5, label=f'avg{window}')
    axes[1].set_ylabel('Total Reward')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Steps plot
    avg_steps = [sum(steps[max(0,i-window+1):i+1]) / len(steps[max(0,i-window+1):i+1]) for i in range(n)]
    axes[2].plot(episodes, avg_steps, color='green', linewidth=1.5, label=f'avg{window}')
    axes[2].set_ylabel('Steps per Episode')
    axes[2].set_xlabel('Episode')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir is None:
        save_dir = os.path.dirname(path)
    out_path = os.path.join(save_dir, 'training_plot.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'\nPlot saved: {out_path}')

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else r'E:\vsc projects\snake\logs\20260309-000302\rewards.csv'
    plot = '--plot' in sys.argv or '-p' in sys.argv
    if plot:
        plot_training(path)
    else:
        analyze(path)
