import zarr
import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset(zarr_path):
    data = zarr.open(zarr_path, mode = 'r')
    
    actions = data['action'][:]
    states = data['obs/state'][:]
    episode_ends = data['episode_ends'][:]
    
    print(f"\nDATASET ANALYSIS: {zarr_path}")
    
    # basic stats
    print(f"\nBASIC STATS:")
    print(f"  Total frames: {len(actions):,}")
    print(f"  Total episodes: {len(episode_ends)}")
    
    # **************** episode length distribution ****************
    episode_lengths = []
    start = 0
    for end in episode_ends:
        length = end - start
        episode_lengths.append(length)
        start = end
    
    print(f"\nEPISODE LENGTHS:")
    print(f"  Mean: {np.mean(episode_lengths):.1f} frames")
    print(f"  Std: {np.std(episode_lengths):.1f}")
    print(f"  Min: {np.min(episode_lengths)}")
    print(f"  Max: {np.max(episode_lengths)}")
    print(f"  Target range: 30-50 frames")
    
    # **************** action analysis ****************
    action_norms = np.linalg.norm(actions, axis = 1)
    
    print(f"\nACTION QUALITY:")
    print(f"  Mean norm: {action_norms.mean():.4f}")
    print(f"  Std norm: {action_norms.std():.4f}")
    print(f"  25th: {np.percentile(action_norms, 25):.4f}")
    print(f"  50th: {np.percentile(action_norms, 50):.4f}")
    print(f"  75th: {np.percentile(action_norms, 75):.4f}")
    print(f"  95th: {np.percentile(action_norms, 95):.4f}")
    
    # near-zero actions
    near_zero = np.sum(action_norms < 0.001) / len(actions) * 100
    print(f"  Near-zero actions: {near_zero:.1f}%")
    print(f"  Target: <10%")
    
    # **************** quality assessment ****************
    print(f"\nQUALITY ASSESSMENT:")
    
    issues = []
    
    if len(episode_ends) < 50:
        issues.append(f"[!] Too few episodes ({len(episode_ends)} < 50)")
    
    if np.mean(episode_lengths) < 25:
        issues.append(f"[!] Episodes too short ({np.mean(episode_lengths):.1f} frames)")
    
    if action_norms.mean() < 0.03:
        issues.append(f"[!] Actions too small (mean: {action_norms.mean():.4f})")
    
    if near_zero > 15:
        issues.append(f"[!] Too many idle frames ({near_zero:.1f}%)")
    
    if not issues:
        print("[Good] Dataset looks good for training!")
    else:
        print("  Issues found:")
        for issue in issues:
            print(f"    {issue}")
    
    # plot
    fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    
    # **************** episode lengths ****************
    axes[0].hist(episode_lengths, bins = 20, alpha = 0.7)
    axes[0].axvline(30, color = 'r', linestyle = '--', label = 'Min target')
    axes[0].axvline(50, color = 'g', linestyle = '--', label = 'Max target')
    axes[0].set_xlabel('Episode Length (frames)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Episode Length Distribution')
    axes[0].legend()
    
    # **************** action norms ****************
    axes[1].hist(action_norms, bins = 50, alpha = 0.7, range = (0, 0.1))
    axes[1].axvline(0.03, color = 'r', linestyle = '--', label = 'Min target')
    axes[1].axvline(0.1, color = 'g', linestyle = '--', label = 'Ideal target')
    axes[1].set_xlabel('Action Norm')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Action Norm Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png')
    plt.show()
    
    return {
        'total_frames': len(actions),
        'total_episodes': len(episode_ends),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_action_norm': action_norms.mean(),
        'near_zero_pct': near_zero
    }

if __name__ == "__main__":
    # analyze all datasets in the directory
    import glob
    datasets = glob.glob("/home/gd/Desktop/6d_on_dp_for_soarm100_pick_place_in_webots/diffusion_policy/data/so100_lowdim/14D/merged_10s.zarr")
    
    for dataset in datasets:
        print(f"\nAnalyzing: {dataset}")
        try:
            analyze_dataset(dataset)
        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")