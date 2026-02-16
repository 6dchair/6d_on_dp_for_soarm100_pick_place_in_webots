"""
python merge_zarrs_14d.py \
    --input_pattern="/home/gd/Desktop/6d_on_dp_for_soarm100_pick_place_in_webots/diffusion_policy/data/so100_lowdim/14D/batches_0_1_2/*.zarr" \
    --output_path="/home/gd/Desktop/6d_on_dp_for_soarm100_pick_place_in_webots/diffusion_policy/data/so100_lowdim/14D/300_eps.zarr" \
    --clear_existing \
"""
import zarr
import numpy as np
import glob
import os
import shutil
from tqdm import tqdm

def merge_flat_zarrs(input_pattern, output_path, clear_existing = False):
    zarr_files = sorted(glob.glob(input_pattern))
    zarr_files = [f for f in zarr_files if f != output_path]

    if not zarr_files:
        raise RuntimeError(f"No Zarr files found for pattern: {input_pattern}")

    print(f"Found {len(zarr_files)} Zarr files")

    if clear_existing and os.path.exists(output_path):
        print(f"Removing existing output: {output_path}")
        shutil.rmtree(output_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok = True)

    merged_action = []
    merged_obs = {}
    merged_episode_ends = []

    # **************** store metadata from all files ****************
    all_state_means = []
    all_action_means = []
    all_state_stds = []
    all_action_stds = []

    timestep_offset = 0
    obs_keys = None

    for path in tqdm(zarr_files, desc = "Merging Zarr files"):
        store = zarr.open(path, mode = "r")

        # actions
        action = store['action'][:]
        merged_action.append(action)

        # episode ends
        ep_ends = store['episode_ends'][:] + timestep_offset
        merged_episode_ends.append(ep_ends)

        # observations
        obs_group = store['obs']
        current_keys = [k for k in obs_group.keys() if hasattr(obs_group[k], 'shape')]
        
        if obs_keys is None:
            obs_keys = current_keys
            for k in obs_keys:
                merged_obs[k] = []
        else:
            if set(current_keys) != set(obs_keys):
                raise RuntimeError(f"Obs keys mismatch in {path}")

        for k in obs_keys:
            merged_obs[k].append(obs_group[k][:])

        # **************** collect metadata from source file ****************
        if hasattr(store, 'attrs'):
            if 'state_mean' in store.attrs:
                all_state_means.append(store.attrs['state_mean'])
            if 'action_mean' in store.attrs:
                all_action_means.append(store.attrs['action_mean'])
            if 'state_std' in store.attrs:
                all_state_stds.append(store.attrs['state_std'])
            if 'action_std' in store.attrs:
                all_action_stds.append(store.attrs['action_std'])

        timestep_offset += action.shape[0]

    # **************** convert to float32 during concatenation ****************
    merged_action = np.concatenate(merged_action, axis = 0).astype(np.float32)
    merged_episode_ends = np.concatenate(merged_episode_ends, axis = 0).astype(np.int64)
    for k in merged_obs:
        merged_obs[k] = np.concatenate(merged_obs[k], axis = 0).astype(np.float32)

    # **************** save to new Zarr ****************
    root = zarr.open(output_path, mode = 'w')
    root.create_array('action', data = merged_action, chunks = (min(1024, len(merged_action)), merged_action.shape[1]))
    root.create_array('episode_ends', data = merged_episode_ends, chunks = (len(merged_episode_ends),))

    obs_out = root.create_group('obs')
    for k, arr in merged_obs.items():
        obs_out.create_array(k, data = arr, chunks = (min(1024, len(arr)),) + arr.shape[1:])

    # **************** compute FINAL statistics on FULL merged data ****************
    print("\nComputing normalization statistics on merged dataset...")
    
    # compute state statistics
    state_data = merged_obs['state']
    state_mean = np.mean(state_data, axis = 0).astype(np.float32)
    state_std = np.std(state_data, axis = 0).astype(np.float32)
    
    # add small epsilon to avoid division by zero
    state_std = np.maximum(state_std, 1e-6)
    
    # compute action statistics
    action_mean = np.mean(merged_action, axis = 0).astype(np.float32)
    action_std = np.std(merged_action, axis = 0).astype(np.float32)
    action_std = np.maximum(action_std, 1e-6)
    
    # **************** save COMPLETE metadata ****************
    root.attrs.update({
        # normalization statistics
        'state_mean': state_mean.tolist(),
        'state_std': state_std.tolist(),
        'action_mean': action_mean.tolist(),
        'action_std': action_std.tolist(),

        # dataset info
        'description': 'Merged SO-ARM100 keyboard control dataset for Diffusion Policy',
        'control_method': 'keyboard_teleop',
        'recording_frequency_hz': 10.0,
        'action_dim': merged_action.shape[1],
        'state_dim': state_data.shape[1],
        "state_components": "6q1..q5, g, box(xyz,yaw), goal(xyz,yaw)",
        "action_components": "Δq1..Δq5 (arm), Δg (gripper)",
        'action_type': 'delta_position',
        'created': 'merged_' + str(np.datetime64('now')),
        'total_frames': len(merged_action),
        'total_episodes': len(merged_episode_ends),
        'robot': 'SO-ARM100',
        'task': 'pick_and_place',
        'dataset_format': 'diffusion_policy_lowdim_v1',
        'has_normalization_stats': True,
        'has_images': False,
        'source_files': zarr_files,
        'merge_timestamp': str(np.datetime64('now'))
    })
    
    # **************** save source statistics for reference ****************
    if all_state_means:
        root.attrs['source_state_means'] = all_state_means
    if all_action_means:
        root.attrs['source_action_means'] = all_action_means

    print(f"\nMerge complete: {output_path}")
    print(f"Total timesteps: {merged_action.shape[0]}")
    print(f"Total episodes: {len(merged_episode_ends)}")
    
    # episode statistics
    episode_starts = np.concatenate([[0], merged_episode_ends[:-1]])
    episode_lengths = merged_episode_ends - episode_starts
    print(f"Episode lengths - min: {episode_lengths.min()}, max: {episode_lengths.max()}, "
          f"mean: {episode_lengths.mean():.1f}")
    
    print(f"\nNormalization statistics saved:")
    print(f"  State mean shape: {state_mean.shape}")
    print(f"  State std shape: {state_std.shape}")
    print(f"  Action mean shape: {action_mean.shape}")
    print(f"  Action std shape: {action_std.shape}")
    
    # verify normalization
    print(f"\nVerifying normalization:")
    normalized_state = (state_data - state_mean) / state_std
    normalized_action = (merged_action - action_mean) / action_std
    print(f"  Normalized state - mean: {normalized_state.mean(axis = 0)[:3]}, std: {normalized_state.std(axis = 0)[:3]}")
    print(f"  Normalized action - mean: {normalized_action.mean(axis = 0)[:3]}, std: {normalized_action.std(axis = 0)[:3]}")

    return output_path

# ********************************************************************
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Merge multiple flat Zarrs into one")
    parser.add_argument('--input_pattern', type = str, required = True, help = 'Pattern to find Zarr files')
    parser.add_argument('--output_path', type = str, required = True, help = 'Path to save merged Zarr')
    parser.add_argument('--clear_existing', action = 'store_true', help = 'Clsear existing output')
    args = parser.parse_args()

    merge_flat_zarrs(args.input_pattern, args.output_path, args.clear_existing)
