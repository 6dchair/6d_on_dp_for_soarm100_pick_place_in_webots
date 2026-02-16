import zarr
import torch
import numpy as np
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
import copy

class SO100ZarrDataset(BaseLowdimDataset):
    def __init__(self, path, horizon = 16, pad_before = 0, pad_after = 0, 
                 validation_split = 0.2, mode = 'train'):
        self.store = zarr.open(path, mode = 'r')
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.validation_split = validation_split
        self.mode = mode 

        self.episode_ends = self.store["episode_ends"][:]
        self.valid_indices = self._compute_valid_indices()

        # split into train and validation indices
        self._split_train_val_indices()
        
        # set current indices based on mode
        if self.mode == 'train':
            self.current_indices = self.train_indices
            print(f"Created TRAIN dataset with {len(self.current_indices)} samples")
        elif self.mode == 'val':
            self.current_indices = self.val_indices
            print(f"Created VALIDATION dataset with {len(self.current_indices)} samples")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _compute_valid_indices(self):
        valid = []
        start = 0
        for end in self.episode_ends:
            last_start = end - (self.horizon + self.pad_after)
            if last_start >= start:
                valid.extend(
                    range(
                        start + self.pad_before,
                        last_start + 1
                    )
                )
            start = end
        return np.array(valid, dtype = np.int64)
    
    def _split_train_val_indices(self):
        np.random.seed(42)
        
        episode_indices = []
        start = 0
        for end in self.episode_ends:
            episode_start = np.searchsorted(self.valid_indices, start, side = 'left')
            episode_end = np.searchsorted(
                self.valid_indices,
                end - (self.horizon + self.pad_after),
                side = 'right'
            )
            episode_indices.append(self.valid_indices[episode_start:episode_end])
            start = end
        
        self.train_indices = []
        self.val_indices = []
        
        for ep_indices in episode_indices:
            if len(ep_indices) == 0:
                continue
                
            shuffled = np.random.permutation(ep_indices)
            split_idx = int(len(shuffled) * (1 - self.validation_split))
            self.train_indices.extend(shuffled[:split_idx])
            self.val_indices.extend(shuffled[split_idx:])
        
        self.train_indices = np.array(self.train_indices, dtype = np.int64)
        self.val_indices = np.array(self.val_indices, dtype = np.int64)
        
        print(f"Dataset split: {len(self.valid_indices)} total, "
              f"{len(self.train_indices)} train, {len(self.val_indices)} validation")

    def __len__(self): 
        return len(self.current_indices)
    
    def __getitem__(self, idx):
        start_idx = self.current_indices[idx]
        end_idx = start_idx + self.horizon

        obs = self.store["obs"]["state"][start_idx:end_idx]
        action = self.store["action"][start_idx:end_idx]

        return {
            "obs": obs.astype(np.float32),  # ensure float32
            "action": action.astype(np.float32)
        }

    # **************** return a validation dataset instance ****************
    def get_validation_dataset(self):
        # create a copy with validation mode
        val_dataset = copy.copy(self)
        val_dataset.mode = 'val'
        val_dataset.current_indices = self.val_indices
        print(f"get_validation_dataset() returning dataset with {len(val_dataset.current_indices)} samples")
        return val_dataset

    def get_normalizer(self):
        normalizer = LinearNormalizer()

        # always use training data for normalization
        obs = self.store["obs"]["state"]
        act = self.store["action"]

        # we need to get ALL observations & actions from training indices
        # but training indices are for start positions, not all timesteps
        # so we need to collect all timesteps from horizon windows
        
        all_obs = []
        all_actions = []
        
        for start_idx in self.train_indices:
            end_idx = start_idx + self.horizon
            all_obs.append(obs[start_idx:end_idx])
            all_actions.append(act[start_idx:end_idx])
        
        # concatenate all
        all_obs = np.concatenate(all_obs, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        data = {
            "obs": torch.from_numpy(all_obs).float(),
            "action": torch.from_numpy(all_actions).float()
        }

        print(f"\n**************** NORMALIZER FITTING ****************")
        print(f"Fitting normalizer with {len(all_obs)} obs samples and {len(all_actions)} action samples")
        
        normalizer.fit(data)
        
        print(f"Normalizer fitted with keys: {list(normalizer.params_dict.keys())}")
        print("**************** END NORMALIZER FITTING ****************\n")
        
        return normalizer