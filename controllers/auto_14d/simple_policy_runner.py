# simple_policy_runner.py
import torch
import numpy as np
import time
import dill
import sys
from pathlib import Path

# add diffusion_policy to path
# ROOT_DIR = str(Path(__file__).parent.parent / "6d_on_dp" / "diffusion_policy")
# sys.path.append(ROOT_DIR)
# from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace

ROOT_DIR = "/home/gd/Desktop/6d_on_dp_for_soarm100_pick_place_in_webots/diffusion_policy"
sys.path.insert(0, ROOT_DIR)

from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace

class SimplePolicyRunner:
    def __init__(self, checkpoint_path, num_inference_steps = 4, update_every = 8):
        print("Initializing SimplePolicyRunner")
        
        # load policy
        self.policy = self.load_policy(checkpoint_path)
        
        # speed optimizations
        self.policy.num_inference_steps = num_inference_steps
        self.update_every = update_every
        
        # buffers
        self.obs_history = []
        self.action_buffer = []
        self.action_idx = 0
        self.step_counter = 0
        
        # stats
        self.inference_count = 0
        self.total_inference_time = 0
        
        print(f"\n[Success] Policy loaded:")
        print(f"  • Inference steps: {self.policy.num_inference_steps}")
        print(f"  • Action dim: {self.policy.action_dim}")
        print(f"  • Obs dim: {self.policy.obs_dim}")
        print(f"  • n_obs_steps: {self.policy.n_obs_steps}")
        print(f"  • Update every: {self.update_every} timesteps")
        print("_" * 50)
    
    # **************** load policy from checkpoint ****************
    def load_policy(self, checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', pickle_module = dill)
        saved_cfg = checkpoint['cfg']
        
        print(f"\n=== Config from Checkpoint ===")
        print(f"  obs_dim: {saved_cfg.task.get('obs_dim', 'N/A')}")
        print(f"  action_dim: {saved_cfg.task.get('action_dim', 'N/A')}")
        
        # create workspace
        workspace = TrainDiffusionUnetLowdimWorkspace(saved_cfg)
        
        # load checkpoint WITHOUT the strict parameter
        # the workspace.load_checkpoint method adds its own kwargs
        workspace.load_checkpoint(path = checkpoint_path)
        
        policy = workspace.model
        policy.eval()
        
        return policy
    
    # **************** update observation history ****************
    def update_observation(self, current_obs):
        # check observation dimension
        expected_dim = self.policy.obs_dim
        current_dim = len(current_obs)
        
        if current_dim != expected_dim:
            print(f"[!] Observation dim mismatch: {current_dim} != {expected_dim}")
        
        self.obs_history.append(current_obs.copy())
        
        # keep only last n_obs_steps
        if len(self.obs_history) > self.policy.n_obs_steps:
            self.obs_history.pop(0)
        
        # pad if needed
        while len(self.obs_history) < self.policy.n_obs_steps:
            self.obs_history.insert(0, current_obs.copy())
    
    def get_action(self):
        # **************** et next action to execute ****************
        self.step_counter += 1
        
        # check if we need new inference
        needs_inference = (
            len(self.action_buffer) == 0 or 
            self.action_idx >= len(self.action_buffer) or
            self.step_counter % self.update_every == 0
        )
        
        if needs_inference:
            self._run_policy_inference()
            self.action_idx = 0
        
        # get current action
        if len(self.action_buffer) > 0 and self.action_idx < len(self.action_buffer):
            action = self.action_buffer[self.action_idx]
            self.action_idx += 1
        else:
            # fallback: zero action
            action = np.zeros(self.policy.action_dim, dtype = np.float32)
        
        return action
    
    def _run_policy_inference(self):
        # **************** run policy inference ****************
        if len(self.obs_history) < self.policy.n_obs_steps:
            print(f"Warning: Not enough observations ({len(self.obs_history)}/{self.policy.n_obs_steps})")
            self.action_buffer = []
            return
        
        # prepare observation
        obs_seq = np.stack(self.obs_history[-self.policy.n_obs_steps:], axis = 0)
        obs_seq = obs_seq.reshape(1, self.policy.n_obs_steps, -1)
        obs_tensor = torch.from_numpy(obs_seq).float()
        
        # run inference
        start_time = time.time()
        
        with torch.no_grad():
            result = self.policy.predict_action({'obs': obs_tensor})
        
        inference_time = time.time() - start_time
        
        # update stats
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # get action sequence
        self.action_buffer = result['action'].cpu().numpy()[0]
        
        # log
        avg_time = self.total_inference_time / max(self.inference_count, 1)
        # print(f"Inference #{self.inference_count}: {inference_time*1000:.1f}ms "
        #       f"(avg: {avg_time*1000:.1f}ms) | "
        #       f"Actions: {len(self.action_buffer)}")
    
    def reset(self):
        # **************** reset for new episode ****************
        self.obs_history = []
        self.action_buffer = []
        self.action_idx = 0
        self.step_counter = 0
        self.inference_count = 0
        self.total_inference_time = 0
        # print("Policy runner reset")
    
    def get_stats(self):
        # **************** get performance statistics ****************
        if self.inference_count > 0:
            avg_time = self.total_inference_time / self.inference_count
            return {
                'inference_count': self.inference_count,
                'avg_inference_time_ms': avg_time * 1000,
                'fps': 1 / avg_time if avg_time > 0 else 0
            }
        return {
            'inference_count': 0,
            'avg_inference_time_ms': 0,
            'fps': 0
        }

# ────────────────────────────────────────────────────────────────
# Test function
# ────────────────────────────────────────────────────────────────
def test_runner():
    CHECKPOINT_PATH = "/home/gd/Desktop/6d_on_dp_for_soarm100_pick_place_in_webots/diffusion_policy/data/outputs/2026.02.15/11.07.56_train_diffusion_unet_lowdim_so100_pick_place_lowdim/checkpoints/latest.ckpt"
    print("\nTesting SimplePolicyRunner...")
    
    runner = SimplePolicyRunner(
        checkpoint_path = CHECKPOINT_PATH,
        num_inference_steps = 4,
        update_every = 8
    )
    
    # simulate Webots loop
    np.random.seed(42)
    for i in range(30):
        # simulate 14D observation
        dummy_obs = np.random.randn(runner.policy.obs_dim).astype(np.float32)
        
        runner.update_observation(dummy_obs)
        action = runner.get_action()
        
        # if i % 10 == 0:
        #     print(f"Step {i:2d}: Action shape = {action.shape}, norm = {np.linalg.norm(action):.3f}")
    
    stats = runner.get_stats()
    # print("\n" + "_" * 50)
    # print("Performance Statistics:")
    # print(f"Total inferences: {stats['inference_count']}")
    # print(f"Avg inference time: {stats['avg_inference_time_ms']:.1f} ms")
    
    return runner


if __name__ == "__main__":
    test_runner()