# Sim-to-Sim Transfer of Low-Dimensional Diffusion Policy for SO-ARM100 Pick-and-Place Using Webots Demonstrations
> February 2026

---

## Overview

This repo contains the full implementation for my thesis investigating the **sim-to-sim generalization** of a low-dimensional [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) for robotic pick-and-place using the SO-ARM100 arm in Webots.

A complete pipeline was developed covering:
- Leader–follower teleoperated data collection in Webots
- Dataset construction with randomized object and goal configurations
- Offline training of a conditional diffusion model (1D U-Net backbone)
- Autonomous policy deployment across four novel Webots environments

The trained policy operates on a **14-dimensional state representation** (joint positions + object and goal poses) with no visual input, and is evaluated without fine-tuning across 4 novel simulated configurations.

---

## Repository Structure

```
.
├── controllers/              # Webots robot controller scripts
│                             #   - Teleoperation + data collection controller + environment randomizer
│                             #   - Autonomous policy runner (inference)
│
├── worlds/                   # Webots world files (.wbt)
│                             #   - Data collection world
│                             #   - Four novel inference worlds
│
├── protos/                   # Custom PROTO definitions
│                             #   - SO-ARM100 modified PROTO (stabilized)
|                             #   - Table proto
│
├── diffusion_policy/         # Adapted training pipeline
│                             #   - Custom dataset loader (.py)
│                             #   - Custom task and workspace configs (.yaml)
│                             #   - Built on Chi et al. (2023) codebase
│
└── utility_scripts/          # Helper utilities
    ├── dataset_analyzer.py   # Inspect and visualize collected Zarr datasets
    └── merge_zarrs.py        # Merge multiple per-session Zarr files into one
```

---

## Setup

### Prerequisites

- [Webots R2025a](https://cyberbotics.com/)
- Python 3.8+
- PyTorch (CUDA recommended for training)
- The [official Diffusion Policy repository](https://github.com/real-stanford/diffusion_policy) dependencies

### Install Python dependencies

```bash
pip install -r diffusion_policy/requirements.txt
```

> The training pipeline is adapted from Chi et al. (2023). Please also follow the setup instructions in the original Diffusion Policy repository.

---

## Usage

### 1. Data Collection

Open the collection world in Webots:

```
worlds/collection_world.wbt
```

Run the teleoperation controller (assigned to the SO-ARM100 follower node). Connect the SO-ARM101 leader hardware via USB serial before starting.

**Keyboard controls during collection:**

| Key | Action |
|-----|--------|
| `R` | Start recording episode |
| `S` | Stop, save episode, randomize environment |
| `D` | Discard current episode |
| `F` | Save all episodes to Zarr format |
| `Esc` | Exit Webots |

Episodes are saved per-session. Use `utility_scripts/merge_zarrs.py` to combine sessions:

```bash
python utility_scripts/merge_zarrs.py --input_dir ./sessions/ --output ./dataset.zarr
```

### 2. Training

```bash
cd diffusion_policy
python train.py --config-name=low_dim_soarm100_pick_place
```

Training logs and checkpoints are tracked via [Weights & Biases](https://wandb.ai/). Set your W&B credentials before running.

Key training configuration (see `diffusion_policy/config/`):

| Parameter | Value |
|-----------|-------|
| Architecture | Conditional U-Net 1D |
| Observation dim | 28 (14D × 2 timesteps) |
| Action dim | 6 (joint deltas) |
| Diffusion steps | 100 (DDPM) |
| Epochs | 200 |
| Batch size | 64 |
| Learning rate | 1e-4 (AdamW) |

### 3. Evaluation (Autonomous Inference)

Load one of the four inference worlds in Webots:

```
worlds/inference_world_1_expanded_workspace.wbt
worlds/inference_world_2_novel_orientations.wbt
worlds/inference_world_3_novel_arm_init.wbt
worlds/inference_world_4_combined.wbt
```

Point the policy runner to your trained checkpoint:

```python
# In controllers/policy_runner/policy_runner.py
CHECKPOINT_PATH = "path/to/your/checkpoint.ckpt"
```

The runner uses **5 diffusion inference steps** (vs. 100 during training) and caches 8-step action sequences for real-time execution in Webots.

---

## Dataset

The 300-episode dataset collected for this study is included in this repository (Zarr format). Each episode captures causally aligned state–action pairs at 10 Hz.

**State (14D):** 6 joint positions + 4D box pose (x, y, z, yaw) + 4D goal pose (x, y, z, yaw)  
**Action (6D):** Delta joint commands (5 arm joints + gripper)

Object and goal positions were randomized within a 0.10 m × 0.25 m bounded region. Box orientations were sampled from `{0°, 15°, 30°, 45°, 60°, 75°, 90°}` during collection.

---

## Notes on the SO-ARM100 PROTO

The SO-ARM100 PROTO included in this repo is a **modified version** of the original from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100). Modifications were necessary to stabilize the arm in Webots:

- All components wrapped in a parent `Solid` node
- Physics properties replaced with simplified approximations
- Minor color adjustments for visibility

These changes alter the original dynamic specification but are necessary for stable simulation. See Appendix B of the thesis for a full comparison of the real hardware, the leader (SO-ARM101), and the simulated PROTO.

---

## Citation

If you use this work, please cite:

```bibtex
@thesis{torsino2026sim2sim,
  title     = {Sim-to-Sim Transfer of Low-Dimensional Diffusion Policy for SO-ARM100 Pick-and-Place Using Webots Demonstrations},
  author    = {Torsino, Gracedave J.},
  school    = {Caraga State University -- Main Campus},
  year      = {2026}
}
```

Also cite the original Diffusion Policy paper:

```bibtex
@inproceedings{chi2023diffusion,
  title     = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author    = {Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year      = {2023}
}
```

---

## Acknowledgments

This project was conducted at the AI and Robotics Laboratory (AIRLab), Caraga State University, under the supervision of Professor Rudolph Joshua U. Candare. The training pipeline is adapted from the official [Diffusion Policy repository](https://github.com/real-stanford/diffusion_policy) by Stanford and Toyota Research Institute.

---

## License

This repository is released for academic and research use. See `LICENSE` for details.
