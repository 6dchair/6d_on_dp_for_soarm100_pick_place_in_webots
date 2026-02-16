import time
import math
import numpy as np
import torch
import sys
import pathlib
from controller import Supervisor, Keyboard

# add the optimized runner
ROOT_DIR = "/home/gd/Desktop/6d_on_dp_for_pick_place_in_webots/diffusion_policy"
sys.path.append(ROOT_DIR)

# use the working SimplePolicyRunner
from simple_policy_runner import SimplePolicyRunner

class AutonomousSO100(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = 32  # webots timestep
        
        # initialize keyboard
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
        
        # control modes - renamed to avoid conflict with Supervisor.mode property
        self.control_mode = "autonomous"  # "autonomous" or "manual"
        
        # environment randomization settings
        self.MIN_DISTANCE_BETWEEN_OBJECTS = 0.15
        
        # task completion state
        self.task_completed = False
        self.task_count = 0
        
        # initialize robot components
        self.JOINT_NAMES = ["1", "2", "3", "4", "5", "6"]
        self.GRIPPER_NAME = "6"
        self.motors = {}
        self.sensors = {}
        for name in self.JOINT_NAMES:
            self.motors[name] = self.getDevice(name)
            self.sensors[name] = self.getDevice(f"{name}_sensor")
            if self.sensors[name]:
                self.sensors[name].enable(self.timestep)
        
        # keyboard control setup
        self.KEY_MAP = {
            ord('T'): ("1", -1),
            ord('G'): ("1", 1),
            ord('Y'): ("2", -1),
            ord('H'): ("2", 1),
            ord('U'): ("3", -1),
            ord('J'): ("3", 1),
            ord('I'): ("4", -1),
            ord('K'): ("4", 1),
            ord('O'): ("5", -1),
            ord('L'): ("5", 1),
            ord(','): "open",
            ord('.'): "close",
        }
        
        self.ANGLE_STEP = 0.05
        self.GRIPPER_STEP = 0.05
        
        # objects
        self.target_box = self.getFromDef('TARGET_BOX')
        self.goal_zone = self.getFromDef('GOAL_ZONE')
        
        # wait one step so Webots updates joint sensors
        for _ in range(2):
            self.step(self.timestep)
            
        # sync motors to actual joint angles initially
        for name in self.JOINT_NAMES:
            current = self.safe_get_value(self.sensors[name])
            self.motors[name].setPosition(current)
            self.motors[name].setVelocity(0.5)
        
        # initialize gripper position
        self.gripper_pos = self.safe_get_value(self.sensors[self.GRIPPER_NAME])
        
        # ────────────────────────────────────────────────────────────────
        # Environment randomization setup
        # ────────────────────────────────────────────────────────────────
        try:
            from randomization_controller import AutoDetectRandomizer
            self.randomizer = AutoDetectRandomizer(self)
            print("[Randomizer] AutoDetectRandomizer loaded successfully")
        except ImportError as e:
            print(f"[!] randomization_controller not found: {e}")
            self.randomizer = None
        
        # **************** OPTIMIZED POLICY RUNNER ****************
        CHECKPOINT_PATH = "/home/gd/Desktop/6d_on_dp/diffusion_policy/data/outputs/2026.02.15/11.07.56_train_diffusion_unet_lowdim_so100_pick_place_lowdim/checkpoints/latest.ckpt"    
        print("\nInitializing Autonomous Controller")
        
        self.policy_runner = SimplePolicyRunner(
            checkpoint_path = CHECKPOINT_PATH,
            num_inference_steps = 4, # try 4 steps for ~100ms inference
            update_every = 8 # match n_action_steps
        )
        
        print(f"  Inference: {self.policy_runner.policy.num_inference_steps} diffusion steps")
        print(f"  Update every: {self.policy_runner.update_every} timesteps")
        
        # display control instructions
        self.print_instructions()

    def print_instructions(self):
        print("\n[CONTROL INSTRUCTIONS]")
        print("MODE SWITCHING:")
        print("  P .......... Pause autonomous & switch to MANUAL control")
        print("  R .......... Resume AUTONOMOUS control")
        print("  N .......... Randomize environment and continue")
        print("\nMANUAL CONTROL KEYS:")
        print("  Joint 1 (Yaw) ---------------- T / G")
        print("  Joint 2 (Pitch) -------------- Y / H")
        print("  Joint 3 (Pitch) -------------- U / J")
        print("  Joint 4 (Pitch) -------------- I / K")
        print("  Joint 5 (Yaw) ---------------- O / L")
        print("  Gripper ---------------------- , (Open), . (Close)")
        print(f"\nCURRENT MODE: {self.control_mode.upper()}")
        print(f"TASK STATUS: {'COMPLETED' if self.task_completed else 'IN PROGRESS'}")

    # **************** helper methods ****************
    def safe_get_value(self, sensor):
        if sensor:
            val = sensor.getValue()
            return 0.0 if math.isnan(val) else val
        return 0.0

    def clamp(self, val, lo, hi):
        return max(lo, min(hi, val))

    def get_box_pose(self):
        if not self.target_box:
            return [0.0, 0.0, 0.0, 0.0]
        pos = self.target_box.getField('translation').getSFVec3f()
        rot = self.target_box.getField('rotation').getSFRotation()
        yaw = rot[3] if abs(rot[2]-1.0)<0.001 else 0.0
        return [pos[0], pos[1], pos[2], yaw]

    def get_goal_pose(self):
        if not self.goal_zone:
            return [0.0, 0.0, 0.0, 0.0]
        pos = self.goal_zone.getField('translation').getSFVec3f()
        return [pos[0], pos[1], pos[2], 0.0]

    # **************** get current 14D observation ****************
    def get_state(self):
        # joint positions (6D)
        joint_state = np.array([
            self.safe_get_value(self.sensors[n]) 
            for n in self.JOINT_NAMES
        ], dtype = np.float32)
        
        # box pose (4D: x, y, z, yaw)
        box_pose = np.array(self.get_box_pose(), dtype = np.float32)
        
        # goal pose (4D: x, y, z, 0)
        goal_pose = np.array(self.get_goal_pose(), dtype = np.float32)
        
        # concatenate: [6 joints] + [4 box] + [4 goal] = 14D
        return np.concatenate([joint_state, box_pose, goal_pose])

    # **************** apply autonomous action to robot ****************
    def apply_autonomous_action(self, action):
        # action[0:6] = joint position deltas
        action = np.clip(action, -0.5, 0.5)  # limit max delta per step
        
        # apply joint deltas
        for i, name in enumerate(self.JOINT_NAMES):
            current_pos = self.safe_get_value(self.sensors[name])
            target_pos = current_pos + action[i]
            self.motors[name].setPosition(target_pos)

    # **************** manual control methods ****************
    def move_joint(self, joint_name, direction):
        current_pos = self.safe_get_value(self.sensors[joint_name])
        min_pos = self.motors[joint_name].getMinPosition()
        max_pos = self.motors[joint_name].getMaxPosition()

        if math.isnan(min_pos) or min_pos <= -1e9:
            min_pos = -3.14
        if math.isnan(max_pos) or max_pos >= 1e9:
            max_pos = 3.14

        new_pos = current_pos + direction * self.ANGLE_STEP
        new_pos = self.clamp(new_pos, min_pos, max_pos)
        self.motors[joint_name].setPosition(new_pos)

    def move_gripper(self, action):
        if action == "open":
            self.gripper_pos = self.clamp(self.gripper_pos + self.GRIPPER_STEP, 0.0, 2.0)
        elif action == "close":
            self.gripper_pos = self.clamp(self.gripper_pos - self.GRIPPER_STEP, 0.0, 2.0)
        self.motors[self.GRIPPER_NAME].setPosition(self.gripper_pos)

    # **************** environment randomization ****************
    def randomize_environment(self):
        if self.randomizer:
            try:
                box_pos, box_rot, goal_pos, distance = self.randomizer.randomize_positions(
                    min_distance = self.MIN_DISTANCE_BETWEEN_OBJECTS,
                    max_attempts = 200
                )
                return True
            except Exception as e:
                print(f"[!] Randomization failed: {e}")
                return False
        else:
            print("[!] Randomizer not available")
            return False

    def handle_keyboard(self):
        key = self.keyboard.getKey()
        last_key = -1  # track last processed key
        
        while key != -1:
            # only process if this is a different key than last time
            if key != last_key:
                last_key = key  # remember this key
                
                if key == ord('P') or key == ord('p'):
                    if self.control_mode == "autonomous":
                        self.control_mode = "manual"
                        print("\n[Pause] Switching to MANUAL control mode")
                        print("Use keyboard to control joints, press 'R' to resume autonomous")
                elif key == ord('R') or key == ord('r'):
                    if self.control_mode == "manual":
                        self.control_mode = "autonomous"
                        print("\n[Resume] Switching back to AUTONOMOUS control mode")
                        # reset policy runner when resuming
                        self.policy_runner.reset()
                elif key == ord('N') or key == ord('n'):
                    # randomize environment - works in both modes
                    print("\n[Randomize] Manual randomization triggered")
                    success = self.randomize_environment()
                    
                    if success and self.task_completed:
                        # reset for new task
                        self.task_completed = False
                        self.task_count += 1
                        self.policy_runner.reset()
                        
                        # reset step count
                        if hasattr(self, '_main_loop_step_count'):
                            self._main_loop_step_count = 0
                            print("[Reset]Step count reset to 0")
                        
                        print(f"[Continue] Starting task #{self.task_count + 1}")
                        print("\tPress 'P' to switch to manual mode anytime")
                    elif success:
                        # just randomized without task completion
                        print("[Randomize] Environment randomized, task still in progress")
                    else:
                        print("[Randomize] Failed to randomize environment")
                        
                elif self.control_mode == "manual":
                    # handle manual control keys
                    cmd = self.KEY_MAP.get(key)
                    if cmd:
                        if isinstance(cmd, str):
                            self.move_gripper(cmd)
                        else:
                            joint_name, direction = cmd
                            self.move_joint(joint_name, direction)
            
            key = self.keyboard.getKey()

    def run(self):
        print("\nStarting autonomous pick-and-place...")
        
        # use instance variable for step count so it can be reset
        self._main_loop_step_count = 0
        episode_start = time.time()
        
        # reset policy runner for new episode
        self.policy_runner.reset()
        
        try:
            while self.step(self.timestep) != -1:
                self._main_loop_step_count += 1
                
                # 1. handle keyboard input for mode switching
                self.handle_keyboard()
                
                # 2. autonomous mode
                if self.control_mode == "autonomous":
                    # get current state from sensors
                    current_state = self.get_state()
                    
                    # update policy with new observation
                    self.policy_runner.update_observation(current_state)
                    
                    # get action from policy (with smart caching)
                    action = self.policy_runner.get_action()
                    
                    # apply action to robot
                    self.apply_autonomous_action(action)
                    
                    # log progress
                    if self._main_loop_step_count % 20 == 0:
                        action_norm = np.linalg.norm(action)
                        stats = self.policy_runner.get_stats()
                        
                        # print(f"Step {self._main_loop_step_count:4d}: "
                        #     f"Action norm: {action_norm:.3f}, "
                        #     f"Inferences: {stats['inference_count']}, "
                        #     f"Avg: {stats['avg_inference_time_ms']:.1f}ms")
                        # print(f"Step {self._main_loop_step_count:4d}: ")
                
                # 3. manual mode - already handled by keyboard input in handle_keyboard()
                # the joint movements are applied directly in move_joint() and move_gripper()
                
                # # 4. safety: reset step count after 5000 steps
                # if self._main_loop_step_count >= 5000:
                #     print(f"\n[Reset]Step count reset from {self._main_loop_step_count} to 0")
                #     self._main_loop_step_count = 0
        
        except KeyboardInterrupt:
            print("\n[!] Manual interrupt by user")
        
        finally:
            # final statistics
            episode_time = time.time() - episode_start
            stats = self.policy_runner.get_stats()
            
            print("\n" + "_" * 50)
            print("Episode Summary:")
            print(f"  Total steps: {self._main_loop_step_count}")
            print(f"  Total time: {episode_time:.1f}s")
            print(f"  Steps/sec: {self._main_loop_step_count/episode_time:.1f}")
            print(f"  Policy inferences: {stats['inference_count']}")
            print(f"  Avg inference time: {stats['avg_inference_time_ms']:.1f}ms")
            print(f"  Inference FPS: {stats['fps']:.1f}")
            print(f"  Final mode: {self.control_mode.upper()}")
           
# ********************************************************************
if __name__ == "__main__":
    controller = AutonomousSO100()
    controller.run()