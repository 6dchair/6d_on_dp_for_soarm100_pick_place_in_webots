import serial
import time
from controller import Supervisor, Keyboard
import math
import numpy as np
import random
from datetime import datetime
import zarr
import os
import shutil
import threading
from collections import deque

# ────────────────────────────────────────────────────────────────
# Main controller class
# ────────────────────────────────────────────────────────────────
class LeaderFollowerController(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        self.step_counter = 0

        # **************** episode timing constraints ****************
        self.MIN_EPISODE_TIME = 5.0    
        self.MAX_EPISODE_TIME = 60.0   
        self.POST_SUCCESS_TIME = 3.0   

        self.episode_start_time = None
        self.success_time = None
        
        # **************** webots joint limits ****************
        self.JOINT_LIMITS = {
            "1": (-2.0, 2.0),
            "2": (0.0, 3.5),
            "3": (-3.14159, 0.0),
            "4": (-2.5, 1.2),
            "5": (-3.14159, 3.14159),
            "6": (-0.2, 2.0),  # gripper
        }
        
        # **************** data collection constants (10Hz) ****************
        self.TIME_STEP = 32
        self.record_interval = 0.1  # 10Hz
        self.MAX_EPISODE_FRAMES = int(self.MAX_EPISODE_TIME / self.record_interval)
        self.last_record_time = self.getTime()
        self.MIN_DISTANCE_BETWEEN_OBJECTS = 0.15
        
        self.leader_update_interval = 0.02
        self.last_leader_update = self.getTime()
        
        # **************** add object nodes for tracking ****************
        self.target_box = self.getFromDef('TARGET_BOX')
        self.goal_zone = self.getFromDef('GOAL_ZONE')

        # **************** state and Action Dimensions ****************
        self.STATE_DIM = 14  # 6 joints + 4 box pose + 4 goal pose
        self.ACTION_DIM = 6  # 5 arm joint deltas + 1 gripper delta
        
        # self.last_gripper_pos = 0.0

        # **************** recording state ****************
        self.RECORDING = False
        self.EPISODE_COUNT = 0
        self.current_episode_start_idx = 0
        self.episode_discarded = False
        
        # **************** data buffers ****************
        self.obs_state = []
        self.actions = []
        self.episode_ends = []
        self.episode_starts = []
    
        # **************** object pose debug buffers ****************
        self.box_positions = []
        self.box_orientations = []
        self.goal_positions = []
    
        # **************** leader position buffers ****************
        self.leader_positions = [0.0] * 6
        self.target_positions = [0.0] * 6
        self.position_buffer = deque(maxlen = 3)
        self.last_raw_positions = [0.0] * 6
        
        # **************** threading ****************
        self.serial_thread = None
        self.serial_running = False
        self.serial_lock = threading.Lock()

        # **************** action buffering for causal alignment ****************
        self.pending_action = np.zeros(self.ACTION_DIM, dtype = np.float32)
        self.has_pending_action = False

        
        # ────────────────────────────────────────────────────────────────
        # Webots setup
        # ────────────────────────────────────────────────────────────────
        print("\n" + "_" * 60)
        print("SO-ARM100 LEADER-FOLLOWER - LOW-DIM DATA COLLECTION")
        print("_" * 60)
        
        # **************** enable keyboard ****************
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
        
        # **************** joint setup ****************
        self.JOINT_NAMES = ["1", "2", "3", "4", "5", "6"]
        self.GRIPPER_NAME = "6"
        self.motors = {}
        self.sensors = {}
        
        for name in self.JOINT_NAMES:
            self.motors[name] = self.getDevice(name)
            self.sensors[name] = self.getDevice(f"{name}_sensor")
            if self.sensors[name]:
                self.sensors[name].enable(self.timestep)
        
        # **************** initialize motors ****************
        for _ in range(2):
            self.step(self.timestep)
        
        for name in self.JOINT_NAMES:
            if self.motors[name] and self.sensors[name]:
                self.motors[name].setPosition(self.safe_get_value(self.sensors[name]))
                self.motors[name].setVelocity(1.0)
        
        # **************** gripper sensor ****************
        # self.gripper_sensor = self.getDevice(f"{self.GRIPPER_NAME}_sensor")
        # if self.gripper_sensor:
        #     self.gripper_sensor.enable(self.timestep)
        #     self.gripper_pos = self.safe_get_value(self.gripper_sensor)
        # else:
        #     self.gripper_pos = 0.0

        # ────────────────────────────────────────────────────────────────
        # Environment randomization setup
        # ────────────────────────────────────────────────────────────────
        try:
            from randomization_controller import AutoDetectRandomizer
            self.randomizer = AutoDetectRandomizer(self)
        except ImportError:
            print("[WARNING] randomization_controller not found")
            self.randomizer = None
        
        # ────────────────────────────────────────────────────────────────
        # Leader calibration
        # ────────────────────────────────────────────────────────────────
        self.CALIB = {
            0: {
                "type": "centered",
                "raw_min": 650,
                "raw_max": 4000,
                "raw_center": 2000,
                "web_min": -2.0,
                "web_max": 2.0,
                "web_center": 0.0,
            },
            1: {
                "type": "linear",
                "raw_min": 803,
                "raw_max": 3294,
                "web_min": 0.0,
                "web_max": 3.5,
            },
            2: {
                "type": "linear",
                "raw_min": 853,
                "raw_max": 3064,
                "web_min": -3.14159,
                "web_max": 0.0,
            },
            3: {
                "type": "linear",
                "raw_min": 815,
                "raw_max": 3184,
                "web_min": -2.5,
                "web_max": 1.2,
            },
            4: {
                "type": "centered",
                "raw_min": 0,
                "raw_max": 4200,
                "raw_center": 2050,
                "web_min": -3.14159,
                "web_max": 3.14159,
                "web_center": 0.0,
            },
            5: {
                "type": "gripper",
                "raw_closed": 2030,
                "raw_open": 3258,
                "web_closed": -0.2,
                "web_open": 2.0,
            },
        }
        
        print("[CALIB]", self.CALIB)

        # **************** leader connection ****************
        self.ser = None
        self.leader_connected = False
        self.connect_to_leader()
        
        # **************** start serial reading thread ****************
        if self.leader_connected:
            self.start_serial_thread()
        
        # **************** Initialize target positions ****************
        if self.leader_connected:
            initial_positions = self.read_leader_positions_sync()
            
            for i, (name, motor) in enumerate(self.motors.items()):
                clamped_pos = self.clamp_to_joint_limits(name, initial_positions[i])
                motor.setPosition(clamped_pos)
                self.target_positions[i] = clamped_pos
            
            self.leader_positions = initial_positions
            self.last_raw_positions = initial_positions[:]
        
        # **************** perform initial randomization ****************
        self.randomize_environment()
        
        print("\n" + "_"*60)
        print("\n[CONTROL INSTRUCTIONS]")
        print("     R   ..........    Start recording")
        print("     S   ..........    Stop & save + Randomize")
        print("     D   ..........    Discard episode (no save)")
        print("     F   ..........    Save all data")
        print("     ESC ........    Exit")
        print("_" * 60)
        print(f"\n[STATUS] Recording: {'ON' if self.RECORDING else 'OFF'}")
        print(f"[LEADER] {'Connected' if self.leader_connected else 'Disconnected'}")
        print(f"\n[DATASET SPECIFICATION]")
        print(f"    State dimension:    {self.STATE_DIM}D")
        print(f"    Action dimension:   {self.ACTION_DIM}D")
        print(f"    Recording rate: {1/self.record_interval} Hz")
        print(f"    State:  Joints(6) + Box(4) + Goal(4)")
        print(f"    Action: ΔJoints(5) + ΔGripper(1)")
        print(f"    Mode: Low-dimensional only (no images)")
        print("_"*60)

    # ────────────────────────────────────────────────────────────────
    # Helper functions
    # ────────────────────────────────────────────────────────────────
    def safe_get_value(self, sensor):
        if sensor:
            val = sensor.getValue()
            return 0.0 if math.isnan(val) else val
        return 0.0
    
    def clamp(self, val, lo, hi):
        return max(lo, min(hi, val))
    
    def clamp_to_joint_limits(self, joint_name, position):
        if joint_name in self.JOINT_LIMITS:
            min_limit, max_limit = self.JOINT_LIMITS[joint_name]
            clamped = self.clamp(position, min_limit, max_limit)
            return clamped
        return position

    def get_box_pose_simple(self):
        if not self.target_box:
            return [0.0, 0.0, 0.0, 0.0]  # x,y,z, yaw
        
        pos = self.target_box.getField('translation').getSFVec3f()
        rot = self.target_box.getField('rotation').getSFRotation()
        
        # extract yaw (rotation around Z)
        if abs(rot[0]) < 0.001 and abs(rot[1]) < 0.001 and abs(rot[2] - 1.0) < 0.001:
            yaw = rot[3]
        else:
            yaw = 0.0
        
        return [pos[0], pos[1], pos[2], yaw]

    def get_goal_pose_simple(self):
        if not self.goal_zone:
            return [0.0, 0.0, 0.0, 0.0]
        
        pos = self.goal_zone.getField('translation').getSFVec3f()
        return [pos[0], pos[1], pos[2], 0.0]
    
    # ────────────────────────────────────────────────────────────────
    # Recording functions
    # ────────────────────────────────────────────────────────────────
    # **************** start recording ****************
    def start_recording(self):
        if not self.RECORDING:
            self.RECORDING = True
            self.EPISODE_COUNT += 1
            self.current_episode_start_idx = len(self.obs_state)
            self.episode_starts.append(self.current_episode_start_idx)

            self.episode_start_time = self.getTime()
            self.success_time = None

            # reset action buffer
            self.has_pending_action = False
            self.pending_action[:] = 0.0

            # force immediate first frame
            self.last_record_time = self.getTime() - self.record_interval
            # self.record_step()

            print(f"\n[RECORD] Started episode {self.EPISODE_COUNT} "
                f"(frame {self.current_episode_start_idx})")

    # **************** stop and save episode ****************
    def stop_and_save_episode(self):
        if not self.RECORDING:
            return

        duration = self.getTime() - self.episode_start_time
        frames = len(self.obs_state) - self.current_episode_start_idx

        # **************** enforce minimum episode length ****************
        if duration < self.MIN_EPISODE_TIME or frames < 5:
            print(f"[DISCARD] Episode too short "
                f"({duration:.2f}s, {frames} frames) - rolling back data")

            start_idx = self.current_episode_start_idx
            
            # **************** roll back main data buffers ****************
            self.obs_state = self.obs_state[:start_idx]
            self.actions = self.actions[:start_idx]

            # **************** roll back debug buffers ****************
            self.box_positions = self.box_positions[:start_idx]
            self.box_orientations = self.box_orientations[:start_idx]
            self.goal_positions = self.goal_positions[:start_idx]
            
            # **************** remove episode start marker ****************
            if self.episode_starts:
                self.episode_starts.pop()

            self.RECORDING = False
            return

        self.RECORDING = False
        self.episode_ends.append(len(self.obs_state))

        print(f"[SAVE] Saved episode {self.EPISODE_COUNT} "
            f"({duration:.2f}s, {frames} frames)")
        
    # **************** record step ****************
    def record_step(self):
        # **************** robot state + object poses ****************
        joint_state = np.array(
            [self.safe_get_value(self.sensors[n]) for n in self.JOINT_NAMES],
            dtype = np.float32
        )

        box_pose = self.get_box_pose_simple()      # [x,y,z,yaw]
        goal_pose = self.get_goal_pose_simple()    # [x,y,z,yaw]
        state = np.concatenate([joint_state, box_pose, goal_pose])
        self.obs_state.append(state)

        # **************** action (causal) ****************
        if self.has_pending_action:
            self.actions.append(self.pending_action.copy())
        else:
            # first frame: zero action
            self.actions.append(np.zeros(self.ACTION_DIM, dtype = np.float32))

        # **************** safety check ****************
        assert len(self.obs_state) == len(self.actions)

    # **************** save data ****************
    def save_data(self):
        # save low-dimensional data in Diffusion Policy compatible format
        if len(self.obs_state) == 0:
            print("[WARNING] No data to save")
            return

        # create data directory if it doesn't exist
        data_dir = "/home/gd/Desktop/6d_on_dp/diffusion_policy/data/so100_lowdim"
        os.makedirs(data_dir, exist_ok = True)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zarr_path = os.path.join(data_dir, f"{timestamp}.zarr")
        
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        
        # prepare data dictionary
        data_dict = {}
        
        # actions
        if self.actions:
            data_dict['action'] = np.asarray(self.actions, dtype = np.float32)
        
        # states
        if self.obs_state:
            data_dict['obs/state'] = np.asarray(self.obs_state, dtype = np.float32)
            
        # episode metadata
        if self.episode_ends:
            data_dict['episode_ends'] = np.asarray(self.episode_ends, dtype = np.int64)
        
        # **************** save using zarr.save ****************
        try:
            print(f"[DEBUG] Saving keys: {list(data_dict.keys())}")
            print(f"[DEBUG] State shape: {data_dict['obs/state'].shape}")
            print(f"[DEBUG] Action shape: {data_dict['action'].shape}")

            zarr.save(zarr_path, **data_dict)
            
            # **************** add metadata ****************
            root = zarr.open(zarr_path, mode = 'r+')

            if 'obs/state' in root:
                states = root['obs/state'][:]
                root.attrs['state_mean'] = states.mean(axis = 0).tolist()
                root.attrs['state_std'] = (states.std(axis = 0) + 1e-6).tolist()
                print(f"[DEBUG] State stats saved: mean shape = {states.mean(axis = 0).shape}")
        
            if 'action' in root:
                actions = root['action'][:]
                root.attrs['action_mean'] = actions.mean(axis = 0).tolist()
                root.attrs['action_std'] = (actions.std(axis = 0) + 1e-6).tolist()
                print(f"[DEBUG] Action stats saved: mean shape = {actions.mean(axis = 0).shape}")
            
            metadata = {
                "description": "SO-ARM100 leader-follower low-dimensional dataset for Diffusion Policy",
                "control_method": "leader_follower_teleop",
                "recording_frequency_hz": round(1.0 / self.record_interval, 2),
                "action_dim": self.ACTION_DIM,
                "state_dim": self.STATE_DIM,
                "state_components": "6q1..q5, g, box(xyz,yaw), goal(xyz,yaw)",
                "action_components": "Δq1..Δq5 (arm), Δg (gripper)",
                "action_type": "delta_position",
                "created": timestamp,
                "total_frames": len(self.obs_state),
                "total_episodes": len(self.episode_ends),
                "robot": "SO-ARM100",
                "task": "pick_and_place",
                "leader_connected": self.leader_connected,
                "dataset_format": "diffusion_policy_lowdim_v1",
                "has_normalization_stats": True,
                "has_images": False
            }
            
            root.attrs.update(metadata)
        except Exception as e:
            print(f"Error saving zarr: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # **************** calculate file size ****************
        def get_directory_size(path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        try:
                            total_size += os.path.getsize(fp)
                        except OSError:
                            pass
            return total_size

        if os.path.exists(zarr_path):
            total_bytes = get_directory_size(zarr_path)
            size_mb = total_bytes / (1024 * 1024)
                    
            print(f"\n\n[SUCCESS] Low-dimensional Zarr dataset created: {zarr_path}")
            print(f"[INFO] File size: {size_mb:.2f} MB")
            print(f"[INFO] Episodes: {len(self.episode_ends)}")
            print(f"[INFO] Total frames: {len(self.obs_state)}")
            print(f"[INFO] Recording frequency: {1/self.record_interval} Hz")
            print(f"[INFO] State dimension: {self.STATE_DIM}")
            print(f"[INFO] Action dimension: {self.ACTION_DIM}")
            print(f"[INFO] Dataset saved to: {os.path.abspath(zarr_path)}")
                       
        return zarr_path

    # **************** discard current episode ****************
    def discard_current_episode(self):
        if not self.RECORDING:
            self.episode_discarded = True
            print("[DISCARD] Not recording - nothing to discard")
            return
        
        frames = len(self.obs_state) - self.current_episode_start_idx
        duration = self.getTime() - self.episode_start_time
        
        print(f"\n[DISCARD] Discarding current episode:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Frames: {frames}")
        print(f"  Reason: User requested discard")
        
        # roll back all buffers to the episode start
        start_idx = self.current_episode_start_idx
        
        # roll back main data buffers
        self.obs_state = self.obs_state[:start_idx]
        self.actions = self.actions[:start_idx]
        
        # roll back debug buffers
        self.box_positions = self.box_positions[:start_idx]
        self.box_orientations = self.box_orientations[:start_idx]
        self.goal_positions = self.goal_positions[:start_idx]
        
        # remove the episode start marker
        if self.episode_starts:
            self.episode_starts.pop()

        self.RECORDING = False
        self.EPISODE_COUNT -= 1
        
        print(f"[DISCARD] Episode discarded. Total episodes: {self.EPISODE_COUNT}")
        print(f"[DISCARD] Data rolled back to frame {start_idx}")
        self.randomize_environment()

    # ────────────────────────────────────────────────────────────────
    # Environment randomization
    # ────────────────────────────────────────────────────────────────
    def randomize_environment(self):
        # **************** randomize box and goal positions ****************
        if self.randomizer:
            try:
                box_pos, goal_pos, goal_rot, distance = self.randomizer.randomize_positions(
                    min_distance = self.MIN_DISTANCE_BETWEEN_OBJECTS,
                    max_attempts = 200
                )
                return True
            except Exception as e:
                print(f"[ERROR] Randomization failed: {e}")
                return False
        else:
            # **************** simple fallback randomization ****************
            try:
                if self.target_box:
                    box_x = random.uniform(-0.12, 0.12)
                    box_y = random.uniform(-0.15, 0.15)
                    box_z = 0.04
                    
                    self.target_box.getField('translation').setSFVec3f([box_x, box_y, box_z])
                    print(f"[RANDOM] Box position: {box_x:.3f}, {box_y:.3f}")
                
                if self.goal_zone:
                    goal_x = random.uniform(-0.12, 0.12)
                    goal_y = random.uniform(-0.15, 0.15)
                    goal_z = 0.08
                    
                    # **************** ensure box and goal are not too close ****************
                    if self.target_box:
                        dist = math.sqrt((box_x - goal_x)**2 + (box_y - goal_y)**2)
                        while dist < 0.08:
                            goal_x = random.uniform(-0.12, 0.12)
                            goal_y = random.uniform(-0.15, 0.15)
                            dist = math.sqrt((box_x - goal_x)**2 + (box_y - goal_y)**2)
                    
                    self.goal_zone.getField('translation').setSFVec3f([goal_x, goal_y, goal_z])
                    print(f"[RANDOM] Goal position: {goal_x:.3f}, {goal_y:.3f}")
                
                return True
            except Exception as e:
                print(f"[ERROR] Simple randomization failed: {e}")
                return False
    
    # ────────────────────────────────────────────────────────────────
    # Leader connection functions
    # ────────────────────────────────────────────────────────────────
    def connect_to_leader(self):
        for port in ['/dev/ttyACM0', '/dev/ttyUSB0']:
            try:
                self.ser = serial.Serial(port, 1000000, timeout = 0.02)
                time.sleep(2)
                self.leader_connected = True
                print(f"Connected to leader on {port}")
                return
            except Exception:
                pass
        print("Leader not connected")
    
    def read_leader_positions_sync(self):
        if not self.leader_connected:
            return self.last_raw_positions
        
        positions = []
        for servo_id in range(1, 7):
            try:
                cmd = bytes([0xFF, 0xFF, servo_id, 0x04, 0x02, 0x38, 0x02])
                checksum = (~sum(cmd[2:])) & 0xFF
                self.ser.write(cmd + bytes([checksum]))
                resp = self.ser.read(8)
                if len(resp) >= 8:
                    raw = (resp[6] << 8) | resp[5]
                    positions.append(self.raw_to_webots(raw, servo_id - 1))
                else:
                    positions.append(self.last_raw_positions[servo_id - 1])
            except Exception as e:
                print(f"[SERIAL READ ERROR] {e}")
                positions.append(self.last_raw_positions[servo_id - 1])
        
        self.last_raw_positions = positions
        return positions
    
    def read_leader_positions_thread(self):
        read_interval = 0.01
        last_read = time.time()
        
        while self.serial_running and self.ser and self.leader_connected:
            current_time = time.time()
            if current_time - last_read >= read_interval:
                try:
                    positions = []
                    for servo_id in range(1, 7):
                        cmd = bytes([0xFF, 0xFF, servo_id, 0x04, 0x02, 0x38, 0x02])
                        checksum = (~sum(cmd[2:])) & 0xFF
                        self.ser.write(cmd + bytes([checksum]))
                        resp = self.ser.read(8)
                        if len(resp) >= 8:
                            raw = (resp[6] << 8) | resp[5]
                            positions.append(self.raw_to_webots(raw, servo_id - 1))
                        else:
                            positions.append(self.last_raw_positions[servo_id - 1])
                    
                    with self.serial_lock:
                        self.last_raw_positions = positions
                        self.leader_positions = positions
                    
                    last_read = current_time
                except Exception as e:
                    print(f"[SERIAL ERROR] {e}")
                    time.sleep(0.01)
            else:
                time.sleep(0.001)
    
    def start_serial_thread(self):
        self.serial_running = True
        self.serial_thread = threading.Thread(target = self.read_leader_positions_thread)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        print("[INFO] Started serial reading thread")
    
    def stop_serial_thread(self):
        self.serial_running = False
        if self.serial_thread:
            self.serial_thread.join(timeout = 1.0)
        print("[INFO] Stopped serial reading thread")
    
    def raw_to_webots(self, raw, joint):
        cfg = self.CALIB[joint]
        
        if cfg["type"] == "gripper":
            raw_clamped = max(cfg["raw_closed"], min(cfg["raw_open"], raw))
            t = (raw_clamped - cfg["raw_closed"]) / (cfg["raw_open"] - cfg["raw_closed"])
            webots_pos = cfg["web_closed"] + t * (cfg["web_open"] - cfg["web_closed"])
        
        elif cfg["type"] == "centered":
            raw_clamped = max(cfg["raw_min"], min(cfg["raw_max"], raw))
            if raw_clamped >= cfg["raw_center"]:
                t = (raw_clamped - cfg["raw_center"]) / (cfg["raw_max"] - cfg["raw_center"])
                webots_pos = cfg["web_center"] + t * (cfg["web_max"] - cfg["web_center"])
            else:
                t = (cfg["raw_center"] - raw_clamped) / (cfg["raw_center"] - cfg["raw_min"])
                webots_pos = cfg["web_center"] - t * (cfg["web_center"] - cfg["web_min"])
        
        else:  # linear
            raw_clamped = max(cfg["raw_min"], min(cfg["raw_max"], raw))
            t = (raw_clamped - cfg["raw_min"]) / (cfg["raw_max"] - cfg["raw_min"])
            webots_pos = cfg["web_min"] + t * (cfg["web_max"] - cfg["web_min"])
        
        # **************** reverse direction for specific joints ****************
        if joint == 0:  # Joint 1 (base)
            webots_pos = -webots_pos
        
        joint_name = self.JOINT_NAMES[joint]
        webots_pos = self.clamp_to_joint_limits(joint_name, webots_pos)
        return webots_pos
    
    def smooth_positions(self, positions):
        self.position_buffer.append(positions)
        if len(self.position_buffer) == 0:
            return positions
        
        smoothed = [0.0] * len(positions)
        for i in range(len(positions)):
            total = 0.0
            for buffered in self.position_buffer:
                total += buffered[i]
            smoothed[i] = total / len(self.position_buffer)
        return smoothed
    
    # calculate 6D action: 5 arm joint deltas + 1 gripper delta
    def calculate_action(self, prev_positions, current_positions):
        action = np.zeros(self.ACTION_DIM, dtype = np.float32)
        # for i in range(min(len(current_positions), 5)):
        for i in range(5):
            action[i] = current_positions[i] - prev_positions[i]
        
        # gripper joint
        action[5] = current_positions[5] - prev_positions[5]
        # self.last_gripper_pos = current_gripper
        
        return action

    # ────────────────────────────────────────────────────────────────
    # Main loop
    # ────────────────────────────────────────────────────────────────
    def run(self):
        print("Starting leader-follower control loop with low-dim data collection")
        
        prev_positions = self.leader_positions[:] if self.leader_connected else [0.0] * 6
        
        while self.step(self.timestep) != -1:
            # **************** read keyboard ****************
            key = self.keyboard.getKey()
            while key != -1:
                if key == ord('R'):
                    if not self.RECORDING:
                        self.start_recording()
                elif key == ord('S'):
                    if self.RECORDING:
                        self.stop_and_save_episode()
                        self.randomize_environment()
                elif key == ord('D'):
                    self.discard_current_episode()
                elif key == ord('F'):
                    self.save_data()
                elif key == 27:  # ESC
                    if self.RECORDING:
                        self.stop_and_save_episode()
                    self.save_data()
                    self.stop_serial_thread()
                    self.simulationQuit(0)
                key = self.keyboard.getKey()
            
            now = self.getTime()

            # **************** update leader positions ****************
            if self.leader_connected and (now - self.last_leader_update >= self.leader_update_interval):
                with self.serial_lock:
                    current_positions = self.leader_positions[:]
                
                smoothed_positions = self.smooth_positions(current_positions)
                
                for i, (name, motor) in enumerate(self.motors.items()):
                    if i < len(smoothed_positions):
                        clamped_pos = self.clamp_to_joint_limits(name, smoothed_positions[i])
                        motor.setPosition(clamped_pos)
                        self.target_positions[i] = clamped_pos
                
                # self.gripper_pos = self.safe_get_value(self.gripper_sensor)
                
                action = self.calculate_action(prev_positions, smoothed_positions)

                # buffer action for next observation (causal)
                self.pending_action = action.copy()
                self.has_pending_action = True

                prev_positions = smoothed_positions[:]
                
                self.last_leader_update = now

            # **************** task success handling ****************
            if self.RECORDING:
                success = self.task_success()

                if success and self.success_time is not None:
                    if now - self.success_time >= self.POST_SUCCESS_TIME:
                        self.stop_and_save_episode()
                        self.randomize_environment()

            # **************** max episode duration ****************
            if self.RECORDING:
                if now - self.episode_start_time >= self.MAX_EPISODE_TIME:
                    print("[AUTO-STOP] Max episode duration reached")
                    self.discard_current_episode()

            # **************** max episode frame count ****************
            if self.RECORDING:
                frames = len(self.obs_state) - self.current_episode_start_idx
                if frames >= self.MAX_EPISODE_FRAMES:
                    print("[AUTO-STOP] Max episode frame count reached")
                    self.discard_current_episode()
    
            # **************** fixed-rate recording ****************
            if self.RECORDING and (now - self.last_record_time >= self.record_interval):
                self.record_step()
                self.last_record_time = now
        
        self.stop_serial_thread()
        
        if len(self.obs_state) > 0:
            self.save_data()
    
    # ────────────────────────────────────────────────────────────────
    # Task success detection
    # ────────────────────────────────────────────────────────────────
    def task_success(self):
        if self.episode_start_time is None:
            return False

        if self.getTime() - self.episode_start_time < self.MIN_EPISODE_TIME:
            return False

        try:
            box_pose = self.get_box_pose_simple()
            goal_pose = self.get_goal_pose_simple()
            
            dist_xy = math.sqrt((box_pose[0] - goal_pose[0])**2 + 
                            (box_pose[1] - goal_pose[1])**2)
            
            box_on_table = box_pose[2] < 0.1
            
            success = dist_xy < 0.04 and box_on_table
            
            if success and self.RECORDING and self.success_time is None:
                self.success_time = self.getTime()
                print(f"[SUCCESS] Task completed! Distance: {dist_xy:.3f}m")
    
            return success
        except Exception as e:
            print(f"[ERROR] task_success check failed: {e}")
            return False

# ********************************************************************
if __name__ == "__main__":
    controller = LeaderFollowerController()
    controller.run()