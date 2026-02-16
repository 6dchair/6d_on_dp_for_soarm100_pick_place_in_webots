# With Supervisor for Overhead Camera
from controller import Supervisor, Keyboard
import math

# Simulation constants
TIME_STEP = 32
ANGLE_STEP = 0.05
GRIPPER_STEP = 0.05

KEY_MAP = {
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

JOINT_NAMES = ["1", "2", "3", "4", "5", "6"]
GRIPPER_NAME = "6"

# ------------------ Helpers ------------------
def safe_get_value(sensor):
    val = sensor.getValue()
    return 0.0 if math.isnan(val) else val

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# ------------------ Initialization ------------------
robot = Supervisor() 
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

motors, sensors = {}, {}

for name in JOINT_NAMES:
    motor = robot.getDevice(name)
    sensor = robot.getDevice(f"{name}_sensor")
    sensor.enable(TIME_STEP)

    motors[name] = motor
    sensors[name] = sensor

# Wait one step so Webots updates joint sensors ...
for _ in range(2):
    robot.step(TIME_STEP)

# After sensors are valid, sync motors to actual joint angles
for name in JOINT_NAMES:
    current = safe_get_value(sensors[name])
    motors[name].setPosition(current)
    motors[name].setVelocity(0.5)   # soft velocity limit (no jerks)

# ------------------ Cameras ------------------
cameras = {}
try:
    # ------------------ Gripper camera ------------------
    gripper_camera = robot.getDevice("gripper_camera")
    gripper_camera.enable(TIME_STEP)
    cameras["gripper"] = gripper_camera
    print("[INFO] Gripper camera enabled.")
except Exception as e:
    cameras["gripper"] = None
    print(f"[WARN] No gripper camera found: {e}")

try:
    # ------------------ Overhead cam (fixed on teh world) ------------------
    overhead_camera = robot.getDevice("overhead_camera")
    overhead_camera.enable(TIME_STEP)
    cameras["overhead"] = overhead_camera
    print("[INFO] Overhead camera enabled.")
    print(f"[INFO] Camera resolution: {overhead_camera.getWidth()}x{overhead_camera.getHeight()}")
except Exception as e:
    cameras["overhead"] = None
    print(f"[WARN] No overhead camera found: {e}")

#  ------------------ Debug: list all devices ------------------
print("\nDebug: All available devices:\n")
for i in range(robot.getNumberOfDevices()):
    device = robot.getDeviceByIndex(i)
    print(f"  Device {i}: {device.getName()}")

# Sync gripper variable with actual sim state
gripper_pos = safe_get_value(sensors[GRIPPER_NAME])

# ------------------ Motion control ------------------
def move_joint(joint_name, direction):
    current_pos = safe_get_value(sensors[joint_name])
    min_pos = motors[joint_name].getMinPosition()
    max_pos = motors[joint_name].getMaxPosition()

    if math.isnan(min_pos) or min_pos <= -1e9:
        min_pos = -3.14
    if math.isnan(max_pos) or max_pos >= 1e9:
        max_pos = 3.14

    new_pos = current_pos + direction * ANGLE_STEP
    new_pos = clamp(new_pos, min_pos, max_pos)
    motors[joint_name].setPosition(new_pos)

def move_gripper(action):
    global gripper_pos
    if action == "open":
        gripper_pos = clamp(gripper_pos + GRIPPER_STEP, 0.0, 2.0)
    elif action == "close":
        gripper_pos = clamp(gripper_pos - GRIPPER_STEP, 0.0, 2.0)
    motors[GRIPPER_NAME].setPosition(gripper_pos)

# ------------------ Instructions ------------------
print("\n=== SO-ARM100 KEYBOARD CONTROL ===")
print(" Joint 1 (Yaw) ---------------- T / G")
print(" Joint 2 (Pitch) -------------- Y / H")
print(" Joint 3 (Pitch) -------------- U / J")
print(" Joint 4 (Pitch) -------------- I / K")
print(" Joint 5 (Yaw) ---------------- O / L")
print(" Gripper ---------------------- , (Open), . (Close)")

# ------------------ Main control loop ------------------
while robot.step(TIME_STEP) != -1:
    key = keyboard.getKey()
    while key != -1:
        cmd = KEY_MAP.get(key)
        if cmd:
            if isinstance(cmd, str):
                move_gripper(cmd)
            else:
                joint_name, direction = cmd
                move_joint(joint_name, direction)
        key = keyboard.getKey()

    # Test camera images occasionally
    if robot.getTime() % 5.0 < TIME_STEP/1000.0:  # Every ~5 seconds
        if cameras.get("overhead"):
            try:
                img = cameras["overhead"].getImage()
                print(f"[DEBUG] Overhead camera image available at t={robot.getTime():.1f}s")
            except:
                pass
