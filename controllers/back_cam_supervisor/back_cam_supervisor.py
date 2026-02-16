from controller import Supervisor
import time

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# get camera device
camera = robot.getDevice("back_cam")
camera.enable(timestep)

print(f"Back camera enabled: {camera.getWidth()}x{camera.getHeight()}")

# keep the cam active
while robot.step(timestep) != -1:
    # camera is now accessible from overlay
    pass