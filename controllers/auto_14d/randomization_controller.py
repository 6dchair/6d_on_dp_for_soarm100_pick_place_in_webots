from controller import Supervisor
import random
import math

class AutoDetectRandomizer:
    def __init__(self, robot, mode = "training"):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.mode = mode  # "training" or "inference" for box orientation
        
        # get nodes
        self.target_box = robot.getFromDef('TARGET_BOX')
        self.goal_zone = robot.getFromDef('GOAL_ZONE')
        self.work_table = robot.getFromDef('WORK_TABLE')

        # get workspace visualization from world
        try:
            self.workspace_viz = robot.getFromDef('WORKSPACE_VISUALIZATION')
        except Exception as e:
            self.workspace_viz = None
            print(f"[!] WORKSPACE_VISUALIZATION not found: {e}")
        
        # square box angle strategy
        self.setup_angle_strategy()
        
        # auto-detect all values
        self.detect_values()

    # **************** setup angle generation strategy for square box ****************
    def setup_angle_strategy(self):
        print(f"[Mode] Running in {self.mode.upper()} mode")
        
        # for SQUARE box (width == depth): 0-90° covers all unique orientations
        # training: Cover 0-90° systematically
        self.training_angles_deg = [
            0,    # aligned with X
            15,   # 
            30,   #
            45,   # diagonal
            60,   #
            75,   #
            90    # aligned with Y
        ]
        
        # inference: fill gaps in 0-90° range
        self.inference_angles_deg = [
            5, 10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 85
        ]
        
        # convert to radians for internal use
        self.training_angles_rad = [a * math.pi / 180 for a in self.training_angles_deg]
        self.inference_angles_rad = [a * math.pi / 180 for a in self.inference_angles_deg]

    # **************** auto-detect all parameters from the world ****************
    def detect_values(self):
        # table detection
        if self.work_table:
            self.table_translation = self.work_table.getField('translation').getSFVec3f()
            self.table_rotation = self.work_table.getField('rotation').getSFRotation()
            size_field = self.work_table.getField('size')
            self.table_size = size_field.getSFVec3f() if size_field else [0.4, 0.1, 0.7]
        else:
            self.table_translation = [0.1, 0, 3e-06]
            self.table_rotation = [1, 0, 0, 1.5708]
            self.table_size = [0.25, 0.05, 0.45]
        
        # box detection
        if self.target_box:
            size_field = self.target_box.getField('size')
            self.box_size = size_field.getSFVec3f() if size_field else [0.04, 0.04, 0.04]
            current_box_pos = self.target_box.getField('translation').getSFVec3f()
            self.box_center_z = current_box_pos[2]  # store as box_center_z

        else:
            self.box_size = [0.04, 0.04, 0.04]
            self.box_initial_z = 0.0699445

        # goal detection
        if self.goal_zone:
            current_goal_pos = self.goal_zone.getField('translation').getSFVec3f()
            self.goal_center_z = current_goal_pos[2]  # store as goal_center_z

            # try to get size from boundingObject or geometry
            bounding_field = self.goal_zone.getField('boundingObject')
            if bounding_field and bounding_field.getSFNode():
                bbox = bounding_field.getSFNode()
                size_field = bbox.getField('size')
                self.goal_size = size_field.getSFVec3f() if size_field else [0.08, 0.08, 0.001]
            else:
                self.goal_size = [0.1, 0.1, 0.001]
        else:
            self.goal_size = [0.08, 0.08, 0.001]
            self.goal_initial_z = 0.0500635 #if table height is 0.1: 0.095

        # calculate derived values
        self.calculate_derived_values()

        # update workspace visualization
        self.update_workspace_visualization()

    # **************** calculate derived values based on detected parameters ****************
    def calculate_derived_values(self):
        # table dimensions after rotation
        rx, ry, rz, angle = self.table_rotation
        is_rotated_90x = (abs(rx - 1.0) < 0.001 and abs(ry) < 0.001 and 
                          abs(rz) < 0.001 and abs(angle - 1.5708) < 0.001)
        
        if is_rotated_90x:
            # after 90 degrees  X rotation: X remains X, Y becomes Z, Z becomes Y
            self.table_width = self.table_size[0]      # X
            self.table_thickness = self.table_size[1]  # Y -> Z
            self.table_depth = self.table_size[2]      # Z -> Y
        else:
            # no rotation or different rotation
            self.table_width = self.table_size[0]      # X
            self.table_thickness = self.table_size[2]  # Z (thickness)
            self.table_depth = self.table_size[1]      # Y (depth)
        
        # table center and surface
        self.table_center_x = self.table_translation[0]
        self.table_center_y = self.table_translation[1]
        self.table_center_z = self.table_translation[2]
        self.table_surface_z = self.table_center_z + (self.table_thickness / 2)

        # object dimensions
        self.box_width = self.box_size[0]
        self.box_height = self.box_size[1]
        self.box_depth = self.box_size[2]

        # check if box is square (width == depth within tolerance)
        if max(self.box_width, self.box_depth) > 0:
            relative_diff = abs(self.box_width - self.box_depth) / max(self.box_width, self.box_depth)
            self.is_box_square = relative_diff < 0.01  # 1% tolerance
        else:
            self.is_box_square = True
        
        self.goal_width = self.goal_size[0]
        self.goal_depth = self.goal_size[1]
        self.goal_height = self.goal_size[2]

        # **************** CUSTOM REGION ****************
        # define offsets from TABLE CENTER (not edges)
        offset_from_center_x = 0.0  # keep centered in X; training: 0.0, max: 0.02 (for training)
        offset_from_center_y = 0.0  # keep centered in Y; training: 0.0, max: 0.02
        custom_width = 0.12 #inference: 0.12, training: 0.10
        custom_height = 0.30 #inference: 0.30, training: 0.25

        # calculate bounds relative to table center
        self.x_min = self.table_center_x - custom_width/2 + offset_from_center_x
        self.x_max = self.table_center_x + custom_width/2 + offset_from_center_x
        self.y_min = self.table_center_y - custom_height/2 + offset_from_center_y
        self.y_max = self.table_center_y + custom_height/2 + offset_from_center_y

    # **************** update workspace visualization based on calculated bounds ****************
    def update_workspace_visualization(self):
        if self.workspace_viz:
            try:
                # calculate center and dimensions
                center_x = (self.x_min + self.x_max) / 2
                center_y = (self.y_min + self.y_max) / 2
                width = self.x_max - self.x_min
                height = self.y_max - self.y_min

                current_pos = self.workspace_viz.getField('translation').getSFVec3f()
                if abs(current_pos[0] - center_x) > 0.01 or abs(current_pos[1] - center_y) > 0.01:
                    # position needs updating (first time or bounds changed)
                    viz_pos = [center_x, center_y, 0.051]  # fixed height
                    self.workspace_viz.getField('translation').setSFVec3f(viz_pos)
                    print(f"[Visualization] Initialized position: {viz_pos}")
                
                # get the Shape node and update IndexedLineSet coordinates
                shape_node = self.workspace_viz.getField('children').getMFNode(0)
                indexed_line_set = shape_node.getField('geometry').getSFNode()
                coord_node = indexed_line_set.getField('coord').getSFNode()
                
                 # update the coordinate points for the red outline
                points = [
                    [-width/2, -height/2, 0],
                    [width/2, -height/2, 0],
                    [width/2, height/2, 0],
                    [-width/2, height/2, 0]
                ]
                    
                point_field = coord_node.getField('point')
                point_field.setMFVec3f(0, points[0])
                point_field.setMFVec3f(1, points[1])
                point_field.setMFVec3f(2, points[2])
                point_field.setMFVec3f(3, points[3])
                    
            except Exception as e:
                print(f"[!] Could not update workspace visualization: {e}")
                import traceback
                traceback.print_exc()
                
    # **************** generate and set random positions ****************
    def randomize_positions(self, min_distance = 0.15, max_attempts = 200):
        box_x = box_y = goal_x = goal_y = 0
        distance = 0
        
        for attempt in range(max_attempts):
            box_x = random.uniform(self.x_min, self.x_max)
            box_y = random.uniform(self.y_min, self.y_max)
            goal_x = random.uniform(self.x_min, self.x_max)
            goal_y = random.uniform(self.y_min, self.y_max)
            
            distance = math.sqrt((goal_x - box_x) ** 2 + (goal_y - box_y) ** 2)
            
            if distance >= min_distance:
                break

        random_angle = 0
        base_angle_deg = 0
        
        # **************** SYSTEMATIC ANGLE GENERATION FOR SQUARE BOX ****************
        # for square box: use systematic angle strategy
        if self.is_box_square:
            if self.mode == "training":
                # select from training angles (0-90° range)
                base_angle_rad = random.choice(self.training_angles_rad)
                angle_set = self.training_angles_deg
                mode_str = "TRAINING"
            else:
                # select from inference angles (fill gaps in 0-90°)
                base_angle_rad = random.choice(self.inference_angles_rad)
                angle_set = self.inference_angles_deg
                mode_str = "INFERENCE"
            
            base_angle_deg = base_angle_rad * 180 / math.pi
            
            # for visual variety in simulation, map to random quadrant
            # (all quadrants are equivalent for square box due to symmetry)
            quadrants = [0, math.pi/2, math.pi, 3 * math.pi/2]
            quadrant = random.choice(quadrants)
            random_angle = (base_angle_rad + quadrant) % (2 * math.pi)

            # calculate final angle in degrees
            # final_angle_deg = random_angle * 180 / math.pi
            
            print(f"[Box angle] Square box: {base_angle_deg:.1f}°")
            # print(f"[Box angle] Square box: {base_angle_deg:.1f}° (unique), {random_angle*180/math.pi:.1f}° (visual)")

        
        else:
            # non-square box: use full random (0-360°)
            random_angle = random.uniform(0, 2 * math.pi)
            base_angle_deg = random_angle * 180 / math.pi
            # final_angle_deg = random_angle * 180 / math.pi
            print(f"[Box angle] Non-square box: {base_angle_deg:.1f}°")
        
        # set positions using calculated Z values
        box_pos = [box_x, box_y, self.box_center_z]
        goal_pos = [goal_x, goal_y, self.goal_center_z]
        
        # set box rotation - only around Z axis [0, 0, 1]
        box_rotation = [0, 0, 1, random_angle]
        
        # apply positions and rotations
        self.target_box.getField('translation').setSFVec3f(box_pos)
        self.target_box.getField('rotation').setSFRotation(box_rotation)
        self.goal_zone.getField('translation').setSFVec3f(goal_pos)

        # reset physics
        self.target_box.resetPhysics()
        self.goal_zone.resetPhysics()
        
        self.update_workspace_visualization()

        return box_pos, box_rotation, goal_pos, distance

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    # create auto-detecting randomizer with specified mode
    # default to "training" if not specified
    mode = "training"  # change to "inference" for novel angles
    randomizer = AutoDetectRandomizer(robot, mode = mode)

    box_pos, box_rot, goal_pos, distance = randomizer.randomize_positions()

    # **************** main loop ****************
    while robot.step(timestep) != -1:
        pass

# ********************************************************************
if __name__ == "__main__":
    main()