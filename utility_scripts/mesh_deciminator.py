# reduces polygon count of STL files for faster simulation
# uses PyMeshLab with meshing_decimation_quadric_edge_collapse filter

import pymeshlab as ml
from pathlib import Path

def decimate_mesh(input_file, output_file, target_reduction = 0.3):
    try:
        # create a new mesh set
        ms = ml.MeshSet()
        
        # load the mesh
        ms.load_new_mesh(input_file)
        mesh = ms.current_mesh()
        
        original_faces = mesh.face_number()
        target_faces = int(original_faces * target_reduction)
        
        print(f"Processing: {input_file}")
        print(f"Original triangles: {original_faces}")
        print(f"Target triangles: {target_faces}")
        
        # apply quadric edge collapse decimation using the filter that exists in this PyMeshLab version
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                       targetfacenum=target_faces)
        
        # get decimated mesh info
        decimated_mesh = ms.current_mesh()
        decimated_faces = decimated_mesh.face_number()
        
        # save the decimated mesh
        ms.save_current_mesh(output_file)
        
        reduction_ratio = (original_faces - decimated_faces) / original_faces * 100
        print(f"Decimated triangles: {decimated_faces}")
        print(f"Reduction: {reduction_ratio:.1f}%")
        print(f"Saved to: {output_file}\n")
        
        return True
    
    except Exception as e:
        print(f"[!] Error: {e}\n")
        return False

def main():
    # list of STL files to decimate
    stl_files = [
        "Base.stl",
        "Base_Motor.stl",
        "Rotation_Pitch.stl",
        "Rotation_Pitch_Motor.stl",
        "Upper_Arm.stl",
        "Upper_Arm_Motor.stl",
        "Lower_Arm.stl",
        "Lower_Arm_Motor.stl",
        "Wrist_Pitch_Roll.stl",
        "Wrist_Pitch_Roll_Motor.stl",
        "Fixed_Jaw.stl",
        "Fixed_Jaw_Motor.stl",
        "Moving_Jaw.stl",
    ]
    
    # Set paths
    assets_dir = Path("./assets/original")  # current directory
    output_dir = assets_dir / "simplified"
    
    # create output directory
    output_dir.mkdir(exist_ok = True)

    print("SO-ARM100 Mesh Decimation Tool (PyMeshLab)")
    print(f"Input directory: {assets_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Reduction target: 10% (keep 10% of original triangles)")
    print(f"Filter: meshing_decimation_quadric_edge_collapse\n")
    
    success_count = 0
    fail_count = 0
    
    # Process each STL file
    for stl_file in stl_files:
        input_path = assets_dir / stl_file
        output_path = output_dir / stl_file
        
        if not input_path.exists():
            print(f"[!] ]File not found: {input_path}\n")
            fail_count += 1
            continue
        
        if decimate_mesh(str(input_path), str(output_path), target_reduction = 0.01):
            success_count += 1
        else:
            fail_count += 1
    

if __name__ == "__main__":
    main()