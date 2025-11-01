#!/usr/bin/env python3
"""
Update MuJoCo XML with new joint reference positions.
This is the simplest approach - just update the 'ref' attribute of joints.
"""

import sys
import os

# Import mujoco first, before any functions that might shadow it
import mujoco as mj
import numpy as np
import xml.etree.ElementTree as ET


def update_joint_refs_in_xml(xml_path, joint_angles, output_path):
    """
    Update joint reference positions in the XML file.
    This sets the default/home position for each joint.
    """
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    print("\nUpdating joint reference positions in XML:")
    updated_count = 0
    
    # Find and update joint elements
    for joint in root.iter('joint'):
        joint_name = joint.get('name')
        if joint_name and joint_name in joint_angles:
            old_ref = joint.get('ref', '0.0')
            new_ref = str(joint_angles[joint_name])
            joint.set('ref', new_ref)
            print(f"  {joint_name}: {old_ref} -> {new_ref} ({np.degrees(float(new_ref)):.1f}°)")
            updated_count += 1
    
    # Format XML nicely
    def indent_xml(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    indent_xml(root)
    
    # Write the updated XML
    tree.write(output_path, encoding='unicode', xml_declaration=True)
    
    print(f"\nUpdated {updated_count} joints")
    print(f"Saved to: {output_path}")
    
    # List joints that were in the dict but not found
    all_joint_names = set()
    for joint in root.iter('joint'):
        name = joint.get('name')
        if name:
            all_joint_names.add(name)
    
    not_found = set(joint_angles.keys()) - all_joint_names
    if not_found:
        print(f"\nWarning: These joints were not found in XML: {not_found}")
    
    return output_path


def verify_with_mujoco(xml_path, joint_angles):
    """
    Load the model in MuJoCo and verify the joint positions.
    """
    print("\n" + "="*50)
    print("Verifying with MuJoCo simulation:")
    print("="*50)
    
    # Load model
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    
    # Reset to default positions (uses ref values)
    mj.mj_resetData(model, data)
    
    # The model should now be in the pose defined by ref values
    # Let's check the actual joint positions
    print("\nJoint positions after reset (should match ref values):")
    for joint_name in joint_angles.keys():
        try:
            joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_addr = model.jnt_qposadr[joint_id]
                actual_angle = data.qpos[qpos_addr]
                expected_angle = joint_angles[joint_name]
                diff = actual_angle - expected_angle
                status = "✓" if abs(diff) < 0.001 else "✗"
                print(f"  {joint_name}: {actual_angle:.4f} (expected: {expected_angle:.4f}, diff: {diff:.6f}) {status}")
        except:
            print(f"  {joint_name}: NOT FOUND")
    
    # Run forward kinematics
    mj.mj_forward(model, data)
    
    # Print some body positions
    print("\nKey body positions:")
    important_bodies = ["base", "Torso", "foot_right", "foot_left", "right_hand", "left_hand"]
    for body_name in important_bodies:
        try:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                pos = data.xpos[body_id]
                print(f"  {body_name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        except:
            pass
    
    # Check if the pose looks reasonable
    try:
        base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base")
        foot_right_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "foot_right")
        foot_left_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "foot_left")
        
        if base_id >= 0 and foot_right_id >= 0 and foot_left_id >= 0:
            base_height = data.xpos[base_id][2]
            right_foot_height = data.xpos[foot_right_id][2]
            left_foot_height = data.xpos[foot_left_id][2]
            
            print(f"\nHeight check:")
            print(f"  Base height: {base_height:.3f} m")
            print(f"  Right foot height: {right_foot_height:.3f} m")
            print(f"  Left foot height: {left_foot_height:.3f} m")
            print(f"  Base above right foot: {base_height - right_foot_height:.3f} m")
            print(f"  Base above left foot: {base_height - left_foot_height:.3f} m")
            
            if right_foot_height > base_height or left_foot_height > base_height:
                print("\n⚠ WARNING: Feet are above the base! The pose may be unrealistic.")
                print("  Consider checking your joint angles, especially knee angles.")
    except:
        pass
    
    return model, data


def main():
    # Define desired joint angles (in radians)
    joint_angles = {
        "right_shoulder_pitch": 0,
        "right_shoulder_yaw": 0,
        "right_elbow_yaw": 0,
        "left_shoulder_pitch": 0,
        "left_shoulder_yaw": 0,
        "left_elbow_yaw": 0,
        "right_hip_pitch": -0.0471,
        "right_hip_yaw": -0.0951,
        "right_hip_roll": -0.785,
        "right_knee_pitch": -2.98,  # This is very extreme! -171 degrees
        "right_ankle_pitch": 0.298,
        "left_hip_pitch": 0.0943,
        "left_hip_yaw": -0.0951,
        "left_hip_roll": 0.785,
        "left_knee_pitch": 0.0943,
        "left_ankle_pitch": 0.0157
    }
    
    # Get input file path
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        # Try to find the file
        possible_paths = [
            "robot.mjcf",
            "z001.xml",
            "./robot_model.xml",
            "./z001.xml"
        ]
        xml_path = None
        for path in possible_paths:
            if os.path.exists(path):
                xml_path = path
                break
        
        if not xml_path:
            print("Error: No MuJoCo XML file found!")
            print("Usage: python update_quaternions_working.py [path_to_xml_file]")
            sys.exit(1)
    
    if not os.path.exists(xml_path):
        print(f"Error: File '{xml_path}' not found!")
        sys.exit(1)
    
    # Output file path
    base_name = os.path.splitext(xml_path)[0]
    output_path = f"{base_name}_updated.xml"
    
    print(f"Input file: {xml_path}")
    print(f"Output file: {output_path}")
    
    # Update the XML
    output_file = update_joint_refs_in_xml(xml_path, joint_angles, output_path)
    
    # Verify with MuJoCo
    try:
        model, data = verify_with_mujoco(output_file, joint_angles)
        
        # Ask if user wants to visualize
        print("\n" + "="*50)
        print("Would you like to visualize the result? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            print("Launching viewer...")
            print("Controls: Space=pause, Tab=UI, Mouse=camera")
            mj.viewer.launch(model, data)
            
    except Exception as e:
        print(f"\nError during MuJoCo verification: {e}")
        print("The XML file has been updated, but verification failed.")
    
    print(f"\n✓ Done! Updated file saved as: {output_path}")


if __name__ == "__main__":
    main()
