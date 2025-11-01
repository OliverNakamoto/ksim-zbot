#!/usr/bin/env python3
"""
Simple script to update joint reference positions in MuJoCo XML.
This sets the 'ref' attribute of joints, which defines their default/home position.
"""

import xml.etree.ElementTree as ET
import numpy as np

def update_joint_refs(xml_path, joint_angles, output_path):
    """Update joint reference positions in XML file."""
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Update joint ref values
    updated_joints = []
    for joint in root.iter('joint'):
        joint_name = joint.get('name')
        if joint_name in joint_angles:
            old_ref = joint.get('ref', '0.0')
            new_ref = str(joint_angles[joint_name])
            joint.set('ref', new_ref)
            updated_joints.append(joint_name)
            print(f"Updated {joint_name}: ref={old_ref} -> {new_ref} ({np.degrees(float(new_ref)):.2f}°)")
    
    # Save the updated XML
    tree.write(output_path, encoding='unicode', xml_declaration=True)
    
    print(f"\nUpdated {len(updated_joints)} joint references")
    print(f"Saved to: {output_path}")
    
    # Check for joints that were in the dict but not found in XML
    not_found = set(joint_angles.keys()) - set(updated_joints)
    if not_found:
        print(f"\nWarning: These joints were not found in the XML: {not_found}")

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
        "right_knee_pitch": -2.98,
        "right_ankle_pitch": 0.298,
        "left_hip_pitch": 0.0943,
        "left_hip_yaw": -0.0951,
        "left_hip_roll": 0.785,
        "left_knee_pitch": 0.0943,
        "left_ankle_pitch": 0.0157
    }
    
    # File paths
    xml_path = "robot.mjcf"  # Input file
    output_path = "robot_model_with_refs.xml"  # Output file
    
    print("Updating joint reference positions...\n")
    update_joint_refs(xml_path, joint_angles, output_path)
    
    print("\nTo test the updated model:")
    print("python3 -c \"import mujoco, mujoco.viewer; m = mujoco.MjModel.from_xml_path('" + output_path + "'); d = mujoco.MjData(m); mujoco.mj_resetData(m, d); mujoco.viewer.launch(m, d)\"")

if __name__ == "__main__":
    main()
