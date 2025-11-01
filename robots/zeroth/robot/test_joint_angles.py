#!/usr/bin/env python3
"""
Script to test and visualize robot with specified joint angles.
This shows how to control the robot at runtime without modifying the XML.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

def controller(model, data):
    """Controller callback called at each simulation step."""
    
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
    
    # If you want to use position control (assuming position actuators)
    # Set the control inputs to the desired positions
    for i, (joint_name, angle) in enumerate(joint_angles.items()):
        try:
            # Find actuator index for this joint
            actuator_name = joint_name + "_ctrl"
            actuator_id = model.actuator(actuator_name).id
            
            # For position control, set ctrl to desired angle
            # For torque control, you'd need a PD controller
            data.ctrl[actuator_id] = angle
            
        except KeyError:
            pass  # Actuator not found
    
    # Alternative: Directly set joint positions (bypasses dynamics)
    # Uncomment this if you want to force positions regardless of physics
    """
    for joint_name, angle in joint_angles.items():
        try:
            joint_id = model.joint(joint_name).id
            qpos_addr = model.joint(joint_name).qposadr
            data.qpos[qpos_addr] = angle
        except KeyError:
            pass
    """

def main():
    # Load model
    xml_path = "robot.mjcf"  # Update this path
    
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Define desired joint angles
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
    
    # Set initial joint positions
    print("\nSetting initial joint positions:")
    for joint_name, angle in joint_angles.items():
        try:
            joint_id = model.joint(joint_name).id
            qpos_addr = model.joint(joint_name).qposadr
            data.qpos[qpos_addr] = angle
            print(f"  {joint_name}: {angle:.4f} rad ({np.degrees(angle):.2f}°)")
        except KeyError:
            print(f"  Warning: Joint '{joint_name}' not found")
    
    # Update the model with forward kinematics
    mujoco.mj_forward(model, data)
    
    # Print resulting body positions and quaternions
    print("\nResulting body configurations:")
    important_bodies = ["base", "Torso", "foot_right", "foot_left", "right_hand", "left_hand"]
    for body_name in important_bodies:
        try:
            body_id = model.body(body_name).id
            pos = data.xpos[body_id]
            quat = data.xquat[body_id]  # wxyz format
            print(f"  {body_name}:")
            print(f"    pos: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
            print(f"    quat: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
        except KeyError:
            pass
    
    # Launch viewer
    print("\nLaunching viewer...")
    print("Controls:")
    print("  - Space: pause/unpause")
    print("  - Tab: show control panel")
    print("  - Right-click + drag: rotate camera")
    print("  - Scroll: zoom")
    print("  - Esc: quit")
    
    # Use the controller callback for continuous control
    mujoco.viewer.launch(model, data, controller)

if __name__ == "__main__":
    main()
