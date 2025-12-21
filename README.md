# README

## 1. Data Structure

```python
`-<task_name>/  # e.g. pick_moving_object
  `-all_variations/
    `-episodes/
      `-<episode_name>/ # e.g. episode0
        `-cam0_depth/
            `-0.png, 1.png, ...
        `-cam0_rgb/
            `-0.png, 1.png, ...
        `-cam1_depth/
        `-cam1_rgb/
        `-cam2_depth/
        `-cam2_rgb/
        `-cam3_depth/
        `-cam3_rgb/
        `-robot_state/
            `-0.npy, 1.npy, ...
        # left_gripper_end_effector: [x,y,z,rx,ry,rz,rw, openess] (np.ndarray, (8,), np.float32), 
            # robot base coordinate
            # openness: 0 = close/ 1 = open
        # right_gripper_end_effector
        `-camera_parameters/
            `-cam0.npy, cam1.npy, cam2.npy
        # intrinsics: (np.ndarray, (3,3), np.float32)
        # extrinsics: (np.ndarray, (4,4), np.float32) 
            # camera coordinate to robot base coordinate
        `-meta_data.json
        # task_name: str
        # robot_joint_states: Dict: all joint angles, in case we need to reset the robot
            # 0: all joint angles
            # 1: all joint angles
            # ...
```

## 2. Observations

- cam[1-4]: RGB-D camera [frequency:20Hz] [resolution:480x270]
- [left/right]_gripper_end_effector: end-effector-pose [20Hz:synchronized to the camera]
- Tips: Data collection can be performed at 80 Hz, then downsampled to 20 Hz during synchronization. We require synchronization accuracy < 8 ms

## 3. Scene Requirements

- Uniform lighting and no clutter within the field of view of all four cameras
- Each episode should be approximately 15-20 seconds long

