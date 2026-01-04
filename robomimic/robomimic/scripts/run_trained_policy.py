from dataclasses import dataclass
from typing import Optional

import os
import tyro
import torch
import numpy as np
import imageio
import h5py
from tqdm import tqdm
import time
import pickle
import cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image
import h5py

# --- robomimic imports ---
from robomimic.envs.env_real_panda_gello import RL2RobotEnv
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

camera_config_dict = {
    "agentview": {"sn": "AGENTVIEW_CAM_SERIAL_NUMBER", "type": "RealSense", "width": 640, "height": 360},
    "wrist": {"sn": "WRIST_CAM_SERIAL_NUMBER", "type": "Zed", "width": 640, "height": 360}
}

@dataclass
class Args:
    """Arguments for rollout."""
    ckpt_path: str                           # path to Robomimic checkpoint .pth
    norm_path: str                           # path to normalization stats file .pkl
    horizon: int = 10000                     # number of steps in the rollout
    camera_config_dict = camera_config_dict  # path to or dict for camera config
    controller_type: str = "OSC_POSE"
    gripper_type: str = "robotiq"            # ['umi', 'robotiq', 'allegro', 'ability']
    control_rate_hz: float = 30.0             # control process rate, how fast we send to Deoxys
    seed: int = 1                            # seed for rollout
    n_rollouts: int = 1                      # number of rollouts
    temp_agg: bool = True                   # whether to do temporal ensembling
    pred_action_horizon: int = 32           # for each of the timestep predict pred_action_horizon
    inference_frequency: int = 31            # do the inference each interval

def action_quat_to_axis_angle(action):
    pos = action[0:3]
    rot = action[3:7]
    rot = quat_to_axis_angle(rot)
    gripper = action[7:len(action)]
    return np.concatenate((pos, rot, gripper), axis=0)

def action_axis_angle_to_quat(action):
    pos = action[0:3]
    rot = action[3:6]
    rot = axis_angle_to_quat(rot)
    gripper = action[6:len(action)]
    return np.concatenate((pos, rot, gripper), axis=0)

def quat_to_axis_angle(q):
    """
    Convert a quaternion [w, x, y, z] to an axis-angle (rotation vector).
    Parameters:
        q (numpy array): Quaternion in [w, x, y, z] format.
    Returns:
        numpy array: Axis-angle (rotation vector) representation.
    """
    # [w, x, y, z] -> [x, y, z, w]
    q_scipy = np.array([q[1], q[2], q[3], q[0]])
    rot = R.from_quat(q_scipy)
    return rot.as_rotvec()

def axis_angle_to_quat(axis_angle):
    """
    Convert an axis-angle (rotation vector) to a quaternion [w, x, y, z].
    Parameters:
        axis_angle (numpy array): A 3D vector representing axis-angle (rotation vector).
    Returns:
        numpy array: Quaternion in [w, x, y, z] format.
    """
    rot = R.from_rotvec(axis_angle)
    q_scipy = rot.as_quat()  # [x, y, z, w]
    return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]) # [w, x, y, z]

def obs_preprocess(obs, gripper_type):
    # change gripper state range from -1 to 1
    if gripper_type == "umi" or gripper_type == "robotiq":
        obs["eef_pose_w_gripper"][-1] = obs["eef_pose_w_gripper"][-1] / 0.441 * 2 - 1
    
    for k, v in obs.items():
        # downsize and normalize image observation
        if "image" in k:
            obs[k] = cv2.resize(obs[k], (320, 180)) / 255.0
    return obs

def rollout(policy, env, horizon, gripper_type, temp_agg=True, pred_action_horizon=1, inference_frequency=1):

    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)

    policy.start_episode()
    env.reset()

    action_preds = []  # List to store tuples: (inference_step, action_list)
    action_list = None

    for step_t in range(horizon):
        obs = env.get_observation()
        
        obs["eef_pose_w_gripper"] = np.concatenate((obs["eef_pose"], np.array([obs["gripper_position"]])), axis=0)
        
        for k, v in obs.items():
            obs[k] = v.copy()

        obs = obs_preprocess(obs=obs, gripper_type=gripper_type)

        new_action = policy(ob=obs)
        if step_t % inference_frequency == 0:
            action_list = new_action
            action_preds.append((step_t, action_list))

        if not temp_agg:
            offset = step_t % inference_frequency
            action = action_list[offset]
        else:
            alpha = 0.8
            collected_actions = []
            weights = []
            for (pred_step, pred_list) in action_preds:
                delta = step_t - pred_step
                if delta < (pred_action_horizon - 1):
                    collected_actions.append(pred_list[delta])
                    weights.append(alpha ** delta)
            if collected_actions:
                collected_actions = np.array(collected_actions)
                weights = np.array(weights)
                action = np.sum(collected_actions * weights[:, None], axis=0) / np.sum(weights)
            else:
                raise NotImplementedError

        # hard constraint to avoid collision
        action[0] = min(max(0.30, action[0]),0.75)
        action[1] = min(max(-0.2, action[1]),0.30)
        action[2] = min(max(0.2, action[2]),0.4)

        action = action_axis_angle_to_quat(action)
        print(f'action: {action}')
        action = action.tolist()        
        obs, action = env.step(action)

    stats, traj = None, None
    return stats, traj
        
def main():    
    env = RL2RobotEnv(
        camera_config_dict=camera_config_dict,
        controller_type=args.controller_type,
        gripper_type=args.gripper_type,
        control_rate_hz=args.control_rate_hz
    )

    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.ckpt_path)

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print("=======================================================================")
    print(f"Device being used : {device}")
    print("=======================================================================")

    # load policy from ckpt
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    # set normalization stats of policy
    if args.norm_path:
        with open(args.norm_path, "rb") as f:
            unnorm_stats = pickle.load(f)
        policy.action_normalization_stats = unnorm_stats
    # pixel-wise normalization stats from robomimic training is not used here
    policy.obs_normalization_stats = None

    # maybe set seed
    if args.seed is not None:
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    for i in tqdm(range(args.n_rollouts)):
        try:
            stats, traj = rollout(
                policy=policy,
                env=env,
                horizon=args.horizon,
                gripper_type=args.gripper_type,
                temp_agg=args.temp_agg,
                pred_action_horizon=args.pred_action_horizon,
                inference_frequency=args.inference_frequency
            )
        except KeyboardInterrupt:
            print("ctrl-C catched, stop execution")
            continue
        print("TERMINATING SUCCESSFULLY WITHOUT KEYBOARD INTERRUPT...")
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    main()