from typing import Dict

# --- robomimic imports ---
import robomimic.envs.env_base as EB

# --- gello and robot imports ---
import sys
sys.path.insert(0, "/immimic/gello_software")
from gello.rl2_env import RobotEnv
from gello.robots.panda_deoxys_simple import PandaRobot
from gello.cameras.zed_camera import ZedCamera
from gello.cameras.realsense_camera import RealSenseCamera

def create_cameras() -> dict:
    cam_dict = {}
    cam_config_dict = {
        "agentview": {"sn": "AGENTVIEW_CAM_SERIAL_NUMBER", "type": "RealSense", "width": 640, "height": 360},
        "wrist": {"sn": "WRIST_CAM_SERIAL_NUMBER", "type": "Zed", "width": 640, "height": 360}
    }

    if cam_config_dict is not None:
        for cam in cam_config_dict:
            cam_config = cam_config_dict[cam]
            if cam_config["type"] == "Zed":
                from gello.cameras.zed_camera import ZedCamera
                cam_dict[cam] = ZedCamera(cam, cam_config)
            elif cam_config["type"] == "RealSense":
                from  gello.cameras.realsense_camera import RealSenseCamera
                cam_dict[cam] = RealSenseCamera(device_id=cam_config['sn'])
    return cam_dict

class RL2RobotEnv(EB.EnvBase):
    def __init__(
        self, 
        env_name = None, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        use_depth_obs=False, 
        postprocess_visual_obs=True,
        # camera setup
        camera_config_dict: Dict = {},
        camera_cap_fps: float = 30.0,
        save_depth_obs = False,
        # robot control setup
        gripper_type: str = "robotiq",      # ['umi', 'robotiq', 'allegro', 'ability']
        controller_type: str = "OSC_POSE",
        control_rate_hz: float = 30.0      
    ):
        super().__init__(env_name, render, render_offscreen, use_image_obs, use_depth_obs, postprocess_visual_obs)

        self.robot_client = PandaRobot(controller_type=controller_type, gripper_type=gripper_type)
        self.camera_dict = create_cameras()
        self.controller = controller_type
        self.gripper = gripper_type

        self.env = RobotEnv(
            robot=self.robot_client,
            camera_dict=self.camera_dict,
            control_rate_hz=control_rate_hz,
            save_depth_obs=False,
        )

    def get_observation(self):
        return self.env.get_obs()

    def reset(self):
        self.robot_client.reset()

    def step(self, action):
        obs, action = self.env.step(action)
        return obs, action

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        if self.gripper == "robotiq":
            return 8
        elif self.gripper == "umi":
            return 8
        elif self.gripper == "ability":
            return 13
        elif self.gripper == "allegro":
            return 23
        else:
            raise NotImplementedError("gripper type is not supported")

    @property
    def controller_type(self):
        return self.controller
    
    @property
    def version(self):
        """
        Returns version of environment (str).
        This is not an abstract method, some subclasses do not implement it
        """
        return None

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        # we don't wrap any env
        return self

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state

        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        raise NotImplementedError("reset_to is not implemented")

    def render(self, mode="human", height=None, width=None, camera_name=None):
        """Render"""
        raise NotImplementedError("render is not implemented")

    def get_state(self):
        """Get environment simulator state, compatible with @reset_to"""
        raise NotImplementedError("get_state is not implemented")

    def get_reward(self):
        """
        Get current reward.
        """
        raise NotImplementedError("get_reward is not implemented")

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError("get_goal is not implemented")

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError("set_goal is not implemented")

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        raise NotImplementedError("is_done is not implemented")

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        # real robot environments don't usually have a success check - this must be done manually
        raise NotImplementedError("is_success is not implemented")

    def name(self):
        """
        Returns name of environment name (str).
        """
        return "PandaDeoxysSimple"

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        from robomimic.envs.env_base import EnvType
        return EnvType.PANDA_DEOXYS_SIMPLE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        raise NotImplementedError("serialize is not implemented")
    
    def create_for_data_processing(
        cls,
        camera_names,
        camera_height,
        camera_width,
        reward_shaping,
        render=None,
        render_offscreen=None,
        use_image_obs=None,
        use_depth_obs=None,
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions.

        Args:
            camera_names ([str]): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
            render (bool or None): optionally override rendering behavior. Defaults to False.
            render_offscreen (bool or None): optionally override rendering behavior. The default value is True if
                @camera_names is non-empty, False otherwise.
            use_image_obs (bool or None): optionally override rendering behavior. The default value is True if
                @camera_names is non-empty, False otherwise.
            use_depth_obs (bool): if True, use depth observations

        Returns:
            env (EnvBase instance)
        """
        raise NotImplementedError("create_for_data_processing")

    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        raise NotImplementedError("rollout_exceptions is not implemented")