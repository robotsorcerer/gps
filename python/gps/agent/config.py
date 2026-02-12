"""
Default configuration and hyperparameters for agent objects.

ROS 2 migration notes
─────────────────────
The original ROS 1 code used::

    import rospkg
    import roslib
    roslib.load_manifest('gps_agent_pkg')

In ROS 2 (ament / colcon) the equivalent is::

    from ament_index_python.packages import get_package_share_directory

Both are tried in order so that the config still loads correctly when:

  - Running under ROS 2 Humble (ament_index_python path).
  - Running in a plain Python / CI environment with neither ROS installed
    (AGENT_ROS falls back to an empty dict with a DEBUG log message).
"""
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base agent config (no ROS dependency)
# ---------------------------------------------------------------------------

AGENT = {
    'dH': 0,
    'x0var': 0,
    'noisy_body_idx': np.array([]),
    'noisy_body_var': np.array([]),
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'smooth_noise': True,
    'smooth_noise_var': 2.0,
    'smooth_noise_renormalize': True,
    'mode': None,
}


# ---------------------------------------------------------------------------
# AgentROS config — populated only when a ROS workspace is available
# ---------------------------------------------------------------------------

def _build_agent_ros_config() -> dict:
    """
    Attempt to build the AGENT_ROS config dict.

    Resolution order:
    1. ament_index_python (ROS 2 / Humble) — verifies gps_agent_pkg is in
       the ament resource index (i.e. the workspace has been colcon-built and
       sourced).
    2. rospkg + roslib (ROS 1 / Noetic) — kept as a fallback for the dual
       ROS 1/2 build scenario.
    3. Empty dict on ImportError / PackageNotFoundError in both cases.

    Returns:
        Populated AGENT_ROS dict, or {} if no ROS workspace is found.
    """
    # ---- ROS 2 path (preferred) ----------------------------------------
    try:
        from ament_index_python.packages import (
            get_package_share_directory,
            PackageNotFoundError,
        )
        try:
            _share = get_package_share_directory('gps_agent_pkg')
            LOGGER.debug('AgentROS config: gps_agent_pkg found via ament at %s', _share)
            return _make_agent_ros_dict()
        except PackageNotFoundError as exc:
            LOGGER.debug(
                'AgentROS config: gps_agent_pkg not in ament index '
                '(run colcon build and source the workspace): %s', exc
            )
            # Fall through to ROS 1 attempt
    except ImportError:
        pass  # ament_index_python not on path — try ROS 1

    # ---- ROS 1 fallback ------------------------------------------------
    try:
        import rospkg
        import roslib
        roslib.load_manifest('gps_agent_pkg')
        LOGGER.debug('AgentROS config: gps_agent_pkg loaded via roslib (ROS 1)')
        return _make_agent_ros_dict()
    except ImportError as exc:
        LOGGER.debug('AgentROS config: roslib/rospkg not available: %s', exc)
    except Exception as exc:
        LOGGER.debug('AgentROS config: roslib.load_manifest failed: %s', exc)

    # ---- No ROS available ----------------------------------------------
    return {}


def _make_agent_ros_dict() -> dict:
    """
    Return the AgentROS hyperparameter dictionary.

    Topic names must match the C++ plugin defaults in robotplugin.cpp.
    PD gains are the 7-DOF PR2 arm defaults (Kp, Ki, Kd, max_effort
    repeated per joint).
    """
    return {
        # ROS 2 topic names (no leading slash — rclpy resolves them
        # relative to the node namespace; add one if using absolute names).
        'trial_command_topic':  'gps_controller_trial_command',
        'reset_command_topic':  'gps_controller_position_command',
        'relax_command_topic':  'gps_controller_relax_command',
        'data_request_topic':   'gps_controller_data_request',
        'sample_result_topic':  'gps_controller_report',

        'trial_timeout':   20,    # seconds to wait for a trial to finish
        'reset_conditions': [],   # list of {TRIAL_ARM: {mode, data}, AUXILIARY_ARM: …}
        'frequency':        20,   # Hz — controller update rate

        'end_effector_points': np.array([]),

        # PR2 7-DOF arm PD gains: [Kp, Ki, Kd, max_effort] × 7 joints
        'pid_params': np.array([
            2400.0, 0.0, 18.0, 4.0,
            1200.0, 0.0, 20.0, 4.0,
            1000.0, 0.0,  6.0, 4.0,
             700.0, 0.0,  4.0, 4.0,
             300.0, 0.0,  6.0, 2.0,
             300.0, 0.0,  4.0, 2.0,
             300.0, 0.0,  2.0, 2.0,
        ]),
    }


AGENT_ROS = _build_agent_ros_config()


# ---------------------------------------------------------------------------
# AgentMuJoCo config
# ---------------------------------------------------------------------------

AGENT_MUJOCO = {
    'substeps': 1,
    'camera_pos': np.array([2., 3., 2., 0., 0., 0.]),
    'image_width': 640,
    'image_height': 480,
    'image_channels': 3,
    'meta_include': [],
}

AGENT_MUJOCO_ADV = {
    'substeps': 1,
    'camera_pos': np.array([2., -3., 2., 0., 0., 0.]),
    'image_width': 640,
    'image_height': 480,
    'image_channels': 3,
    'meta_include': [],
}

# ---------------------------------------------------------------------------
# AgentBox2D config
# ---------------------------------------------------------------------------

AGENT_BOX2D = {
    'render': True,
}
