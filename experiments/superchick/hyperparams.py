# To get started, copy over hyperparams from another experiment.
# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.

""" Hyperparameters for Superchick trajectory optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

#ros tf imports
import rospy
import roslib
roslib.load_manifest('gps_agent_pkg')
from tf import transformations as trans

from gps import __file__ as gps_filepath
from gps.agent.ros.agent_chick import AgentCHICK
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE, TASK_SPACE, \
        RIGHT_BLADDER, LEFT_BLADDER, BASE_BLADDER
from gps.utility.general_utils import get_ee_points
from gps.gui.config import generate_experiment_info

#We have one head and we are going to a target point in the world.
EE_POINTS = np.array([[0.02, -0.025, 0.05], [0.02, -0.025, -0.05], [0.02, 0.05, 0.0]
                      ])

SENSOR_DIMS = {
    JOINT_ANGLES: 1,
    JOINT_VELOCITIES: 1,
    END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0],
    END_EFFECTOR_POINT_VELOCITIES: 3 * EE_POINTS.shape[0],
    ACTION: 3,  #using 1 bladder for now
}

SUPERCHICK_GAINS = np.array([3.09, 1.08, 0.393])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/superchick/'

x0s = []
ee_tgts = []
reset_conditions = []

common = {
    'experiment_name': 'superchick' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

# Set up each condition.
#cause the enums are described from 3, 4, 5 for the bladders in proto
"""
We override the default joint and end_effector eigens for the pr2 robot
and instantiate our soft robot
"""
DEFAULT_JOINT_ANGLES = np.zeros(3) 
DEFAULT_END_EFFECTOR_POSITIONS = np.array([544.5532, 304.3763, 957.4792])
quaternion_init = np.array([0.506603297202, -0.52078853261,
                            0.464484263034, 0.506346494962])
proj_init = trans.quaternion_matrix(quaternion_init)
#slice the rotation part from the homogeneous matrix
DEFAULT_END_EFFECTOR_ROTATIONS = proj_init[0:3, 0:3]

TARGET_END_EFFECTOR_POSITIONS = np.array([544.5532, 304.3763, 961.4792])  #raise head by ~4mm
quaternion_tgt = np.array([0.506603297202+0.2, -0.52078853261-0.2,
                            0.464484263034+0.2, 0.506346494962+0.2])
proj_tgt = trans.quaternion_matrix(quaternion_tgt)
TARGET_END_EFFECTOR_ROTATIONS = proj_tgt[0:3, 0:3]

for i in xrange(common['conditions']#, common['conditions']+6
                ):

    ja_x0, ee_pos_x0, ee_rot_x0 = load_pose_from_npz(
        common['target_filename'], 'base_bladder', str(i), 'initial',
        default_ja=DEFAULT_JOINT_ANGLES,
        default_ee_pos=DEFAULT_END_EFFECTOR_POSITIONS,
        default_ee_rot=DEFAULT_END_EFFECTOR_ROTATIONS
    )
    ja_right, _, _ = load_pose_from_npz(
        common['target_filename'], 'right_bladder', str(i), 'initial',
        default_ja=DEFAULT_JOINT_ANGLES,
        default_ee_pos=DEFAULT_END_EFFECTOR_POSITIONS,
        default_ee_rot=DEFAULT_END_EFFECTOR_ROTATIONS
    )
    ja_left, _, _ = load_pose_from_npz(
        common['target_filename'], 'left_bladder', str(i), 'initial',
        default_ja=DEFAULT_JOINT_ANGLES,
        default_ee_pos=DEFAULT_END_EFFECTOR_POSITIONS,
        default_ee_rot=DEFAULT_END_EFFECTOR_ROTATIONS
    )
    _, ee_pos_tgt, ee_rot_tgt = load_pose_from_npz(
        common['target_filename'], 'base_bladder', str(i), 'target',
        default_ja=DEFAULT_JOINT_ANGLES, #same anyway
        default_ee_pos=np.array([544.5532, 304.3763, 961.4792]),
        default_ee_rot=TARGET_END_EFFECTOR_ROTATIONS
    )

    print('ja_x0: ', ja_x0)

    """
    The state includes the joint angles and velocities (7x2=14) 
    and the pose&velocity of the end effector, represented as 
    3 points in 3D (3x3x2=18).
    The rest of the state is left to zero because we assume that
    the initial velocity of the arm is 0.

    =================================================================
    For superchck, we have joint angles and vels (3x2=6)
    pose and vel of end effector points as 3 point in 3D = (3x3x2=18).
    The rest of the state is initialized zero because 
    I assume the head velocity is 0.

    
    """
    x0 = np.zeros(18)
    x0[:3] = ja_x0
    x0[6:(6+3*EE_POINTS.shape[0])] = np.ndarray.flatten(
        get_ee_points(EE_POINTS, ee_pos_x0, ee_rot_x0).T #3X3 mat
    )  

    ee_tgt = np.ndarray.flatten(
        get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T
    )

    right_x0 = np.zeros(3)
    right_x0[:] = ja_right

    left_x0 = np.zeros(3)
    left_x0[:] = ja_left

    """
    superchick cannot be controlled in the joint space since 
    it has no encoders
    """

    reset_condition = {
        BASE_BLADDER: {
            'mode': TASK_SPACE,
            'data': x0[0:3],
        },
        RIGHT_BLADDER: {
            'mode': TASK_SPACE,
            'data': right_x0,
        },
        LEFT_BLADDER: {
            'mode': TASK_SPACE,
            'data': left_x0,
        },
    }

    x0s.append(x0)
    ee_tgts.append(ee_tgt)
    reset_conditions.append(reset_condition)


if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentCHICK,
    'dt': 0.05,
    'conditions': common['conditions'],
    'T': 100,
    'x0': x0s,
    'ee_points_tgt': ee_tgts,
    'reset_conditions': reset_conditions,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'end_effector_points': EE_POINTS,
    'obs_include': [],
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 10,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / SUPERCHICK_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 0.5,
    'stiffness_vel': 0.25,
    'final_weight': 50,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-3 / SUPERCHICK_GAINS,
}

fk_cost1 = {
    'type': CostFK,
    # Target end effector is subtracted out of EE_POINTS in ROS so goal
    # is 0.
    'target_end_effector': np.zeros(3),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 0.0001,
    'ramp_option': RAMP_LINEAR,
}

fk_cost2 = {
    'type': CostFK,
    'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 1.0,
    'l2': 0.0,
    'wp_final_multiplier': 10.0,  # Weight multiplier on final timestep.
    'ramp_option': RAMP_FINAL_ONLY,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost1, fk_cost2],
    'weights': [1.0, 1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': 0,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': 5,
}

common['info'] = generate_experiment_info(config)
