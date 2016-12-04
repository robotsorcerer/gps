"""This is what I used in saving the initial and final pose of superchick"""

import os.path
import numpy as np
from gps.gui.util import save_pose_to_npz as saver
from gps.gui.util import load_pose_from_npz as loader
from gps import __file__ as gps_filepath


BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
print('BASE_DIR', BASE_DIR)
EXP_DIR = BASE_DIR + '/../experiments/superchick/'
print('EXP DIR', EXP_DIR)

#pose - (joint angle, end effector position, end effector rotation) tuple

ja = np.zeros(3)  #assume the other two bladders are working
ee_pos = np.array([544.5532, 304.3763, 957.4792])
ee_rot = np.zeros((3,3))

#define start pose
start_pose = (ja, ee_pos, ee_rot)

saver(EXP_DIR + 'target.npz', 'base_bladder', str(3), 'initial', start_pose) 
saver(EXP_DIR + 'target.npz', 'right_bladder', str(4), 'initial', start_pose) 
saver(EXP_DIR + 'target.npz', 'left_bladder', str(5), 'initial', start_pose) 

ja_tgt = np.array([0])
ee_pos_tgt = np.array([544.5532, 304.3763, 961.4792])
ee_rot_tgt = np.array([ [0, 0, 0.12], [0., 0., 0.], [0., 0.13,0.105] ]) #Made up

#define end pose
end_pose = (ja_tgt, ee_pos_tgt, ee_rot_tgt)

#save the targets
saver(EXP_DIR + 'target.npz', 'base_bladder', str(3), 'target', end_pose) 
saver(EXP_DIR + 'target.npz', 'right_bladder', str(4), 'target', end_pose) 
saver(EXP_DIR + 'target.npz', 'left_bladder', str(5), 'target', end_pose) 

ja_res, ee_pos_res, ee_rot_res = loader('target.npz', 'left_bladder', str(1), 'initial',
        default_ja=ja,
        default_ee_pos=ee_pos,
        default_ee_rot=ee_rot
        )

print('init joint angles')
print ja_res

print('\ninit ee_pos')
print ee_pos_res

print('\ninit ee_rot')
print ee_rot_res

ja_res_tgt, ee_pos_res_tgt, ee_rot_res_tgt = loader('target.npz', 'left_bladder', str(1), 'target',
        default_ja=ja_tgt,
        default_ee_pos=ee_pos_tgt,
        default_ee_rot=ee_rot_tgt
        )

print('\ntarget joint angles')
print ja_res_tgt

print('\ntarget ee_pos')
print ee_pos_res_tgt

print('\ntarget ee_rot')
print ee_rot_res_tgt