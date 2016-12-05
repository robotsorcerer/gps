"""This is what I used in saving the initial and final pose of superchick

Lekan O'Molux::Dec 03, 2016
High Bay

These are the parameters retrieved from vicon in the home pose of the head

/vicon/headtwist:
	linear: 
	  x: 531.557739258
	  y: 312.454711914
	  z: 956.795532227
	angular: 
	  x: 0.0
	  y: 2.07609391212
	  z: -0.0174869764596
/vicon/Superdude/head
	  frame_id: /world
	child_frame_id: vicon/Superdude/head
	transform: 
	  translation: 
	    x: 0.532392488588
	    y: 0.205199755367
	    z: 0.962377883184
	  rotation: 			#this is the rotation quaternion
	    x: 0.506603297202
	    y: -0.52078853261
	    z: 0.464484263034
	    w: 0.506346494962
/vicon/markers:
	  marker_name: fore
	  subject_name: Superdude
	  segment_name: head
	  translation: 
	    x: 532.777038574
	    y: 204.879714966
	    z: 962.852050781
	  occluded: False
	- 
	  marker_name: chin
	  subject_name: Superdude
	  segment_name: head
	  translation: 
	    x: 542.662841797
	    y: 392.132049561
	    z: 974.222412109
	  occluded: False
	- 
	  marker_name: right_cheek
	  subject_name: Superdude
	  segment_name: head
	  translation: 
	    x: 470.894744873
	    y: 324.036834717
	    z: 969.925720215
	  occluded: False
	- 
	  marker_name: left_cheek
	  subject_name: Superdude
	  segment_name: head
	  translation: 
	    x: 583.813781738
	    y: 332.731903076
	    z: 925.693603516
	  occluded: False


"""



import os.path
import numpy as np
from gps.gui.util import save_pose_to_npz as saver
from gps.gui.util import load_pose_from_npz as loader
from gps import __file__ as gps_filepath

import rospy
import roslib
roslib.load_manifest('gps_agent_pkg')
from tf import transformations as tftf

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
print('BASE_DIR', BASE_DIR)
EXP_DIR = BASE_DIR + '/../experiments/superchick/'
print('EXP DIR', EXP_DIR)

#pose - (joint angle, end effector position, end effector rotation) tuple

ja = np.zeros(3)  #assume the other two bladders are working
ee_pos = np.array([544.5532, 304.3763, 957.4792])
#this was retrieved from vicon with head at rest
quaternion_init = np.array([0.506603297202, -0.52078853261,
	    					0.464484263034, 0.506346494962])
ee_rot = tftf.quaternion_matrix(quaternion_init)

#define start pose; I am working in task space
start_pose = (ja, ee_pos, ee_rot)

saver(EXP_DIR + 'target.npz', 'base_bladder', str(3), 'initial', start_pose) 
saver(EXP_DIR + 'target.npz', 'right_bladder', str(4), 'initial', start_pose) 
saver(EXP_DIR + 'target.npz', 'left_bladder', str(5), 'initial', start_pose) 

ja_tgt = np.zeros(3)
ee_pos_tgt = np.array([544.5532, 304.3763, 961.4792])  #raise head by ~4mm
quaternion_tgt = np.array([0.506603297202+0.2, -0.52078853261-0.2,
	    					0.464484263034+0.2, 0.506346494962+0.2])
ee_rot_tgt = tftf.quaternion_matrix(quaternion_tgt)

#define end pose;  I am working in task space
end_pose = (ja_tgt, ee_pos_tgt, ee_rot_tgt)

"""
save the targets; right now, I only care about base_bladders
so the other bladders are being initialized based on base_bladder
parameters
"""

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