import numpy as np
from python.gps.gui.util import save_pose_to_npz as saver
ja = np.array([0])
ee_pos = np.array([544.5532, 304.3763, 957.4792])
ee_rot = np.array([
	               [0, 0, 0], [0., 0., 0.], [0., 0.,0.]
	               ])
xx = saver('/../../experiments/superchick/target.npz', 'base_bladder', 1, 'initial', ja) 
xx = saver('/../../experiments/superchick/target.npz', 'base_bladder', 1, 'initial', ee_pos)
xx = saver('/../../experiments/superchick/target.npz', 'base_bladder', 1, 'initial', ee_rot)  
