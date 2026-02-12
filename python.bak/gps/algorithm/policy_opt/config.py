""" Default configuration for policy optimization. """
from __future__ import annotations

# policy_opt_utils was part of the Caffe backend and is now in deprecated_caffe/.
# The construct_fc_network factory is not used by the PyTorch backend.
construct_fc_network = None

# config options shared by both caffe and tf.
GENERIC_CONFIG = {
    # Initialization.
    'init_var': 0.1,  # Initial policy variance.
    'init_var_v': 0.1,  # Initial policy variance antag.
    'ent_reg': 0.0,  # Entropy regularizer.
    # Solver hyperparameters.
    'iterations': 5000,  # Number of iterations per inner iteration.
    'batch_size': 25,
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'lr_policy': 'fixed',  # Learning rate policy.
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.005,  # Weight decay.
    'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', etc.).
    # set gpu usage.
    'use_gpu': 0,  # Whether or not to use the GPU for training.
    'gpu_id': 0,
    'random_seed': 1,
}


# Deprecated — kept for backwards-compat with any pickled configs that
# reference POLICY_OPT_CAFFE.  The caffe backend has been removed.
POLICY_OPT_CAFFE: dict = {
    'network_model': construct_fc_network,
    'network_arch_params': {},
    'weights_file_prefix': '',
}
POLICY_OPT_CAFFE.update(GENERIC_CONFIG)


POLICY_OPT_TF: dict = {
    'copy_param_scope': 'conv_params',
    'fc_only_iterations': 0,
}
POLICY_OPT_TF.update(GENERIC_CONFIG)


POLICY_OPT_PYTORCH: dict = {
    'copy_param_scope': 'conv_params',
    'fc_only_iterations': 0,
}
POLICY_OPT_PYTORCH.update(GENERIC_CONFIG)
