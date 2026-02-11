""" Default configuration for policy optimization. """
import os

# DEPRECATED: Caffe support removed (2026-02-10)
# Caffe is no longer maintained. Use PyTorch or TensorFlow instead.
# Legacy Caffe files moved to deprecated/caffe_legacy/
construct_fc_network = None

# Config options shared by PyTorch and TensorFlow
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
    'use_gpu': 0,  # Whether or not to use the GPU for caffe training.
    'gpu_id': 0,
    'random_seed': 1,
}


# DEPRECATED: Caffe configuration removed
# Use POLICY_OPT_PYTORCH or POLICY_OPT_TF instead
# If you need Caffe support, see deprecated/caffe_legacy/README.md
POLICY_OPT_CAFFE = None


POLICY_OPT_TF = {
    # Other hyperparameters.
    'copy_param_scope': 'conv_params',
    'fc_only_iterations': 0,
}

POLICY_OPT_TF.update(GENERIC_CONFIG)


POLICY_OPT_PYTORCH = {
    # Other hyperparameters.
    'copy_param_scope': 'conv_params',
    'fc_only_iterations': 0,
}

POLICY_OPT_PYTORCH.update(GENERIC_CONFIG)
