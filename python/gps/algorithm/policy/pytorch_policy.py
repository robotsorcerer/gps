""" This file defines a neural network policy implemented in PyTorch. """
import tempfile
import torch
import numpy as np

from gps.algorithm.policy.policy import Policy


class PyTorchPolicy(Policy):
    """
    A neural network policy implemented in PyTorch. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        test_net: Initialized pytorch network that can run forward.
        var: Du-dimensional noise variance vector.
    """
    def __init__(self, dU, dV, net, obs_tensor, act_op,
            feat_op, var_u, var_v, with_gpu):
        Policy.__init__(self)
        self.net = net
        self.dU = dU
        self.dV = dV
        self.obs_tensor = obs_tensor
        self.act_op = act_op
        self.feat_op = feat_op
        self.chol_pol_covar = np.diag(np.sqrt(var_u))
        self.chol_pol_covar_v = np.diag(np.sqrt(var_v))
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.x_idx = None

    def act(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """

        # Normalize obs.
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias

        if self.with_gpu:
            with torch.cuda.device(0):
                self.net = net.cuda()
                action_mean = self.net(self.act_op.cuda())
        else:
            action_mean = self.net(self.act_op)

        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u[0]  # the DAG computations are batched by default, but we use batch size 1.

    def act_u(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """

        # Normalize obs.
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias

        if self.with_gpu:
            with torch.cuda.device(0):
                self.net = net.cuda()
                action_mean = self.net(self.act_op.cuda())
        else:
            action_mean = self.net(self.act_op)

        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u[0]  # the DAG computations are batched by default, but we use batch size 1.

    def act_v(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """

        # Normalize obs.
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias

        if self.with_gpu:
            with torch.cuda.device(0):
                self.net = net.cuda()
                action_mean = self.net(self.act_op.cuda())
        else:
            action_mean = self.net(self.act_op)

        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar_v.T.dot(noise)
        return u[0]  # the DAG computations are batched by default, but we use batch size 1.

    def get_features(self, obs):
        """
        Return the image features for an observation.
        Args:
            obs: Observation vector.
        """
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        if self.with_gpu:
            # Assume that features don't depend on the robot config, so don't normalize by scale and bias.
            with torch.cuda.device(0):
                feat = self.feat_op(obs.cuda())
        else:
            feat = self.feat_op(obs)
        return feat[0]  # the DAG computations are batched by default, but we use batch size 1.

    # def get_copy_params(self):
    #     param_values = self.sess.run(self.copy_params)
    #     return {self.copy_params[i].name:param_values[i] for i in range(len(self.copy_params))}
    #
    # def set_copy_params(self, param_values):
    #     value_list = [param_values[self.copy_params[i].name] for i in range(len(self.copy_params))]
    #     feeds = {self.copy_params_assign_placeholders[i]:value_list[i] for i in range(len(self.copy_params))}
    #     self.sess.run(self.copy_params_assign_ops, feed_dict=feeds)
    def pickle_policy(self, deg_obs, deg_action, checkpoint_path, goal_state=None, should_hash=False):
        """
        We can save just the policy if we are only interested in running forward at a later point
        without needing a policy optimization class. Useful for debugging and deploying.
        """
        if should_hash is True:
            hash_str = str(uuid.uuid4())
            checkpoint_path += hash_str
        os.mkdir(checkpoint_path + '/')
        checkpoint_path += '/_pol'
        pickled_pol = {'deg_obs': deg_obs, 'deg_action': deg_action, 'chol_pol_covar': self.chol_pol_covar,
                       'checkpoint_path_pytorch': checkpoint_path + '_pytorch_data', 'scale': self.scale, 'bias': self.bias,
                       'with_gpu': self.with_gpu, 'goal_state': goal_state, 'x_idx': self.x_idx}
        pickle.dump(pickled_pol, open(checkpoint_path, "wb"))
        torch.save(self.net.state_dict(), checkpoint_path + '_pytorch_data')

    @classmethod
    def load_policy(cls, policy_dict_path, network_config=None):
        """
        For when we only need to load a policy for the forward pass. For instance, to run on the robot from
        a checkpointed policy.
        """
        pol_dict = pickle.load(open(policy_dict_path, "rb"))

        check_file = pol_dict['checkpoint_path_pytorch']
        self.net.load_state_dict(torch.load(join(check_file))

        device_string = pol_dict['device_string']

        cls_init = cls(pol_dict['deg_action'], np.zeros((1,)), device_string)
        cls_init.chol_pol_covar = pol_dict['chol_pol_covar']
        cls_init.scale = pol_dict['scale']
        cls_init.bias = pol_dict['bias']
        cls_init.x_idx = pol_dict['x_idx']
        return cls_init
