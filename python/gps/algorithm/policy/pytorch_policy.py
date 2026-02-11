""" This file defines a neural network policy implemented in PyTorch. """
import os
import pickle
import logging

import numpy as np
import torch

from gps.algorithm.policy.policy import Policy

LOGGER = logging.getLogger(__name__)


class PyTorchPolicy(Policy):
    """
    A neural network policy implemented in PyTorch. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.

    U = net.forward(obs) + noise, where noise ~ N(0, diag(var_u))

    Args:
        dU: Dimension of protagonist (controller) actions.
        dV: Dimension of antagonist (disturbance) actions.
        net: Initialized ``torch.nn.Module`` producing actions of shape [1, dU].
        var_u: dU-dimensional noise variance vector for the protagonist.
        var_v: dV-dimensional noise variance vector for the antagonist.
        with_gpu: Whether to use CUDA when available.
    """

    def __init__(self, dU, dV, net, var_u, var_v, with_gpu=False):
        Policy.__init__(self)
        self.net = net
        self.dU = dU
        self.dV = dV
        self.with_gpu = with_gpu
        self.chol_pol_covar = np.diag(np.sqrt(var_u))
        self.chol_pol_covar_v = np.diag(np.sqrt(var_v))
        self.scale = None   # set externally after observing samples
        self.bias = None
        self.x_idx = None   # indices of state dimensions used as input
        self._device = torch.device("cuda" if with_gpu and torch.cuda.is_available() else "cpu")

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Normalise and convert an observation array to a float tensor."""
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs_in = obs.copy()
        if self.scale is not None and self.x_idx is not None:
            obs_in[:, self.x_idx] = obs_in[:, self.x_idx].dot(self.scale) + self.bias
        return torch.FloatTensor(obs_in).to(self._device)

    def act(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector (unused; kept for API compatibility).
            obs: Observation vector, shape [dO].
            t: Time step (unused).
            noise: dU-dimensional noise vector, or None for deterministic.
        Returns:
            dU-dimensional action vector (numpy).
        """
        obs_t = self._obs_to_tensor(obs)
        with torch.no_grad():
            action_mean = self.net(obs_t).cpu().numpy()[0]
        if noise is None:
            return action_mean
        return action_mean + self.chol_pol_covar.T.dot(noise)

    def act_u(self, x, obs, t, noise):
        """Protagonist action — identical to act()."""
        return self.act(x, obs, t, noise)

    def act_v(self, x, obs, t, noise):
        """
        Antagonist disturbance action.
        Uses the same network but applies the antagonist noise covariance.
        """
        obs_t = self._obs_to_tensor(obs)
        with torch.no_grad():
            action_mean = self.net(obs_t).cpu().numpy()[0]
        if noise is None:
            return action_mean
        return action_mean + self.chol_pol_covar_v.T.dot(noise)

    def get_features(self, obs):
        """
        Return the penultimate-layer features for an observation.
        Args:
            obs: Observation vector, shape [dO].
        Returns:
            Feature vector (numpy).
        """
        obs_t = self._obs_to_tensor(obs)
        # If the network exposes a forward_features hook, use it; otherwise
        # fall back to the full forward pass.
        with torch.no_grad():
            if hasattr(self.net, 'forward_features'):
                feat = self.net.forward_features(obs_t).cpu().numpy()[0]
            else:
                feat = self.net(obs_t).cpu().numpy()[0]
        return feat

    def pickle_policy(self, deg_obs, deg_action, checkpoint_path,
                      goal_state=None, should_hash=False):
        """
        Save just the policy weights for later deployment.
        Args:
            deg_obs: Observation dimension.
            deg_action: Action dimension.
            checkpoint_path: Base path for the checkpoint directory.
            goal_state: Optional goal state to store.
            should_hash: Append a UUID to avoid collisions.
        """
        if should_hash:
            import uuid
            checkpoint_path += str(uuid.uuid4())
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_path, '_pol')
        state_dict_path = checkpoint_path + '_pytorch_data'
        pickled_pol = {
            'deg_obs': deg_obs,
            'deg_action': deg_action,
            'chol_pol_covar': self.chol_pol_covar,
            'checkpoint_path_pytorch': state_dict_path,
            'scale': self.scale,
            'bias': self.bias,
            'with_gpu': self.with_gpu,
            'goal_state': goal_state,
            'x_idx': self.x_idx,
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(pickled_pol, f)
        torch.save(self.net.state_dict(), state_dict_path)

    @classmethod
    def load_policy(cls, policy_dict_path, network_config=None):
        """
        Load a saved policy for inference only.
        Args:
            policy_dict_path: Path to the pickled policy dict.
            network_config: Optional dict used to reconstruct the network
                architecture (reserved for future use).
        Returns:
            A PyTorchPolicy instance with restored weights.
        """
        with open(policy_dict_path, 'rb') as f:
            pol_dict = pickle.load(f)

        state_dict_path = pol_dict['checkpoint_path_pytorch']
        # Reconstruct a minimal network matching the saved architecture.
        # If network_config is provided, use it; otherwise infer from pol_dict.
        dU = pol_dict['deg_action']
        dO = pol_dict['deg_obs']
        with_gpu = pol_dict.get('with_gpu', False)

        net = _build_default_net(dO, dU)
        map_location = 'cuda' if with_gpu and torch.cuda.is_available() else 'cpu'
        net.load_state_dict(torch.load(state_dict_path, map_location=map_location))
        net.eval()

        policy = cls(
            dU=dU,
            dV=dU,  # default: dV == dU for symmetric game
            net=net,
            var_u=np.zeros(dU),
            var_v=np.zeros(dU),
            with_gpu=with_gpu,
        )
        policy.chol_pol_covar = pol_dict['chol_pol_covar']
        policy.scale = pol_dict.get('scale')
        policy.bias = pol_dict.get('bias')
        policy.x_idx = pol_dict.get('x_idx')
        return policy


def _build_default_net(dO: int, dU: int, dim_hidden: int = 42) -> torch.nn.Module:
    """Build the default 3-hidden-layer fully-connected network."""
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(dO, dim_hidden)
            self.layer2 = nn.Linear(dim_hidden, dim_hidden)
            self.layer3 = nn.Linear(dim_hidden, dim_hidden)
            self.out_layer = nn.Linear(dim_hidden, dU)

        def forward(self, x):
            out = torch.relu(self.layer1(x))
            out = torch.relu(self.layer2(out))
            out = torch.relu(self.layer3(out))
            return self.out_layer(out)

    return _Net()
