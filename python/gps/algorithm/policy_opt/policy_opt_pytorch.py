""" Policy optimization backed by a PyTorch neural network. """
import copy
import logging
import tempfile
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gps.algorithm.policy_opt.config import POLICY_OPT_PYTORCH
from gps.algorithm.policy.pytorch_policy import PyTorchPolicy, _build_default_net
from gps.algorithm.policy_opt.policy_opt import PolicyOpt

LOGGER = logging.getLogger(__name__)


class PolicyOptPyTorch(PolicyOpt):
    """
    Policy optimization using a PyTorch MLP.

    Trains via a weighted Gaussian log-likelihood loss matching the
    GPS trajectory-centric cost formulation:

        L = (1 / 2N) Σ_i  w_i  (π - μ)ᵀ Λ (π - μ)

    where π = net(obs), μ = tgt_mu, Λ = tgt_prc (precision matrix),
    w = tgt_wt (per-sample weight).
    """

    def __init__(self, hyperparams, dO, dU, dV=None):
        config = copy.deepcopy(POLICY_OPT_PYTORCH)
        config.update(hyperparams)
        # dV defaults to dU for a symmetric iDG game
        if dV is None:
            dV = dU
        PolicyOpt.__init__(self, config, dO, dU, dV)

        torch.manual_seed(self._hyperparams['random_seed'])

        self.use_cuda = (
            self._hyperparams.get('use_gpu', 0) == 1
            and torch.cuda.is_available()
        )
        self._device = torch.device("cuda" if self.use_cuda else "cpu")

        self.batch_size = self._hyperparams['batch_size']
        self.pytorch_iter = 0

        # Build the network and move it to the target device.
        dim_hidden = self._hyperparams.get('dim_hidden', 42)
        self._net = _build_default_net(dO, dU, dim_hidden=dim_hidden).to(self._device)

        # Build obs-index lists from network_params (if provided).
        self.x_idx: list = []
        self.img_idx: list = []
        network_params = self._hyperparams.get('network_params', {})
        if network_params:
            if 'obs_image_data' not in network_params:
                network_params['obs_image_data'] = []
            i = 0
            for sensor in network_params.get('obs_include', []):
                dim = network_params['sensor_dims'][sensor]
                if sensor in network_params['obs_image_data']:
                    self.img_idx += list(range(i, i + dim))
                else:
                    self.x_idx += list(range(i, i + dim))
                i += dim
        if not self.x_idx:
            # Fall back: use all observation dimensions
            self.x_idx = list(range(dO))

        # Variance arrays (diagonal of the policy covariance).
        self.var_u = self._hyperparams['init_var'] * np.ones(dU)
        self.var_v = self._hyperparams['init_var_v'] * np.ones(dV)

        # Build the policy wrapper that exposes the GPS act() interface.
        self.policy = PyTorchPolicy(
            dU=dU,
            dV=dV,
            net=self._net,
            var_u=self.var_u,
            var_v=self.var_v,
            with_gpu=self.use_cuda,
        )
        self.policy.x_idx = self.x_idx

        # Optimiser — initialised lazily via init_solver() or on first update().
        self._optimizer: optim.Optimizer | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_optimizer(self) -> None:
        if self._optimizer is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                lr=self._hyperparams['lr'],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self._hyperparams['weight_decay'],
            )

    def _weighted_gaussian_loss(
        self,
        pred: torch.Tensor,
        tgt_mu: torch.Tensor,
        tgt_prc: torch.Tensor,
        tgt_wt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted quadratic loss under precision matrix.

        L = (1 / 2N) Σ_i  w_i  (pred_i - mu_i)ᵀ  Λ_i  (pred_i - mu_i)

        Args:
            pred:    [B, dU]
            tgt_mu:  [B, dU]
            tgt_prc: [B, dU, dU]
            tgt_wt:  [B, 1, 1]   (already folded into tgt_prc on entry)
        Returns:
            Scalar loss tensor.
        """
        diff = (pred - tgt_mu).unsqueeze(1)         # [B, 1, dU]
        mahal = diff.bmm(tgt_prc).bmm(diff.transpose(1, 2))  # [B, 1, 1]
        return mahal.squeeze().mean() * 0.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_solver(self) -> None:
        """Initialise the Adam optimiser (called automatically by update())."""
        self._ensure_optimizer()

    def update(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """
        Update the policy network.

        Args:
            obs:     [N, T, dO]  observation array.
            tgt_mu:  [N, T, dU]  target action mean.
            tgt_prc: [N, T, dU, dU]  precision matrices.
            tgt_wt:  [N, T]  per-sample weights.
        Returns:
            Updated PyTorchPolicy.
        """
        self._ensure_optimizer()
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # ---- Weight normalisation ----
        tgt_prc_orig = np.reshape(tgt_prc, [N * T, dU, dU])
        tgt_wt = tgt_wt.copy()
        tgt_wt *= float(N * T) / np.sum(tgt_wt)
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        tgt_wt = np.minimum(tgt_wt, 2.0 * mn)
        tgt_wt /= mn

        # ---- Flatten ----
        obs = np.reshape(obs, (N * T, dO))
        tgt_mu = np.reshape(tgt_mu, (N * T, dU))
        tgt_prc = np.reshape(tgt_prc, (N * T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N * T, 1, 1))

        # ---- Fold weights into precision ----
        tgt_prc = tgt_wt * tgt_prc

        # ---- Observation normalisation (computed once) ----
        if self.policy.scale is None or self.policy.bias is None:
            self.policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3)
            )
            self.policy.bias = -np.mean(
                obs[:, self.x_idx].dot(self.policy.scale), axis=0
            )
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.policy.scale) + self.policy.bias

        # ---- Convert to tensors ----
        obs_t = torch.FloatTensor(obs).to(self._device)
        mu_t = torch.FloatTensor(tgt_mu).to(self._device)
        prc_t = torch.FloatTensor(tgt_prc).to(self._device)

        # ---- Mini-batch training loop ----
        batches_per_epoch = max(1, int(np.floor(N * T / self.batch_size)))
        idx = np.arange(N * T)
        np.random.shuffle(idx)
        average_loss = 0.0
        iterations = self._hyperparams['iterations']

        self._net.train()
        for i in range(iterations):
            start_idx = int(i * self.batch_size % (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx: start_idx + self.batch_size]

            self._optimizer.zero_grad()
            pred = self._net(obs_t[idx_i])
            loss = self._weighted_gaussian_loss(
                pred, mu_t[idx_i], prc_t[idx_i],
                torch.FloatTensor(tgt_wt[idx_i]).to(self._device),
            )
            if not torch.isfinite(loss):
                LOGGER.warning(
                    'NaN/Inf loss at iteration %d (value=%s) — skipping backward pass',
                    i, loss.item()
                )
                self._optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=10.0)
            self._optimizer.step()

            average_loss += loss.item()
            if (i + 1) % 50 == 0:
                LOGGER.info('PyTorch iteration %d, average loss %f',
                            i + 1, average_loss / 50)
                average_loss = 0.0
        self._net.eval()

        self.pytorch_iter += iterations

        # ---- Update diagonal variance ----
        A = np.sum(tgt_prc_orig, axis=0) + 2 * N * T * \
            self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A /= np.sum(tgt_wt)
        self.var_u = 1.0 / np.diag(A)
        self.policy.chol_pol_covar = np.diag(np.sqrt(self.var_u))

        return self.policy

    def prob(self, obs):
        """
        Run policy forward.
        Args:
            obs: [N, T, dO]
        Returns:
            (output, pol_sigma, pol_prec, pol_det_sigma)
        """
        dU = self._dU
        N, T = obs.shape[:2]

        if self.policy.scale is not None:
            obs = obs.copy()
            # obs[n, :, x_idx] is shape [T, len(x_idx)]; dot with [len, len] scale
            x = self.x_idx
            for n in range(N):
                obs[n, :][np.ix_(range(T), x)] = (
                    obs[n][:, x].dot(self.policy.scale) + self.policy.bias
                )

        obs_flat = obs.reshape(N * T, -1)
        obs_t = torch.FloatTensor(obs_flat).to(self._device)
        self._net.eval()
        with torch.no_grad():
            out_flat = self._net(obs_t).cpu().numpy()
        output = out_flat.reshape(N, T, dU)

        pol_sigma = np.tile(np.diag(self.var_u), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var_u), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var_u), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg: float) -> None:
        """Set the entropy regularisation coefficient."""
        self._hyperparams['ent_reg'] = ent_reg

    def save_model(self, fname: str) -> None:
        """Save network weights to fname."""
        LOGGER.debug('Saving model to: %s', fname)
        torch.save(self._net.state_dict(), fname)

    def restore_model(self, fname: str) -> None:
        """Restore network weights from fname."""
        LOGGER.debug('Restoring model from: %s', fname)
        state = torch.load(fname, map_location=self._device)
        self._net.load_state_dict(state)

    def export_torchscript(self) -> bytes:
        """
        Serialise the trained policy network to TorchScript bytes.

        The bytes can be shipped over ROS (TorchParams.model_bytes) and
        loaded in C++ with::

            torch::jit::load(std::istringstream(model_bytes), torch::kCPU)

        Returns:
            Raw bytes of the TorchScript archive.
        """
        self._net.eval()
        scripted = torch.jit.script(self._net)
        buf = torch.jit.freeze(scripted) if hasattr(torch.jit, 'freeze') else scripted
        import io
        buffer = io.BytesIO()
        torch.jit.save(buf, buffer)
        return buffer.getvalue()

    def get_torch_params_dict(self) -> dict:
        """
        Return a dict ready to populate a gps_agent_pkg/TorchParams ROS message.

        Keys match TorchParams.msg fields:
            model_bytes (bytes)
            scale       (list[float], length dO — diagonal of scale matrix)
            bias        (list[float], length dO)
            noise       (list[float], length T*dU, row-major)  — zeros placeholder
            dim_bias    (int = dO)
            dU          (int)

        The noise field is left all-zero here; the caller should overwrite it
        with the pre-sampled noise array for the current trial.
        """
        model_bytes = self.export_torchscript()
        dO = self._dO
        dU = self._dU

        if self.policy.scale is not None:
            scale_diag = list(float(v) for v in self.policy.scale.diagonal())
            bias = list(float(v) for v in self.policy.bias)
        else:
            scale_diag = [1.0] * dO
            bias = [0.0] * dO

        return {
            'model_bytes': model_bytes,
            'torch_version': torch.__version__,   # for C++ version handshake
            'scale': scale_diag,
            'bias': bias,
            'noise': [],       # caller fills per-trial noise
            'dim_bias': dO,
            'dU': dU,
        }

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------

    def __getstate__(self):
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmp_path = f.name
        try:
            self.save_model(tmp_path)
            with open(tmp_path, 'rb') as f:
                wts = f.read()
        finally:
            os.unlink(tmp_path)
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'dV': self._dV,
            'scale': self.policy.scale,
            'bias': self.policy.bias,
            'x_idx': self.policy.x_idx,
            'chol_pol_covar': self.policy.chol_pol_covar,
            'pytorch_iter': self.pytorch_iter,
            'wts': wts,
        }

    def __setstate__(self, state):
        self.__init__(state['hyperparams'], state['dO'], state['dU'], state.get('dV'))
        self.policy.scale = state.get('scale')
        self.policy.bias = state.get('bias')
        self.policy.x_idx = state.get('x_idx')
        self.policy.chol_pol_covar = state.get('chol_pol_covar')
        self.pytorch_iter = state.get('pytorch_iter', 0)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmp_path = f.name
            f.write(state['wts'])
        try:
            self.restore_model(tmp_path)
        finally:
            os.unlink(tmp_path)
