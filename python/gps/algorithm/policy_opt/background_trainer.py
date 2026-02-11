"""
BackgroundPolicyTrainer — non-blocking policy training.

The GPS main loop alternates between collecting samples and running
policy_opt.update(). On a real robot the update() call (5 000 Adam steps)
can take several seconds, stalling the data-collection thread.

BackgroundPolicyTrainer wraps any PolicyOpt and offloads update() to a
daemon thread so the GPS loop can immediately start collecting the next
batch of samples.  PyTorch's C++ backend releases the GIL during the
forward/backward pass, so a training thread and an inference thread do
not meaningfully contend with each other.

Usage::

    trainer = BackgroundPolicyTrainer(policy_opt)

    # Non-blocking — returns immediately:
    trainer.update_async(obs, tgt_mu, tgt_prc, tgt_wt)

    # ... collect next samples while training runs in the background ...

    # Block until the latest update is incorporated (call before next GPS iter):
    trainer.wait()

    # The policy is always safe to read from any thread:
    action = trainer.policy.act(x, obs, t, noise)
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class BackgroundPolicyTrainer:
    """
    Thread-safe wrapper around any PolicyOpt that moves training off the
    main GPS thread.

    Thread safety model:
      - update_async() spawns a daemon thread that calls _opt.update().
      - When training completes the new policy is swapped in under _lock.
      - policy, last_loss, and last_grad_norm are always read under _lock,
        so they are safe to access from the main thread at any time.
      - Only one training job runs at a time; a second call to update_async()
        while training is active is logged and dropped.
    """

    def __init__(self, policy_opt: Any) -> None:
        self._opt = policy_opt
        self._lock = threading.Lock()
        self._latest_policy = policy_opt.policy
        self._training = False
        self._thread: threading.Thread | None = None
        self._last_loss: float = float('nan')
        self._last_grad_norm: float = float('nan')
        self._error: Exception | None = None

    # ------------------------------------------------------------------
    # Public read-only properties (thread-safe)
    # ------------------------------------------------------------------

    @property
    def policy(self):
        """Most recently trained policy; safe to call from any thread."""
        with self._lock:
            return self._latest_policy

    @property
    def is_training(self) -> bool:
        """True while a background update() is running."""
        return self._training

    @property
    def last_loss(self) -> float:
        """Final training loss from the most recent completed update."""
        with self._lock:
            return self._last_loss

    @property
    def last_grad_norm(self) -> float:
        """Gradient norm from the most recent completed update."""
        with self._lock:
            return self._last_grad_norm

    @property
    def last_error(self) -> Exception | None:
        """Exception raised by the last background update, or None."""
        with self._lock:
            return self._error

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update_async(
        self,
        obs: np.ndarray,
        tgt_mu: np.ndarray,
        tgt_prc: np.ndarray,
        tgt_wt: np.ndarray,
    ) -> bool:
        """
        Start a training update in a background thread.

        Returns:
            True  — training started.
            False — a previous update is still running; this call is a no-op.
        """
        if self._training:
            LOGGER.warning(
                'BackgroundPolicyTrainer: update_async() called while '
                'training is still in progress — skipping this update.'
            )
            return False

        with self._lock:
            self._error = None

        self._training = True
        self._thread = threading.Thread(
            target=self._train_worker,
            args=(obs, tgt_mu, tgt_prc, tgt_wt),
            daemon=True,
            name='gps-policy-trainer',
        )
        self._thread.start()
        LOGGER.debug('BackgroundPolicyTrainer: training thread started')
        return True

    def update(
        self,
        obs: np.ndarray,
        tgt_mu: np.ndarray,
        tgt_prc: np.ndarray,
        tgt_wt: np.ndarray,
    ):
        """
        Blocking update — equivalent to calling update_async() + wait().

        Provides a drop-in replacement for PolicyOpt.update() when the
        non-blocking behaviour is not needed (e.g. during unit tests).
        Raises any exception that occurred during training.
        """
        self.update_async(obs, tgt_mu, tgt_prc, tgt_wt)
        self.wait()
        if self._error is not None:
            raise self._error
        return self.policy

    def wait(self, timeout: float | None = None) -> bool:
        """
        Block the calling thread until the current training job finishes.

        Args:
            timeout: Maximum seconds to wait (None = wait forever).

        Returns:
            True if training completed, False if timeout was reached.
        """
        if self._thread is None or not self._thread.is_alive():
            return True
        self._thread.join(timeout=timeout)
        timed_out = self._thread.is_alive()
        if timed_out:
            LOGGER.warning(
                'BackgroundPolicyTrainer.wait() timed out after %.1fs', timeout
            )
        return not timed_out

    # ------------------------------------------------------------------
    # Delegate attribute access to the wrapped PolicyOpt
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Delegate anything not defined here to the underlying PolicyOpt
        # so BackgroundPolicyTrainer can be used as a drop-in replacement.
        return getattr(self._opt, name)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_worker(
        self,
        obs: np.ndarray,
        tgt_mu: np.ndarray,
        tgt_prc: np.ndarray,
        tgt_wt: np.ndarray,
    ) -> None:
        try:
            new_policy = self._opt.update(obs, tgt_mu, tgt_prc, tgt_wt)
            loss = getattr(self._opt, '_last_loss', float('nan'))
            grad_norm = getattr(self._opt, '_last_grad_norm', float('nan'))
            with self._lock:
                self._latest_policy = new_policy
                self._last_loss = loss
                self._last_grad_norm = grad_norm
            LOGGER.debug(
                'BackgroundPolicyTrainer: training complete  '
                'loss=%.6f  grad_norm=%.4f', loss, grad_norm,
            )
        except Exception as exc:
            LOGGER.error('BackgroundPolicyTrainer: training failed: %s', exc)
            with self._lock:
                self._error = exc
        finally:
            self._training = False
