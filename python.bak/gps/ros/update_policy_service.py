"""
UpdatePolicy ROS service helpers (Arch 9.2).

Provides two thin wrappers:

    UpdatePolicyServer  — C++-facing ROS service server (run inside the robot
                          plugin node) that receives a new TorchScript model
                          and forwards it to the controller.

    UpdatePolicyClient  — GPS-algorithm-facing client; replaces the direct
                          TorchParams topic publish with a typed service call
                          so the trainer blocks until the C++ side confirms
                          the model is loaded.

Both classes guard against ROS being unavailable so the module can be imported
in unit tests without a running ROS master.

Service definition (gps_agent_pkg/srv/UpdatePolicy.srv):

    # request
    string  torch_version   # e.g. "2.1.2+cu121"
    uint8[] model_bytes     # raw TorchScript bytes from torch.jit.save()

    ---

    # response
    bool   success
    string message
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

LOGGER = logging.getLogger(__name__)

# ROS is optional; presence is checked lazily so the module loads in CI.
_ROS_AVAILABLE = False
try:
    import rospy  # noqa: F401
    _ROS_AVAILABLE = True
except ImportError:
    pass


class UpdatePolicyClient:
    """
    Typed ROS service client for pushing a new policy to the C++ controller.

    Replaces the ad-hoc ``model_bytes`` string field in ``TorchParams.msg``
    with a first-class ``UpdatePolicy`` service call.  The call is:
      - Synchronous: blocks until the C++ side responds (model loaded or error).
      - Type-safe: ``uint8[]`` for bytes, not ``string``.
      - Introspectable: visible to ``rosservice list`` and rosbag.

    Usage::

        client = UpdatePolicyClient('/gps/update_policy')
        ok, msg = client.call(torch_version='2.1.2', model_bytes=b'...')
    """

    SERVICE_TYPE_NAME = 'gps_agent_pkg/UpdatePolicy'

    def __init__(self, service_name: str = '/gps/update_policy',
                 timeout: float = 10.0) -> None:
        self._service_name = service_name
        self._timeout = timeout
        self._proxy: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Block until the service is available, then cache the ServiceProxy.

        Raises:
            RuntimeError: if ROS is not available or the service times out.
        """
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                'UpdatePolicyClient.connect() requires rospy (ROS not found).'
            )
        import rospy
        from gps_agent_pkg.srv import UpdatePolicy
        LOGGER.info(
            'UpdatePolicyClient: waiting for service %s (timeout=%.1fs)',
            self._service_name, self._timeout,
        )
        rospy.wait_for_service(self._service_name, timeout=self._timeout)
        self._proxy = rospy.ServiceProxy(self._service_name, UpdatePolicy)
        LOGGER.info('UpdatePolicyClient: connected to %s', self._service_name)

    def call(
        self,
        torch_version: str,
        model_bytes: bytes,
    ) -> tuple[bool, str]:
        """
        Send a new TorchScript model to the robot controller.

        Args:
            torch_version: ``torch.__version__`` string (e.g. ``"2.1.2+cu121"``).
            model_bytes:   Raw bytes from ``torch.jit.save()``.

        Returns:
            (success, message) — the C++ side's response.

        Raises:
            RuntimeError: if not connected or ROS service call fails.
        """
        if self._proxy is None:
            self.connect()

        from gps_agent_pkg.srv import UpdatePolicyRequest
        req = UpdatePolicyRequest(
            torch_version=torch_version,
            model_bytes=list(model_bytes),   # uint8[] expects a list/array
        )
        try:
            resp = self._proxy(req)
            if not resp.success:
                LOGGER.warning(
                    'UpdatePolicyClient: controller rejected policy: %s',
                    resp.message,
                )
            else:
                LOGGER.info(
                    'UpdatePolicyClient: policy accepted: %s', resp.message
                )
            return resp.success, resp.message
        except Exception as exc:
            raise RuntimeError(
                f'UpdatePolicyClient: service call to {self._service_name} '
                f'failed: {exc}'
            ) from exc

    # Convenience: accept the dict returned by PolicyOptPyTorch.get_torch_params_dict()
    def call_from_params(self, params: dict) -> tuple[bool, str]:
        """
        Call the service using the dict returned by ``get_torch_params_dict()``.

        Only ``torch_version`` and ``model_bytes`` are used; other fields
        (scale, bias, noise, …) remain in ``TorchParams.msg`` for the trial
        command.
        """
        return self.call(
            torch_version=params['torch_version'],
            model_bytes=params['model_bytes'],
        )


class UpdatePolicyServer:
    """
    ROS service server that receives ``UpdatePolicy`` requests and forwards
    them to a handler callable.

    Intended to be instantiated inside the GPS ROS node (Python side of the
    robot plugin bridge).

    Usage::

        def my_handler(torch_version: str, model_bytes: bytes) -> tuple[bool, str]:
            # load the model and return (success, message)
            ...

        server = UpdatePolicyServer('/gps/update_policy', handler=my_handler)
        server.start()
        rospy.spin()
    """

    def __init__(
        self,
        service_name: str = '/gps/update_policy',
        handler: Optional[Callable[[str, bytes], tuple[bool, str]]] = None,
    ) -> None:
        self._service_name = service_name
        self._handler = handler or self._default_handler
        self._service: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Register the ROS service.  Must be called after ``rospy.init_node()``.

        Raises:
            RuntimeError: if ROS is not available.
        """
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                'UpdatePolicyServer.start() requires rospy (ROS not found).'
            )
        import rospy
        from gps_agent_pkg.srv import UpdatePolicy
        self._service = rospy.Service(
            self._service_name, UpdatePolicy, self._ros_callback
        )
        LOGGER.info('UpdatePolicyServer: listening on %s', self._service_name)

    def stop(self) -> None:
        """Shut down the service."""
        if self._service is not None:
            self._service.shutdown('UpdatePolicyServer stopping')
            self._service = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ros_callback(self, req: Any) -> Any:
        from gps_agent_pkg.srv import UpdatePolicyResponse
        try:
            model_bytes = bytes(req.model_bytes)
            success, message = self._handler(req.torch_version, model_bytes)
        except Exception as exc:
            LOGGER.error('UpdatePolicyServer: handler raised: %s', exc)
            return UpdatePolicyResponse(success=False, message=str(exc))
        return UpdatePolicyResponse(success=success, message=message)

    @staticmethod
    def _default_handler(
        torch_version: str, model_bytes: bytes
    ) -> tuple[bool, str]:
        LOGGER.warning(
            'UpdatePolicyServer: no handler registered; '
            'got %d bytes, torch_version=%s', len(model_bytes), torch_version
        )
        return False, 'no handler registered'
