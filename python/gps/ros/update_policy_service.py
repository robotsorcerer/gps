"""
UpdatePolicy ROS 2 service helpers (Arch 9.2, rclpy migration).

Provides two thin wrappers:

    UpdatePolicyServer  — C++-facing ROS 2 service server (run inside the robot
                          plugin node) that receives a new TorchScript model
                          and forwards it to the controller.

    UpdatePolicyClient  — GPS-algorithm-facing client; replaces the direct
                          TorchParams topic publish with a typed service call
                          so the trainer blocks until the C++ side confirms
                          the model is loaded.

Key differences from the ROS 1 (rospy) version:

    rospy                          rclpy
    ─────────────────────────────  ──────────────────────────────────────
    rospy.wait_for_service()       client.wait_for_service(timeout_sec=)
    rospy.ServiceProxy(srv, T)     node.create_client(T, srv)
    proxy(req)                     client.call(req)  [blocking, with future]
    rospy.Service(srv, T, cb)      node.create_service(T, srv, cb)
    srv.shutdown()                 node.destroy_service(srv_handle)

Both classes guard against ROS being unavailable so the module can be
imported in unit tests without a running ROS 2 daemon.

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

# ---------------------------------------------------------------------------
# ROS 2 is optional — the module must be importable in unit tests without
# a running daemon or sourced workspace.
# ---------------------------------------------------------------------------
_ROS2_AVAILABLE = False
try:
    import rclpy                          # noqa: F401
    from rclpy.node import Node           # noqa: F401
    _ROS2_AVAILABLE = True
except ImportError:
    pass


class UpdatePolicyClient:
    """
    Typed ROS 2 service client for pushing a new policy to the C++ controller.

    Replaces the ad-hoc ``model_bytes`` string field in ``TorchParams.msg``
    with a first-class ``UpdatePolicy`` service call.  The call is:

      - Synchronous: blocks until the C++ side responds (model loaded or error).
      - Type-safe: ``uint8[]`` for bytes, not ``string``.
      - Introspectable: visible to ``ros2 service list`` and rosbag2.

    Usage::

        client = UpdatePolicyClient(node, '/gps/update_policy')
        client.connect()
        ok, msg = client.call(torch_version='2.1.2', model_bytes=b'...')
    """

    SERVICE_TYPE_NAME = 'gps_agent_pkg/srv/UpdatePolicy'

    def __init__(self, node: Any,
                 service_name: str = '/gps/update_policy',
                 timeout: float = 10.0) -> None:
        """
        Args:
            node:         The rclpy Node that owns this client.
            service_name: Fully-qualified ROS 2 service name.
            timeout:      Seconds to wait for the service to become available.
        """
        self._node = node
        self._service_name = service_name
        self._timeout = timeout
        self._client: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Create the rclpy service client and wait until the server is ready.

        Raises:
            RuntimeError: if rclpy is unavailable or the service times out.
        """
        if not _ROS2_AVAILABLE:
            raise RuntimeError(
                'UpdatePolicyClient.connect() requires rclpy (ROS 2 not found).'
            )

        from gps_agent_pkg.srv import UpdatePolicy
        self._client = self._node.create_client(
            UpdatePolicy, self._service_name
        )
        LOGGER.info(
            'UpdatePolicyClient: waiting for service %s (timeout=%.1fs)',
            self._service_name, self._timeout,
        )
        if not self._client.wait_for_service(timeout_sec=self._timeout):
            raise RuntimeError(
                f'UpdatePolicyClient: service {self._service_name!r} '
                f'not available after {self._timeout}s'
            )
        LOGGER.info('UpdatePolicyClient: connected to %s', self._service_name)

    def call(
        self,
        torch_version: str,
        model_bytes: bytes,
    ) -> tuple[bool, str]:
        """
        Send a new TorchScript model to the robot controller.

        This is a synchronous blocking call: it spins the node's executor
        until the C++ side returns a response or an exception is raised.

        Args:
            torch_version: ``torch.__version__`` string (e.g. ``"2.1.2+cu121"``).
            model_bytes:   Raw bytes from ``torch.jit.save()``.

        Returns:
            ``(success, message)`` — the C++ side's response.

        Raises:
            RuntimeError: if not connected, or if the ROS 2 service call fails.
        """
        if self._client is None:
            self.connect()

        from gps_agent_pkg.srv import UpdatePolicy
        req = UpdatePolicy.Request()
        req.torch_version = torch_version
        req.model_bytes = list(model_bytes)   # uint8[] expects list/array

        try:
            # call() is the synchronous blocking API in rclpy.
            # It spins the node internally until the future resolves.
            future = self._client.call_async(req)
            import rclpy
            rclpy.spin_until_future_complete(self._node, future,
                                             timeout_sec=self._timeout)
            if not future.done():
                raise RuntimeError(
                    f'UpdatePolicyClient: service call timed out after '
                    f'{self._timeout}s'
                )
            resp = future.result()
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
                f'UpdatePolicyClient: service call to {self._service_name!r} '
                f'failed: {exc}'
            ) from exc

    def call_from_params(self, params: dict) -> tuple[bool, str]:
        """
        Call the service using the dict returned by
        ``PolicyOptPyTorch.get_torch_params_dict()``.

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
    ROS 2 service server that receives ``UpdatePolicy`` requests and forwards
    them to a handler callable.

    Intended to be instantiated inside the GPS ROS 2 node (Python side of the
    robot plugin bridge).

    Usage::

        def my_handler(torch_version: str, model_bytes: bytes) -> tuple[bool, str]:
            # load the model, return (success, message)
            ...

        server = UpdatePolicyServer(node, '/gps/update_policy', handler=my_handler)
        server.start()
        rclpy.spin(node)
    """

    def __init__(
        self,
        node: Any,
        service_name: str = '/gps/update_policy',
        handler: Optional[Callable[[str, bytes], tuple[bool, str]]] = None,
    ) -> None:
        """
        Args:
            node:         The rclpy Node that owns this server.
            service_name: Fully-qualified ROS 2 service name.
            handler:      Callable ``(torch_version, model_bytes) → (bool, str)``.
                          Defaults to a no-op that logs a warning and returns
                          ``(False, 'no handler registered')``.
        """
        self._node = node
        self._service_name = service_name
        self._handler = handler or self._default_handler
        self._service_handle: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Register the ROS 2 service on the node.

        Raises:
            RuntimeError: if rclpy is unavailable.
        """
        if not _ROS2_AVAILABLE:
            raise RuntimeError(
                'UpdatePolicyServer.start() requires rclpy (ROS 2 not found).'
            )
        from gps_agent_pkg.srv import UpdatePolicy
        self._service_handle = self._node.create_service(
            UpdatePolicy,
            self._service_name,
            self._ros2_callback,
        )
        LOGGER.info(
            'UpdatePolicyServer: listening on %s', self._service_name
        )

    def stop(self) -> None:
        """Destroy the ROS 2 service handle."""
        if self._service_handle is not None:
            self._node.destroy_service(self._service_handle)
            self._service_handle = None
            LOGGER.info(
                'UpdatePolicyServer: stopped %s', self._service_name
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ros2_callback(self, request: Any, response: Any) -> Any:
        """
        ROS 2 service callback.  Signature is (request, response) → response,
        unlike rospy which is request → response.
        """
        from gps_agent_pkg.srv import UpdatePolicy  # noqa: F401
        try:
            model_bytes = bytes(request.model_bytes)
            success, message = self._handler(request.torch_version, model_bytes)
        except Exception as exc:
            LOGGER.error('UpdatePolicyServer: handler raised: %s', exc)
            response.success = False
            response.message = str(exc)
            return response
        response.success = success
        response.message = message
        return response

    @staticmethod
    def _default_handler(
        torch_version: str, model_bytes: bytes
    ) -> tuple[bool, str]:
        LOGGER.warning(
            'UpdatePolicyServer: no handler registered; '
            'got %d bytes, torch_version=%s',
            len(model_bytes), torch_version,
        )
        return False, 'no handler registered'
