"""
Tests for UpdatePolicyClient and UpdatePolicyServer (Arch 9.2 — rclpy).

ROS 2 / rclpy is mocked throughout — no running ROS daemon required.

Covers:
- UpdatePolicyClient.call() serialises torch_version + uint8[] model_bytes
- UpdatePolicyClient.call_from_params() accepts get_torch_params_dict() output
- UpdatePolicyClient raises RuntimeError when service call fails
- UpdatePolicyClient raises RuntimeError when ROS 2 unavailable
- UpdatePolicyServer._ros2_callback() invokes handler with correct args
- UpdatePolicyServer._ros2_callback() returns success=False on handler exception
- UpdatePolicyServer._default_handler returns (False, "no handler registered")
- Round-trip: client serialises → server callback → handler receives correct bytes
- UpdatePolicy.srv field types: uint8[] for model_bytes, string for torch_version
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal fake rclpy + gps_agent_pkg.srv stubs
# ---------------------------------------------------------------------------

def _make_fake_rclpy():
    """Return a minimal rclpy-shaped module with spin helpers."""
    rclpy = types.ModuleType('rclpy')
    rclpy.ok = MagicMock(return_value=True)
    rclpy.init = MagicMock()
    rclpy.shutdown = MagicMock()
    rclpy.spin_until_future_complete = MagicMock()
    rclpy.node = types.ModuleType('rclpy.node')

    class FakeNode:
        def create_client(self, srv_type, srv_name):
            return MagicMock()
        def create_service(self, srv_type, srv_name, cb):
            return MagicMock()
        def destroy_service(self, handle):
            pass

    rclpy.node.Node = FakeNode
    sys.modules['rclpy.node'] = rclpy.node
    return rclpy


def _make_srv_module():
    """Return a mock gps_agent_pkg.srv module with UpdatePolicy stubs."""
    pkg = types.ModuleType('gps_agent_pkg')
    srv_mod = types.ModuleType('gps_agent_pkg.srv')

    class UpdatePolicyRequest:
        def __init__(self, torch_version='', model_bytes=None):
            self.torch_version = torch_version
            self.model_bytes = model_bytes or []

    class UpdatePolicyResponse:
        def __init__(self, success=False, message=''):
            self.success = success
            self.message = message

    class UpdatePolicy:
        Request = UpdatePolicyRequest
        Response = UpdatePolicyResponse

    srv_mod.UpdatePolicyRequest = UpdatePolicyRequest
    srv_mod.UpdatePolicyResponse = UpdatePolicyResponse
    srv_mod.UpdatePolicy = UpdatePolicy
    pkg.srv = srv_mod
    return pkg, srv_mod


def _make_fake_node():
    """Return a MagicMock rclpy Node with the methods used by the client/server."""
    node = MagicMock()
    # create_client returns a client mock whose wait_for_service returns True
    mock_client = MagicMock()
    mock_client.wait_for_service.return_value = True
    node.create_client.return_value = mock_client
    # create_service returns a service handle mock
    mock_svc_handle = MagicMock()
    node.create_service.return_value = mock_svc_handle
    return node, mock_client, mock_svc_handle


@contextmanager
def _ros2_env():
    """
    Context manager: inject fake rclpy + gps_agent_pkg.srv into sys.modules,
    patch _ROS2_AVAILABLE to True, then restore on exit.
    """
    from gps.ros import update_policy_service as mod

    fake_rclpy = _make_fake_rclpy()
    fake_pkg, fake_srv = _make_srv_module()

    prev = {
        k: sys.modules.get(k)
        for k in ('rclpy', 'rclpy.node', 'gps_agent_pkg', 'gps_agent_pkg.srv')
    }

    sys.modules['rclpy'] = fake_rclpy
    sys.modules['rclpy.node'] = fake_rclpy.node
    sys.modules['gps_agent_pkg'] = fake_pkg
    sys.modules['gps_agent_pkg.srv'] = fake_srv

    with patch.object(mod, '_ROS2_AVAILABLE', True):
        try:
            yield fake_rclpy, fake_srv
        finally:
            for k, v in prev.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


# ---------------------------------------------------------------------------
# UpdatePolicyClient
# ---------------------------------------------------------------------------

class TestUpdatePolicyClient:

    def _make_client(self, node=None):
        from gps.ros.update_policy_service import UpdatePolicyClient
        n = node or MagicMock()
        return UpdatePolicyClient(n, '/test/update_policy', timeout=1.0)

    def test_call_sends_correct_fields(self):
        node, mock_client, _ = _make_fake_node()
        client = self._make_client(node)

        with _ros2_env() as (fake_rclpy, fake_srv):
            # Simulate a successful future result
            future = MagicMock()
            future.done.return_value = True
            resp = fake_srv.UpdatePolicyResponse(success=True, message='ok')
            future.result.return_value = resp
            mock_client.call_async.return_value = future
            node.create_client.return_value = mock_client

            ok, msg = client.call('2.1.0', b'\x01\x02\x03')

        assert ok is True
        assert msg == 'ok'
        req_arg = mock_client.call_async.call_args[0][0]
        assert req_arg.torch_version == '2.1.0'
        assert list(req_arg.model_bytes) == [1, 2, 3]

    def test_call_from_params(self):
        """call_from_params() extracts torch_version + model_bytes from dict."""
        node, mock_client, _ = _make_fake_node()
        client = self._make_client(node)

        with _ros2_env() as (fake_rclpy, fake_srv):
            future = MagicMock()
            future.done.return_value = True
            future.result.return_value = fake_srv.UpdatePolicyResponse(
                success=True, message='loaded'
            )
            mock_client.call_async.return_value = future
            node.create_client.return_value = mock_client

            params = {
                'torch_version': '2.3.1',
                'model_bytes': bytes([10, 20, 30]),
                'scale': [1.0],   # extra fields are ignored
            }
            ok, msg = client.call_from_params(params)

        assert ok is True
        req_arg = mock_client.call_async.call_args[0][0]
        assert req_arg.torch_version == '2.3.1'
        assert list(req_arg.model_bytes) == [10, 20, 30]

    def test_call_raises_on_service_exception(self):
        node, mock_client, _ = _make_fake_node()
        client = self._make_client(node)

        with _ros2_env() as (fake_rclpy, fake_srv):
            mock_client.call_async.side_effect = RuntimeError('service died')
            node.create_client.return_value = mock_client

            with pytest.raises(RuntimeError, match='service died'):
                client.call('2.1.0', b'bytes')

    def test_connect_raises_when_ros_unavailable(self):
        from gps.ros import update_policy_service as mod
        client = self._make_client()
        with patch.object(mod, '_ROS2_AVAILABLE', False):
            with pytest.raises(RuntimeError, match='rclpy'):
                client.connect()

    def test_connect_called_implicitly_on_first_call(self):
        """_client is None before connect(); connect() is called inside call()."""
        node, mock_client, _ = _make_fake_node()
        client = self._make_client(node)
        assert client._client is None

        with _ros2_env() as (fake_rclpy, fake_srv):
            future = MagicMock()
            future.done.return_value = True
            future.result.return_value = fake_srv.UpdatePolicyResponse(
                success=False, message='err'
            )
            mock_client.call_async.return_value = future
            node.create_client.return_value = mock_client

            ok, msg = client.call('2.0.0', b'x')

        assert ok is False
        # wait_for_service was called during implicit connect
        mock_client.wait_for_service.assert_called_once()

    def test_rejected_policy_returns_false(self):
        node, mock_client, _ = _make_fake_node()
        client = self._make_client(node)

        with _ros2_env() as (fake_rclpy, fake_srv):
            future = MagicMock()
            future.done.return_value = True
            future.result.return_value = fake_srv.UpdatePolicyResponse(
                success=False, message='version mismatch'
            )
            mock_client.call_async.return_value = future
            node.create_client.return_value = mock_client

            ok, msg = client.call('1.0.0', b'old bytes')

        assert ok is False
        assert 'version mismatch' in msg


# ---------------------------------------------------------------------------
# UpdatePolicyServer
# ---------------------------------------------------------------------------

class TestUpdatePolicyServer:

    def _make_server(self, handler=None):
        from gps.ros.update_policy_service import UpdatePolicyServer
        node = MagicMock()
        mock_svc_handle = MagicMock()
        node.create_service.return_value = mock_svc_handle
        return UpdatePolicyServer(node, '/test/update_policy', handler=handler), node, mock_svc_handle

    def test_ros2_callback_invokes_handler(self):
        handler = MagicMock(return_value=(True, 'all good'))
        server, node, _ = self._make_server(handler=handler)

        fake_pkg, fake_srv = _make_srv_module()
        sys.modules['gps_agent_pkg.srv'] = fake_srv

        req = fake_srv.UpdatePolicyRequest(
            torch_version='2.1.0',
            model_bytes=[5, 6, 7],
        )
        resp = fake_srv.UpdatePolicyResponse()
        result = server._ros2_callback(req, resp)

        handler.assert_called_once_with('2.1.0', bytes([5, 6, 7]))
        assert result.success is True
        assert result.message == 'all good'

    def test_ros2_callback_returns_failure_on_exception(self):
        handler = MagicMock(side_effect=RuntimeError('handler error'))
        server, node, _ = self._make_server(handler=handler)

        fake_pkg, fake_srv = _make_srv_module()
        sys.modules['gps_agent_pkg.srv'] = fake_srv

        req = fake_srv.UpdatePolicyRequest(
            torch_version='2.1.0',
            model_bytes=[1],
        )
        resp = fake_srv.UpdatePolicyResponse()
        result = server._ros2_callback(req, resp)

        assert result.success is False
        assert 'handler error' in result.message

    def test_default_handler_returns_false(self):
        from gps.ros.update_policy_service import UpdatePolicyServer
        ok, msg = UpdatePolicyServer._default_handler('2.0.0', b'bytes')
        assert ok is False
        assert 'no handler registered' in msg

    def test_start_raises_when_ros_unavailable(self):
        from gps.ros import update_policy_service as mod
        server, node, _ = self._make_server()
        with patch.object(mod, '_ROS2_AVAILABLE', False):
            with pytest.raises(RuntimeError, match='rclpy'):
                server.start()

    def test_start_registers_service(self):
        server, node, mock_svc_handle = self._make_server()

        with _ros2_env() as (fake_rclpy, fake_srv):
            server.start()
            node.create_service.assert_called_once()
            # First positional arg is the service type, second is the name
            call_args = node.create_service.call_args
            assert call_args[0][1] == '/test/update_policy'
            assert server._service_handle is mock_svc_handle

    def test_stop_shuts_down_service(self):
        server, node, mock_svc_handle = self._make_server()
        server._service_handle = mock_svc_handle
        server.stop()
        node.destroy_service.assert_called_once_with(mock_svc_handle)
        assert server._service_handle is None

    def test_stop_is_noop_when_not_started(self):
        server, node, _ = self._make_server()
        server.stop()   # should not raise
        node.destroy_service.assert_not_called()


# ---------------------------------------------------------------------------
# Round-trip: client serialises → server deserialises
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_model_bytes_survive_round_trip(self):
        """uint8[] encoding: bytes → list[int] → bytes must be lossless."""
        from gps.ros.update_policy_service import UpdatePolicyServer
        fake_pkg, fake_srv = _make_srv_module()
        sys.modules['gps_agent_pkg.srv'] = fake_srv

        model_bytes = bytes(range(256))   # all byte values
        received: dict = {}

        def handler(tv: str, mb: bytes) -> tuple:
            received['tv'] = tv
            received['mb'] = mb
            return True, 'ok'

        node = MagicMock()
        server = UpdatePolicyServer(node, '/rt', handler=handler)

        # Simulate what the client does: bytes → list[int]
        req = fake_srv.UpdatePolicyRequest(
            torch_version='2.1.0',
            model_bytes=list(model_bytes),
        )
        resp = fake_srv.UpdatePolicyResponse()
        result = server._ros2_callback(req, resp)

        assert result.success is True
        assert received['mb'] == model_bytes   # lossless round-trip
        assert received['tv'] == '2.1.0'
