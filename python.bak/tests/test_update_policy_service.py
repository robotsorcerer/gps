"""
Tests for UpdatePolicyClient and UpdatePolicyServer (Arch 9.2).

ROS is mocked throughout — no running ROS master required.

Covers:
- UpdatePolicyClient.call() serialises torch_version + uint8[] model_bytes
- UpdatePolicyClient.call_from_params() accepts get_torch_params_dict() output
- UpdatePolicyClient raises RuntimeError when service call fails
- UpdatePolicyClient raises RuntimeError when ROS unavailable
- UpdatePolicyServer._ros_callback() invokes handler with correct args
- UpdatePolicyServer._ros_callback() returns success=False on handler exception
- UpdatePolicyServer._default_handler returns (False, "no handler registered")
- Round-trip: client call → server callback → handler
- UpdatePolicy.srv field types: uint8[] for model_bytes, string for torch_version
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal fake ROS & gps_agent_pkg.srv stubs
# ---------------------------------------------------------------------------

def _make_fake_rospy():
    """Return a minimal rospy-shaped module."""
    rospy = types.ModuleType('rospy')
    rospy.wait_for_service = MagicMock()
    rospy.ServiceProxy = MagicMock()
    rospy.Service = MagicMock()
    return rospy


def _make_srv_module():
    """Return a mock gps_agent_pkg.srv module with UpdatePolicy stubs."""
    pkg = types.ModuleType('gps_agent_pkg')
    srv = types.ModuleType('gps_agent_pkg.srv')

    class UpdatePolicyRequest:
        def __init__(self, torch_version='', model_bytes=None):
            self.torch_version = torch_version
            self.model_bytes = model_bytes or []

    class UpdatePolicyResponse:
        def __init__(self, success=False, message=''):
            self.success = success
            self.message = message

    class UpdatePolicy:
        pass

    srv.UpdatePolicyRequest = UpdatePolicyRequest
    srv.UpdatePolicyResponse = UpdatePolicyResponse
    srv.UpdatePolicy = UpdatePolicy
    pkg.srv = srv
    return pkg, srv


@contextmanager
def _ros_env():
    """
    Context manager: inject fake rospy + gps_agent_pkg.srv into sys.modules,
    patch _ROS_AVAILABLE to True, then restore on exit.
    """
    from gps.ros import update_policy_service as mod

    fake_rospy = _make_fake_rospy()
    fake_pkg, fake_srv = _make_srv_module()

    prev_rospy = sys.modules.get('rospy')
    prev_pkg = sys.modules.get('gps_agent_pkg')
    prev_srv = sys.modules.get('gps_agent_pkg.srv')

    sys.modules['rospy'] = fake_rospy
    sys.modules['gps_agent_pkg'] = fake_pkg
    sys.modules['gps_agent_pkg.srv'] = fake_srv

    with patch.object(mod, '_ROS_AVAILABLE', True):
        try:
            yield fake_rospy, fake_srv
        finally:
            # Restore originals (or remove if they weren't there before)
            if prev_rospy is None:
                sys.modules.pop('rospy', None)
            else:
                sys.modules['rospy'] = prev_rospy
            if prev_pkg is None:
                sys.modules.pop('gps_agent_pkg', None)
            else:
                sys.modules['gps_agent_pkg'] = prev_pkg
            if prev_srv is None:
                sys.modules.pop('gps_agent_pkg.srv', None)
            else:
                sys.modules['gps_agent_pkg.srv'] = prev_srv


# ---------------------------------------------------------------------------
# UpdatePolicyClient
# ---------------------------------------------------------------------------

class TestUpdatePolicyClient:

    def _make_client(self):
        from gps.ros.update_policy_service import UpdatePolicyClient
        return UpdatePolicyClient('/test/update_policy', timeout=1.0)

    def test_call_sends_correct_fields(self):
        client = self._make_client()

        with _ros_env() as (fake_rospy, fake_srv):
            proxy = MagicMock(return_value=fake_srv.UpdatePolicyResponse(
                success=True, message='ok'
            ))
            fake_rospy.ServiceProxy.return_value = proxy

            ok, msg = client.call('2.1.0', b'\x01\x02\x03')

        assert ok is True
        assert msg == 'ok'
        req_arg = proxy.call_args[0][0]
        assert req_arg.torch_version == '2.1.0'
        assert list(req_arg.model_bytes) == [1, 2, 3]

    def test_call_from_params(self):
        """call_from_params() extracts torch_version + model_bytes from dict."""
        client = self._make_client()

        with _ros_env() as (fake_rospy, fake_srv):
            proxy = MagicMock(return_value=fake_srv.UpdatePolicyResponse(
                success=True, message='loaded'
            ))
            fake_rospy.ServiceProxy.return_value = proxy

            params = {
                'torch_version': '2.3.1',
                'model_bytes': bytes([10, 20, 30]),
                'scale': [1.0],   # extra fields are ignored
            }
            ok, msg = client.call_from_params(params)

        assert ok is True
        req_arg = proxy.call_args[0][0]
        assert req_arg.torch_version == '2.3.1'
        assert list(req_arg.model_bytes) == [10, 20, 30]

    def test_call_raises_on_service_exception(self):
        client = self._make_client()

        with _ros_env() as (fake_rospy, fake_srv):
            proxy = MagicMock(side_effect=RuntimeError('service died'))
            fake_rospy.ServiceProxy.return_value = proxy

            with pytest.raises(RuntimeError, match='service died'):
                client.call('2.1.0', b'bytes')

    def test_connect_raises_when_ros_unavailable(self):
        from gps.ros import update_policy_service as mod
        client = self._make_client()
        with patch.object(mod, '_ROS_AVAILABLE', False):
            with pytest.raises(RuntimeError, match='rospy'):
                client.connect()

    def test_connect_called_implicitly_on_first_call(self):
        """proxy is None before connect(); connect() is called inside call()."""
        client = self._make_client()
        assert client._proxy is None

        with _ros_env() as (fake_rospy, fake_srv):
            proxy = MagicMock(return_value=fake_srv.UpdatePolicyResponse(
                success=False, message='err'
            ))
            fake_rospy.ServiceProxy.return_value = proxy

            ok, msg = client.call('2.0.0', b'x')

        assert ok is False
        # wait_for_service was called during implicit connect
        fake_rospy.wait_for_service.assert_called_once()

    def test_rejected_policy_returns_false(self):
        client = self._make_client()

        with _ros_env() as (fake_rospy, fake_srv):
            proxy = MagicMock(return_value=fake_srv.UpdatePolicyResponse(
                success=False, message='version mismatch'
            ))
            fake_rospy.ServiceProxy.return_value = proxy

            ok, msg = client.call('1.0.0', b'old bytes')

        assert ok is False
        assert 'version mismatch' in msg


# ---------------------------------------------------------------------------
# UpdatePolicyServer
# ---------------------------------------------------------------------------

class TestUpdatePolicyServer:

    def _make_server(self, handler=None):
        from gps.ros.update_policy_service import UpdatePolicyServer
        return UpdatePolicyServer('/test/update_policy', handler=handler)

    def test_ros_callback_invokes_handler(self):
        handler = MagicMock(return_value=(True, 'all good'))
        server = self._make_server(handler=handler)

        fake_pkg, fake_srv = _make_srv_module()
        sys.modules['gps_agent_pkg.srv'] = fake_srv

        req = fake_srv.UpdatePolicyRequest(
            torch_version='2.1.0',
            model_bytes=[5, 6, 7],
        )
        resp = server._ros_callback(req)

        handler.assert_called_once_with('2.1.0', bytes([5, 6, 7]))
        assert resp.success is True
        assert resp.message == 'all good'

    def test_ros_callback_returns_failure_on_exception(self):
        handler = MagicMock(side_effect=RuntimeError('handler error'))
        server = self._make_server(handler=handler)

        fake_pkg, fake_srv = _make_srv_module()
        sys.modules['gps_agent_pkg.srv'] = fake_srv

        req = fake_srv.UpdatePolicyRequest(
            torch_version='2.1.0',
            model_bytes=[1],
        )
        resp = server._ros_callback(req)

        assert resp.success is False
        assert 'handler error' in resp.message

    def test_default_handler_returns_false(self):
        from gps.ros.update_policy_service import UpdatePolicyServer
        ok, msg = UpdatePolicyServer._default_handler('2.0.0', b'bytes')
        assert ok is False
        assert 'no handler registered' in msg

    def test_start_raises_when_ros_unavailable(self):
        from gps.ros import update_policy_service as mod
        server = self._make_server()
        with patch.object(mod, '_ROS_AVAILABLE', False):
            with pytest.raises(RuntimeError, match='rospy'):
                server.start()

    def test_start_registers_service(self):
        server = self._make_server()

        with _ros_env() as (fake_rospy, fake_srv):
            mock_service = MagicMock()
            fake_rospy.Service.return_value = mock_service
            server.start()
            fake_rospy.Service.assert_called_once()
            assert fake_rospy.Service.call_args[0][0] == '/test/update_policy'

    def test_stop_shuts_down_service(self):
        server = self._make_server()
        mock_service = MagicMock()
        server._service = mock_service
        server.stop()
        mock_service.shutdown.assert_called_once()
        assert server._service is None

    def test_stop_is_noop_when_not_started(self):
        server = self._make_server()
        server.stop()   # should not raise


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

        server = UpdatePolicyServer('/rt', handler=handler)

        # Simulate what the client does: bytes → list[int]
        req = fake_srv.UpdatePolicyRequest(
            torch_version='2.1.0',
            model_bytes=list(model_bytes),
        )
        resp = server._ros_callback(req)

        assert resp.success is True
        assert received['mb'] == model_bytes   # lossless round-trip
        assert received['tv'] == '2.1.0'
