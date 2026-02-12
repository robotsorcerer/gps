"""
Tests for python/gps_agent_pkg/options.py
Ports C++ test_options_variant.cpp cases (translated to plain-dict semantics).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from gps_agent_pkg.options import OptionsMap


class TestOptionsMap:
    """OptionsMap is dict[str, Any] — verify round-trips for all variant types."""

    def test_is_dict(self):
        opts: OptionsMap = {}
        assert isinstance(opts, dict)

    def test_int_round_trip(self):
        opts: OptionsMap = {"T": 100}
        assert opts["T"] == 100

    def test_float_round_trip(self):
        opts: OptionsMap = {"freq": 50.0}
        assert opts["freq"] == pytest.approx(50.0)

    def test_bool_round_trip(self):
        opts: OptionsMap = {"flag": True}
        assert opts["flag"] is True

    def test_string_round_trip(self):
        opts: OptionsMap = {"name": "joint_1"}
        assert opts["name"] == "joint_1"

    def test_list_int_round_trip(self):
        v = [1, 2, 3, 4]
        opts: OptionsMap = {"dtypes": v}
        assert opts["dtypes"] == v

    def test_numpy_vector_round_trip(self):
        v = np.array([1.0, 2.0, 3.0])
        opts: OptionsMap = {"data": v}
        np.testing.assert_array_equal(opts["data"], v)

    def test_numpy_matrix_round_trip(self):
        M = np.eye(3)
        opts: OptionsMap = {"K": M}
        np.testing.assert_array_equal(opts["K"], M)

    def test_bytes_round_trip(self):
        raw = b"\x00\x01\x02"
        opts: OptionsMap = {"model_bytes": raw}
        assert opts["model_bytes"] == raw

    def test_multiple_keys(self):
        opts: OptionsMap = {
            "T": 5,
            "dU": 7,
            "mode": 1,
            "name": "arm",
        }
        assert len(opts) == 4
        assert opts["T"] == 5

    def test_overwrite_key(self):
        opts: OptionsMap = {"mode": 0}
        opts["mode"] = 1
        assert opts["mode"] == 1

    def test_missing_key_raises(self):
        opts: OptionsMap = {}
        with pytest.raises(KeyError):
            _ = opts["missing"]
