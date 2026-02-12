"""
Tests for python/gps_agent_pkg/util.py
Ports all C++ GTest cases from test_util.cpp.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from gps_agent_pkg.util import split, to_string


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

class TestSplit:
    def test_comma_three_tokens(self):
        parts = split("a,b,c", ",")
        assert parts == ["a", "b", "c"]

    def test_space_delimiter(self):
        parts = split("foo bar baz", " ")
        assert parts == ["foo", "bar", "baz"]

    def test_single_token(self):
        parts = split("hello", ",")
        assert len(parts) == 1
        assert parts[0] == "hello"

    def test_empty_string_yields_one_empty_token(self):
        # C++ std::getline semantics: empty input → one empty token
        parts = split("", ",")
        assert len(parts) == 1
        assert parts[0] == ""

    def test_two_tokens(self):
        parts = split("key=value", "=")
        assert parts == ["key", "value"]

    def test_combined_from_two_calls(self):
        # C++ test checks that split() *appends* to the vector.
        # In Python we accumulate via list concatenation.
        combined = split("a,b", ",") + split("c,d", ",")
        assert len(combined) == 4
        assert combined[2] == "c"
        assert combined[3] == "d"

    def test_newline_delimiter(self):
        parts = split("line1\nline2\nline3", "\n")
        assert parts == ["line1", "line2", "line3"]

    def test_trailing_delimiter(self):
        parts = split("a,", ",")
        assert parts == ["a", ""]

    def test_leading_delimiter(self):
        parts = split(",a", ",")
        assert parts == ["", "a"]


# ---------------------------------------------------------------------------
# to_string
# ---------------------------------------------------------------------------

class TestToString:
    def test_zero(self):
        assert to_string(0) == "0"

    def test_positive_int(self):
        assert to_string(42) == "42"

    def test_negative_int(self):
        assert to_string(-7) == "-7"

    def test_float_non_empty(self):
        s = to_string(3.14)
        assert len(s) > 0

    def test_double_non_empty(self):
        s = to_string(2.718)
        assert len(s) > 0

    def test_string_passthrough(self):
        assert to_string("hello") == "hello"

    def test_bool_true(self):
        assert to_string(True) == "True"
