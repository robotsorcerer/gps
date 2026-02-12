"""
Translates gps_agent_pkg/src/util.cpp + include/util.h.

C++ uses std::getline(ss, token, delim) which always yields at least one token
even for an empty input string.  Python str.split() on the other hand returns []
for an empty string.  We replicate the C++ semantics with a custom helper.
"""
from __future__ import annotations


def split(s: str, delim: str) -> list[str]:
    """
    Split *s* on *delim*, replicating C++ std::getline semantics:
      - split("", ":") → [""]   (one empty token)
      - split("a:b", ":") → ["a", "b"]
      - split("a:", ":") → ["a", ""]
    """
    return s.split(delim)


def to_string(v: object) -> str:
    """Equivalent of C++ std::to_string / operator<<."""
    return str(v)
