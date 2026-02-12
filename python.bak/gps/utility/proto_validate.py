"""
Validation helpers for GPS proto3 messages.

Proto3 changed the default value of Sample.T from 100 (proto2 [default=100])
to 0.  Any code that creates a Sample proto without explicitly setting T will
silently produce a zero-length trajectory.  Call check_sample() after every
ParseFromString() or before processing any Sample message.
"""
from __future__ import annotations


def check_sample(sample) -> None:
    """
    Validate a gps_pb2.Sample message.

    Raises:
        ValueError: if T == 0 (proto3 default — caller forgot to set T), or
                    if dU or dX are inconsistent with the packed data arrays.
    """
    if sample.T == 0:
        raise ValueError(
            "Sample.T == 0 (proto3 default). "
            "The sender must set T explicitly before serialising."
        )
    if sample.dU > 0 and len(sample.U) > 0:
        expected = sample.T * sample.dU
        if len(sample.U) != expected:
            raise ValueError(
                f"Sample.U length {len(sample.U)} != T*dU = {expected} "
                f"(T={sample.T}, dU={sample.dU})"
            )
    if sample.dO > 0 and len(sample.obs) > 0:
        expected = sample.T * sample.dO
        if len(sample.obs) != expected:
            raise ValueError(
                f"Sample.obs length {len(sample.obs)} != T*dO = {expected} "
                f"(T={sample.T}, dO={sample.dO})"
            )
