"""Length-prefixed RPC envelopes for single-message transports."""

from __future__ import annotations

import struct

import numpy as np

from ._errors import RpcProtocolError
from ._protocol import Envelope, decode_envelope, encode_envelope

_LENGTH = struct.Struct("!I")


def _validate_limit(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _pack_message(
    envelope: Envelope,
    payload: np.ndarray | bytes = b"",
    *,
    max_envelope_bytes: int,
) -> bytes:
    encoded_envelope = encode_envelope(envelope, max_bytes=max_envelope_bytes)
    return _LENGTH.pack(len(encoded_envelope)) + encoded_envelope + bytes(payload)


def _unpack_message(
    message: bytes,
    *,
    max_envelope_bytes: int,
    max_message_bytes: int,
) -> tuple[Envelope, bytes]:
    if len(message) < _LENGTH.size:
        raise RpcProtocolError("RPC message is missing its envelope length")
    (envelope_size,) = _LENGTH.unpack_from(message)
    if envelope_size > max_envelope_bytes:
        raise RpcProtocolError(f"RPC envelope exceeds the {max_envelope_bytes:,}-byte limit")
    payload_offset = _LENGTH.size + envelope_size
    if payload_offset > len(message):
        raise RpcProtocolError("RPC message contains a truncated envelope")
    payload = message[payload_offset:]
    if len(payload) > max_message_bytes:
        raise RpcProtocolError(f"RPC payload exceeds the {max_message_bytes:,}-byte limit")
    envelope = decode_envelope(
        message[_LENGTH.size : payload_offset],
        max_bytes=max_envelope_bytes,
    )
    return envelope, payload
