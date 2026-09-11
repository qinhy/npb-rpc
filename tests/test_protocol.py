from __future__ import annotations

from uuid import uuid4

import pytest

from npb_rpc import RpcProtocolError, Status
from npb_rpc._protocol import (
    decode_envelope,
    encode_envelope,
    request_envelope,
    response_envelope,
)


def test_request_envelope_round_trip() -> None:
    request_id = uuid4().hex
    encoded = encode_envelope(
        request_envelope(
            request_id,
            "array.sum",
            deadline_unix_ns=123,
            metadata={"trace-id": "abc"},
        )
    )

    restored = decode_envelope(encoded)

    assert restored.kind == "request"
    assert restored.request_id == request_id
    assert restored.method == "array.sum"
    assert restored.deadline_unix_ns == 123
    assert restored.metadata == {"trace-id": "abc"}


def test_error_response_round_trip() -> None:
    request_id = uuid4().hex
    encoded = encode_envelope(
        response_envelope(
            request_id,
            status=Status.INVALID_ARGUMENT,
            error_message="bad input",
            error_details={"field": "values"},
        )
    )

    restored = decode_envelope(encoded)

    assert restored.status == Status.INVALID_ARGUMENT
    assert restored.error_message == "bad input"
    assert restored.error_details == {"field": "values"}


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b"[]",
        b'{"v":99,"kind":"request","id":"bad"}',
    ],
)
def test_invalid_envelope_is_rejected(payload: bytes) -> None:
    with pytest.raises(RpcProtocolError):
        decode_envelope(payload)


def test_unknown_response_status_is_rejected() -> None:
    request_id = uuid4().hex
    payload = (
        '{"v":1,"kind":"response","id":"'
        + request_id
        + '","status":"surprise","error":{"message":"bad","details":{}}}'
    ).encode()

    with pytest.raises(RpcProtocolError, match="unknown RPC response status"):
        decode_envelope(payload)
