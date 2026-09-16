"""Run with: uv run --extra iceoryx2 python examples/iceoryx2_sum.py server|client."""

from __future__ import annotations

import argparse
from contextlib import suppress

import numpy as np
from npb import BinaryModel, binary_schema

from npb_rpc import Iceoryx2RpcClient, Iceoryx2RpcServer, RpcContext


@binary_schema("npb-rpc.example.iceoryx2.sum.request", version=1)
class SumRequest(BinaryModel):
    values: np.ndarray


@binary_schema("npb-rpc.example.iceoryx2.sum.response", version=1)
class SumResponse(BinaryModel):
    total: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Typed RPC over iceoryx2 shared memory")
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("--endpoint", default="iceoryx2://npb-rpc-sum")
    args = parser.parse_args()
    if args.role == "server":
        with Iceoryx2RpcServer.bind(args.endpoint) as server:

            @server.method("array.sum", request=SumRequest, response=SumResponse)
            def array_sum(request: SumRequest, context: RpcContext) -> SumResponse:
                return SumResponse(total=float(request.values.sum()))

            print(f"serving at {server.endpoint}", flush=True)
            with suppress(KeyboardInterrupt):
                server.serve_forever()
    else:
        with Iceoryx2RpcClient.connect(args.endpoint) as client:
            result = client.call(
                "array.sum",
                SumRequest(values=np.arange(1_000_000, dtype=np.float32)),
                SumResponse,
                timeout=5.0,
            )
        print(result.total)


if __name__ == "__main__":
    main()
