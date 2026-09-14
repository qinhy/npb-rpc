from __future__ import annotations

from dataclasses import dataclass
from typing import get_type_hints

@dataclass(frozen=True)
class RpcSpec:
    rpc: str
    http: str | None = None
    path: str | None = None


def api(rpc: str, http: str | None = None, path: str | None = None):
    def wrap(fn):
        fn.__rpc_spec__ = RpcSpec(rpc, http, path)
        return fn
    return wrap

def methods(interface: type):
    for name, fn in interface.__dict__.items():
        if getattr(fn, "__rpc_spec__", None) is not None:
            types = get_type_hints(fn)
            spec:RpcSpec= fn.__rpc_spec__
            yield name, spec, types["request"], types["return"]
