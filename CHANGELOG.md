# Changelog

## 0.1.0

- Added synchronous typed unary RPC over ZeroMQ `DEALER`/`ROUTER` sockets.
- Added request correlation, deadlines, string metadata, structured statuses,
  handler aborts, and configurable envelope/message limits.
- Added NPB schema validation and optional `BlobStore` passthrough in both
  directions.
- Added in-process transport tests and runnable sum client/server examples.
- Added optional typed NNG transport support for portable IPC, including
  Windows Named Pipes, plus unified TCP/IPC examples for ZeroMQ and NNG.
- Accounted for NNG REQ/REP routing headers when applying receive-size limits.
- Added pluggable service discovery with atomic filesystem records, heartbeats,
  stale-instance pruning, service enumeration, round-robin resolution, and
  direct ZMQ/NNG calls.
- Added a uv lockfile and documented locked development workflows.
