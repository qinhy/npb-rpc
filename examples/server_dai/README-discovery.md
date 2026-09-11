# DepthAI camera RPC with filesystem discovery

The camera service can now advertise itself through `FilesystemDiscovery`, and
clients can discover any healthy instance or select a specific `--server-name`.
Direct endpoint mode remains available by passing `--endpoint`.

## Examples

Start an NNG/IPC camera instance:

```bash
python cli.py server --service camera --server-name camera-a
```

Start another instance with ZeroMQ/TCP:

```bash
python cli.py server --service camera --server-name camera-b --backend zmq --transport tcp
```

List healthy services/instances:

```bash
python cli.py list
```

Discover any healthy camera instance and fetch the six-image frame set:

```bash
python cli.py client --service camera
```

Select one named camera instance:

```bash
python cli.py client --service camera --server-name camera-a
```

Use the legacy one-frame diagnostic path:

```bash
python cli.py client --service camera --server-name camera-a --single --stream rgb --thumbnail
```

Bypass discovery and connect directly:

```bash
python cli.py client --endpoint ipc:///path/to/socket --backend nng
```

Use a custom registry path by adding `--registry /path/to/registry` to server,
client, or list commands. When a server binds to a wildcard/private address but
clients need a different address, pass `--advertise-endpoint` on the server.
