"""Launch the ENT receptionist chat UI on the first available port.

Other apps may be holding common ports, so this scans a preferred range and
falls back to an OS-assigned ephemeral port, then prints the URL to open.

    python scripts/serve.py                 # 127.0.0.1, auto port
    python scripts/serve.py --host 0.0.0.0  # expose on the LAN (test devices)
    python scripts/serve.py --port 8123      # force a specific port
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _is_free(host: str, port: int) -> bool:
    # IMPORTANT: do NOT set SO_REUSEADDR here — it makes an occupied port look
    # free (a port held on 0.0.0.0 would still let a 127.0.0.1 bind "succeed"),
    # which causes a silent collision with another app already on that port.
    # We also probe 0.0.0.0 so a wildcard-bound app is detected on any host.
    for h in {host, "0.0.0.0", "127.0.0.1"}:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((h, port))
            except OSError:
                return False
    return True


def find_free_port(host: str, preferred=range(8000, 8101)) -> int:
    for port in preferred:
        if _is_free(host, port):
            return port
    # Fall back to an OS-assigned ephemeral port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def main() -> int:
    p = argparse.ArgumentParser(description="Serve the ENT receptionist chat UI.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None, help="force a port (else auto-pick)")
    args = p.parse_args()

    bind_host = args.host
    port = args.port or find_free_port("127.0.0.1" if bind_host in ("127.0.0.1", "localhost") else "0.0.0.0")

    import uvicorn
    from app.api.server import create_app

    url_host = "127.0.0.1" if bind_host in ("127.0.0.1", "localhost", "0.0.0.0") else bind_host
    print("=" * 60)
    print("  ENT AI Receptionist — Test Console")
    print(f"  Open:  http://{url_host}:{port}")
    if bind_host == "0.0.0.0":
        print("  (reachable from other devices on your network)")
    print("=" * 60)
    uvicorn.run(create_app(), host=bind_host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
