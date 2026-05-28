
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if __package__ in (None, "") and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __package__ in (None, ""):
    from runtime.udp_factory import UdpHostLifecycleServer
else:
    from .udp_factory import UdpHostLifecycleServer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one host lifecycle service.")
    parser.add_argument("--host", type=str, required=False)
    parser.add_argument("--port", type=int, required=False, default=10000)
    parser.add_argument("--physical-system-id", required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    server = UdpHostLifecycleServer(hostifx=args.host, port=args.port,
        physical_system_id=args.physical_system_id)
    try:
        server.serve_forever()
    finally:
        server.close()

if __name__ == "__main__":
    main()
