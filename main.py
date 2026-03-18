import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"\n  VoicePrint-ID Phase 1 MVP\n  Dashboard: http://localhost:{args.port}\n  Swagger:   http://localhost:{args.port}/docs\n")

    (PROJECT_ROOT / "data" / "speakers").mkdir(parents=True, exist_ok=True)

    import server
    import uvicorn
    uvicorn.run(server.app, host=args.host, port=args.port, reload=False, log_level=args.log_level)

if __name__ == "__main__":
    main()