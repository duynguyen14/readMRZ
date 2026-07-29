from __future__ import annotations

import argparse

from mrz_reader.cli import run_server


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local READMRZ API server.")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Local API port. Default: 8080.",
    )
    args = parser.parse_args()
    return run_server(args.port)


if __name__ == "__main__":
    raise SystemExit(main())
