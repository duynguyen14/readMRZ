from __future__ import annotations

import argparse

from mrz_reader.env_config import env_value, read_env_file
from mrz_reader.cli import run_server


def main() -> int:
    env = read_env_file()
    parser = argparse.ArgumentParser(description="Run the local READMRZ API server.")
    parser.add_argument(
        "--host",
        default=env_value(env, "READMRZ_API_HOST", "0.0.0.0"),
        help="API bind host. Default comes from READMRZ_API_HOST or 0.0.0.0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(env_value(env, "READMRZ_API_PORT", "8080")),
        help="API port. Default comes from READMRZ_API_PORT or 8080.",
    )
    args = parser.parse_args()
    return run_server(args.port, host=args.host)


if __name__ == "__main__":
    raise SystemExit(main())
