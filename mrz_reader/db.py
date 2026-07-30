from __future__ import annotations

from pathlib import Path
import re

import pyodbc

from .env_config import env_value, read_env_file


def odbc_yes_no(value: str, default: str = "no") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "yes", "1", "y"}:
        return "yes"
    if normalized in {"false", "no", "0", "n"}:
        return "no"
    return default


def build_connection_string() -> str:
    env = read_env_file()
    driver = env_value(env, "READMRZ_DB_DRIVER", env_value(env, "SQLSERVER_DRIVER", "ODBC Driver 17 for SQL Server"))
    server = env_value(env, "READMRZ_DB_SERVER", env_value(env, "SQLSERVER_SERVER", ""))
    database = env_value(env, "READMRZ_DB_DATABASE", env_value(env, "SQLSERVER_DATABASE", ""))
    trusted_connection = env_value(env, "READMRZ_DB_TRUSTED_CONNECTION", env_value(env, "SQLSERVER_TRUSTED_CONNECTION", "false"))
    trust_server_certificate = env_value(
        env,
        "READMRZ_DB_TRUST_SERVER_CERTIFICATE",
        env_value(env, "SQLSERVER_TRUST_SERVER_CERTIFICATE", "true"),
    )
    username = env_value(env, "READMRZ_DB_USERNAME", env_value(env, "SQLSERVER_USERNAME", ""))
    password = env_value(env, "READMRZ_DB_PASSWORD", env_value(env, "SQLSERVER_PASSWORD", ""))

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"TrustServerCertificate={odbc_yes_no(trust_server_certificate, 'yes')}",
    ]
    if odbc_yes_no(trusted_connection) == "yes":
        parts.append("Trusted_Connection=yes")
    else:
        parts.extend([f"UID={username}", f"PWD={password}"])
    return ";".join(parts)


def connect() -> pyodbc.Connection:
    return pyodbc.connect(build_connection_string(), autocommit=False)


def execute_sql_file(path: Path) -> None:
    sql_text = path.read_text(encoding="utf-8")
    batches = [
        chunk.strip()
        for chunk in re.split(r"^\s*GO\s*$", sql_text, flags=re.IGNORECASE | re.MULTILINE)
        if chunk.strip()
    ]
    with connect() as connection:
        cursor = connection.cursor()
        for batch in batches:
            cursor.execute(batch)
        connection.commit()
