# -*- coding: utf-8 -*-
# C:\apano_Dienste\finance-analytics\db_connection.py
import os
import urllib.parse
from sqlalchemy import create_engine

# .env optional laden
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def _opts_for_driver(driver: str, opts_env: str | None) -> str:
    """
    Liefert ODBC-Optionen. Vorrang hat die .env (opts_env).
    Sonst:
      - ODBC 18: Encrypt=yes;TrustServerCertificate=yes (praktisch für On-Prem)
      - ODBC 17: Encrypt=no
      - SQL Server (alt): keine Defaults
    """
    if opts_env:
        opts_env = opts_env.strip()
        return opts_env if opts_env.endswith(";") else opts_env + ";"
    d = (driver or "").lower()
    if "odbc driver 18" in d:
        return "Encrypt=yes;TrustServerCertificate=yes;"
    if "odbc driver 17" in d:
        return "Encrypt=no;"
    return ""  # alter "SQL Server"-Treiber

def _mk_engine(driver, server, database, uid, pwd, extra_opts_env=None):
    opts = _opts_for_driver(driver, extra_opts_env)
    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={uid};PWD={pwd};"
        f"{opts}"
    )
    params = urllib.parse.quote_plus(conn_str)
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        pool_pre_ping=True,
        future=True,
        # fast_executemany nur auf Connection-Ebene relevant – kann beim to_sql via .execution_options gesetzt werden
    )

def get_engine():
    env = os.getenv("ENV", "prod").lower()
    driver  = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")
    server  = os.getenv("DB_SERVER_PROD") if env == "prod" else os.getenv("DB_SERVER_TEST")
    database= os.getenv("DB_DATABASE")
    uid     = os.getenv("DB_UID")
    pwd     = os.getenv("DB_PWD_PROD") if env == "prod" else os.getenv("DB_PWD_TEST")
    extra   = os.getenv("DB_OPTS")  # z.B. "Encrypt=yes;TrustServerCertificate=yes"
    return _mk_engine(driver, server, database, uid, pwd, extra)

def get_engine_source():
    env = os.getenv("ENV", "prod").lower()
    driver  = os.getenv("DB_DRIVER_SOURCE", os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server"))
    server  = os.getenv("DB_SERVER_PROD_SOURCE") if env == "prod" else os.getenv("DB_SERVER_TEST_SOURCE")
    database= os.getenv("DB_DATABASE_SOURCE")
    uid     = os.getenv("DB_UID_SOURCE", os.getenv("DB_UID"))
    pwd     = os.getenv("DB_PWD_PROD_SOURCE") if env == "prod" else os.getenv("DB_PWD_TEST_SOURCE")
    extra   = os.getenv("DB_OPTS_SOURCE", os.getenv("DB_OPTS"))
    return _mk_engine(driver, server, database, uid, pwd, extra)

# Aliase, falls in Scripts erwartet:
def get_engine_fa():   return get_engine()
def get_engine_prov(): return get_engine_source()
