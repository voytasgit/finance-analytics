# -*- coding: utf-8 -*-
# build_labels_buy_de000a14uww2.py
from __future__ import annotations
import argparse, sys
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import text
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from db_connection import get_engine
# --- Konstanten ---
ISIN = "DE000A14UWW2"
DEST = "dbo.CD_FA_LABELS_DE000A14UWW2"
MIN_TICKET = 1000.0

# --- DB Engines laden ---
def _get_engines():
    from db_connection import get_engine, get_engine_source
    return get_engine(), get_engine_source()

# --- Utilities ---
def _ensure_datetime(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns and not np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# ===============================
# 1) Loader (AGG + first_buy SQL)
# ===============================
def load_agg_and_first_buy(
    *,
    start_date: str,
    end_date: str,
    storno_margin_days: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lädt:
      a) AGG-Zeilen im Fenster [start..end], nur Pre_First_Target=1
      b) first_buy_date je Depotnummer (SQL: gefenstert, Storno-CTE mit Margin)
    """
    eng_fa, eng_src = _get_engines()

    start_date = pd.to_datetime(start_date).normalize().date()
    end_date   = pd.to_datetime(end_date).normalize().date()
    end_plus30 = (pd.to_datetime(end_date) + pd.Timedelta(days=30)).date()
    start_excl = pd.to_datetime(start_date).date()
    storno_from = (pd.to_datetime(start_excl) - pd.Timedelta(days=storno_margin_days)).date()
    storno_to   = (pd.to_datetime(end_plus30) + pd.Timedelta(days=storno_margin_days)).date()

    # a) AGG (nur was wir brauchen)
    q_agg = text("""
        SELECT Depotnummer, Score_Datum AS Stichtag
        FROM dbo.CD_FA_AGG_BUY_DE000A14UWW2
        WHERE Pre_First_Target = 1
          AND Score_Datum BETWEEN :start AND :end
    """)
    agg = pd.read_sql(q_agg, eng_fa, params={"start": start_date, "end": end_date}, parse_dates=["Stichtag"])

    if agg.empty:
        return agg, agg.head(0)

    # b) first_buy_date (performant: Fenster + Storno-CTE + Join)
    q_first_buy = text("""
WITH base_orders AS (
    SELECT
        o.Depotnummer,
        CAST(o.Orderdatum AS date)   AS Orderdatum,
        CAST(o.Transaktionsnummer AS nvarchar(50)) AS tx,
        ABS(COALESCE(o.Cashflow,0))  AS Cashflow_abs
    FROM Provisionen.dbo.cd_Order o
    WHERE o.ISIN = :isin
      AND o.Ordertyp = N'Ankauf'
      AND o.Orderdatum >  :start_excl
      AND o.Orderdatum <= :end_plus30
      AND ABS(COALESCE(o.Cashflow,0)) >= :min_ticket
),
storno_tx AS (
    SELECT DISTINCT CAST(s.Transaktionsnummer AS nvarchar(50)) AS tx
    FROM Provisionen.dbo.cd_Order s
    WHERE s.Orderdatum BETWEEN :storno_from AND :storno_to
      AND s.Bemerkung LIKE N'%storn%'
      AND s.Transaktionsnummer IS NOT NULL
)
SELECT b.Depotnummer, MIN(b.Orderdatum) AS first_buy_date
FROM base_orders b
LEFT JOIN storno_tx st ON b.tx = st.tx
WHERE st.tx IS NULL
GROUP BY b.Depotnummer
""")
    first_buy = pd.read_sql(
        q_first_buy, eng_src,
        params={
            "isin": ISIN,
            "min_ticket": MIN_TICKET,
            "start_excl": start_excl,
            "end_plus30": end_plus30,
            "storno_from": storno_from,
            "storno_to":   storno_to
        },
        parse_dates=["first_buy_date"]
    )

    return agg, first_buy

# ==================
# 2) Labeling
# ==================
def label_from_first_buy(
    agg: pd.DataFrame,
    first_buy: pd.DataFrame
) -> pd.DataFrame:
    df = agg.merge(first_buy, on="Depotnummer", how="left")
    _ensure_datetime(df, ["Stichtag", "first_buy_date"])
    for T in (7, 14, 30):
        df[f"label_T{T}"] = (
            (df["first_buy_date"].notna()) &
            (df["first_buy_date"] > df["Stichtag"]) &
            (df["first_buy_date"] <= df["Stichtag"] + pd.Timedelta(days=T))
        ).astype("int8")
    return df[["Depotnummer","Stichtag","first_buy_date","label_T7","label_T14","label_T30"]]

# ==================
# 3) Persistenz
# ==================
def write_via_staging(out_df: pd.DataFrame, eng_fa, start_date, end_date,
                      batch_size: int = 20000):
    """
    Schnelles Staging via pyodbc.executemany in Batches.
    - Lokale #temp-Tabelle
    - DELETE Zieltabelle im Datumsintervall
    - INSERT ... SELECT mit TABLOCK
    """
    import math
    from sqlalchemy import text

    # sauber typisieren (Dates/Booleans -> für BIT und DATE)
    out = out_df.copy()
    out["Depotnummer"] = out["Depotnummer"].astype(str)
    out["Stichtag"] = pd.to_datetime(out["Stichtag"], errors="coerce").dt.date
    fb = pd.to_datetime(out.get("first_buy_date"), errors="coerce")
    out["first_buy_date"] = [d.date() if pd.notnull(d) else None for d in fb]
    for c in ["label_T7","label_T14","label_T30"]:
        out[c] = out[c].astype(bool)

    with eng_fa.begin() as sa_conn:
        # 1) Temp-Tabelle anlegen
        sa_conn.exec_driver_sql("""
            IF OBJECT_ID('tempdb..#stage_labels') IS NOT NULL DROP TABLE #stage_labels;
            CREATE TABLE #stage_labels (
                Depotnummer    VARCHAR(64) NOT NULL,
                Stichtag       DATE        NOT NULL,
                first_buy_date DATE        NULL,
                label_T7       BIT         NOT NULL,
                label_T14      BIT         NOT NULL,
                label_T30      BIT         NOT NULL
            );
        """)

        # 2) Schneller Bulk-ähnlicher Insert in Batches
        raw = sa_conn.connection            # DBAPI conn (pyodbc)
        cur = raw.cursor()
        cur.fast_executemany = True

        sql_ins = """
            INSERT INTO #stage_labels
            (Depotnummer, Stichtag, first_buy_date, label_T7, label_T14, label_T30)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        cols = ["Depotnummer","Stichtag","first_buy_date","label_T7","label_T14","label_T30"]

        n = len(out)
        steps = math.ceil(n / batch_size)
        for i in range(steps):
            chunk = out.iloc[i*batch_size:(i+1)*batch_size][cols]
            params = list(chunk.itertuples(index=False, name=None))
            if params:
                cur.executemany(sql_ins, params)
        print("-- DELETE start .")
        # 3) Zielbereich ersetzen (eine I/O-Welle statt vieler)
        sa_conn.exec_driver_sql(
            "DELETE FROM dbo.CD_FA_LABELS_DE000A14UWW2 WHERE Stichtag BETWEEN ? AND ?",
            (pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date())
        )
        # vor dem Insert
        out["first_buy_date"] = pd.to_datetime(out["first_buy_date"], errors="coerce")
        out["first_buy_date"] = out["first_buy_date"].dt.date  # NaT -> None -> SQL NULL
        sa_conn.exec_driver_sql("""
            INSERT INTO dbo.CD_FA_LABELS_DE000A14UWW2 WITH (TABLOCK)
            SELECT Depotnummer, Stichtag, first_buy_date, label_T7, label_T14, label_T30
            FROM #stage_labels;
        """)


# ==================
# 4) CLI
# ==================
def parse_args():
    p = argparse.ArgumentParser(description="Build First-Buy labels for DE000A14UWW2")
    p.add_argument("--start_date", help="YYYY-MM-DD (optional)")
    p.add_argument("--end_date", help="YYYY-MM-DD (optional)")
    p.add_argument("--storno_margin_days", type=int, default=60, help="Zeitpolster für Storno-Suche (± Tage)")
    return p.parse_args()

def _resolve_range_if_missing(start_date: Optional[str], end_date: Optional[str]) -> Tuple[str,str]:
    eng_fa, _ = _get_engines()
    if start_date and end_date:
        s = pd.to_datetime(start_date).date().isoformat()
        e = pd.to_datetime(end_date).date().isoformat()
        return s, e
    with eng_fa.begin() as conn:
        mx = conn.execute(text("SELECT MAX(Score_Datum) FROM dbo.CD_FA_AGG_BUY_DE000A14UWW2")).scalar()
        mn = conn.execute(text("SELECT MIN(Score_Datum) FROM dbo.CD_FA_AGG_BUY_DE000A14UWW2")).scalar()
    s = pd.to_datetime(start_date or mn).date().isoformat()
    e = pd.to_datetime(end_date   or mx).date().isoformat()
    if s > e:
        s, e = e, s
    return s, e

def main():
    args = parse_args()
    start_date, end_date = _resolve_range_if_missing(args.start_date, args.end_date)
    print(f"[LABELS] Baue Labels für {start_date} .. {end_date} (storno_margin_days={args.storno_margin_days})")

    agg, first_buy = load_agg_and_first_buy(
        start_date=start_date, end_date=end_date,
        storno_margin_days=args.storno_margin_days
    )
    print("-- load_agg_and_first_buy OK .")
    if agg.empty:
        print("-- Keine AGG-Zeilen im Fenster. Abbruch.")
        return
    out = label_from_first_buy(agg, first_buy)
    print(f"-- label_from_first_buy OK .{start_date} .. {end_date} ")
    eng_fa, _ = _get_engines()
    write_via_staging(out, eng_fa, start_date, end_date)
    print("-- write_via_staging OK .")
    print(f"[OK] {len(out)} Labels geschrieben nach {DEST}.")

if __name__ == "__main__":
    main()
