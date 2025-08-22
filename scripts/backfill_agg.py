# -*- coding: utf-8 -*-
# backfill_agg.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date, timedelta
from backfill_update_kennzahlen import compute_agg, write_agg
from db_connection import get_engine
from db_connection import get_engine_source


# Engines: Quelle (Provisionen) und Ziel (Finance-DB)
engine_src  = get_engine_source()
engine_dest = get_engine()

def available_asof_dates(start_d, end_d):
    q = text("""
      WITH u AS (
        SELECT Datum, MAX(DateiID) AS Umax
        FROM Provisionen.dbo.cd_Datei
        WHERE DateiName LIKE 'comdirect_U-%' AND Pruefziffer > 0
          AND Datum BETWEEN :s AND :e
        GROUP BY Datum
      ),
      b AS (
        SELECT Datum, MAX(DateiID) AS Bid
        FROM Provisionen.dbo.cd_Datei
        WHERE DateiName LIKE 'comdirect_B-%' AND Pruefziffer > 0
          AND Datum BETWEEN :s AND :e
        GROUP BY Datum
      )
      SELECT u.Datum
      FROM u INNER JOIN b ON u.Datum = b.Datum
      ORDER BY u.Datum
    """)
    df = pd.read_sql(q, engine_src, params={"s": start_d, "e": end_d}, parse_dates=["Datum"])
    return [d.date() for d in df["Datum"]]

def load_hist_for_date(as_of_date):
    # Datei-IDs am Stichtag (B & U sind lt. dir immer da)
    row_u = pd.read_sql(text("""
        SELECT TOP 1 DateiID
        FROM Provisionen.dbo.cd_Datei
        WHERE Datum = :d AND DateiName LIKE 'comdirect_U-%' AND Pruefziffer > 0
        ORDER BY Datum DESC, Pruefziffer DESC
    """), engine_src, params={"d": as_of_date})
    row_b = pd.read_sql(text("""
        SELECT TOP 1 DateiID
        FROM Provisionen.dbo.cd_Datei
        WHERE Datum = :d AND DateiName LIKE 'comdirect_B-%' AND Pruefziffer > 0
        ORDER BY Datum DESC, Pruefziffer DESC
    """), engine_src, params={"d": as_of_date})

    Umax = int(row_u.iloc[0,0]); Bid = int(row_b.iloc[0,0])

    # Orders kumulativ bis Umax
    orders = pd.read_sql(
        text("SELECT * FROM Provisionen.dbo.cd_Order WHERE Datei_ID <= :U"),
        engine_src, params={"U": Umax}, parse_dates=["Orderdatum"]
    )
    # Tages-Snapshots
    saldo = pd.read_sql(
        text("SELECT * FROM Provisionen.dbo.cd_Saldo WHERE Datei_ID = :B"),
        engine_src, params={"B": Bid}, parse_dates=["Datum"]
    )
    bestand = pd.read_sql(
        text("SELECT * FROM Provisionen.dbo.cd_Bestand WHERE Datei_ID = :B"),
        engine_src, params={"B": Bid}, parse_dates=["Datum"]
    )
    # Buchungen (falls nur in Ziel-DB vorhanden): bis Stichtag
    buchungen = pd.read_sql(
        text("SELECT * FROM dbo.CD_BUCHUNG WHERE Valutadatum <= :d"),
        engine_dest, params={"d": as_of_date}, parse_dates=["Valutadatum"]
    )

    # Keys/Typen vereinheitlichen
    orders["Depotnummer"]    = orders["Depotnummer"].astype(str)
    saldo["Kontoreferenz"]   = saldo["Kontoreferenz"].astype(str)
    bestand["Depotreferenz"] = bestand["Depotreferenz"].astype(str)
    return orders, saldo, bestand, buchungen

def run_backfill(months=24):
    end_d = pd.Timestamp.today().date()
    start_d = (pd.Timestamp(end_d) - pd.DateOffset(months=months)).date()

    for as_of in available_asof_dates(start_d, end_d):
        # Skip, wenn Tag schon im Ziel existiert (idempotent)
        already = pd.read_sql(text("""
            SELECT 1 FROM dbo.CD_FA_AGG_KENNZAHLEN WHERE Score_Datum = :d
        """), engine_dest, params={"d": as_of})
        if not already.empty:
            print(f"-  {as_of} existiert – übersprungen.")
            continue

        orders, saldo, bestand, buchungen = load_hist_for_date(as_of)
        agg = compute_agg(orders, saldo, bestand, buchungen, as_of)
        write_agg(agg, engine_dest, as_of)
        print(f"- Backfill geschrieben: {as_of}")

if __name__ == "__main__":
    run_backfill(months=24)
