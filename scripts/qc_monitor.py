# -*- coding: utf-8 -*-
# scripts/qc_monitor.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text
from datetime import datetime
from db_connection import get_engine

engine = get_engine()
heute = pd.Timestamp(datetime.today().date())

# Vorhandene Einträge zum Datum löschen
with engine.begin() as conn:
    conn.execute(
        text("DELETE FROM CD_FA_QC_MONITOR WHERE Log_Datum = :d"),
        {"d": heute.date()}
    )

# AGG und IB laden
agg = pd.read_sql(
    text("SELECT * FROM CD_FA_AGG_KENNZAHLEN WHERE Score_Datum = :d"),
    engine, params={"d": heute.date()}
)
ib  = pd.read_sql(
    text("SELECT Depotnummer, Score_Datum, InvestScore FROM CD_FA_INVEST_BEREITSCHAFT WHERE Score_Datum = :d"),
    engine, params={"d": heute.date()}
)
leads = pd.read_sql(
    text("SELECT Depotnummer, Score_Datum, Next_Action FROM CD_FA_LEADS WHERE Score_Datum = :d"),
    engine, params={"d": heute.date()}
)

# Join für Score-Statistik
df = agg.merge(ib, on=["Depotnummer","Score_Datum"], how="left")

def q(series, p):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.quantile(p)) if not s.empty else 0.0

mon = pd.DataFrame([{
    "Log_Datum": heute.date(),
    "P50_InvestScore":    q(df["InvestScore"], 0.50),
    "P80_InvestScore":    q(df["InvestScore"], 0.80),
    "P50_Cash_abs":       q(df.get("Cash_abs"), 0.50),
    "P80_Cash_abs":       q(df.get("Cash_abs"), 0.80),
    "P50_Cashquote_true": q(df.get("Cashquote_true", df.get("Cashquote")), 0.50),
    "P80_Cashquote_true": q(df.get("Cashquote_true", df.get("Cashquote")), 0.80),
    "P50_Freq90":         q(df.get("Orderfrequenz_90d"), 0.50),
    "P80_Freq90":         q(df.get("Orderfrequenz_90d"), 0.80)
}])

mon.to_sql("CD_FA_QC_MONITOR", engine, if_exists="append", index=False)

# Leads-Zusammenfassung
if not leads.empty:
    calls  = (leads["Next_Action"]=="Call").sum()
    emails = (leads["Next_Action"]=="Email").sum()
    total  = len(leads)
    print(f"[MON] Leads: total={total}, Call={calls}, Email={emails}")
else:
    print("[MON] Leads: none")
