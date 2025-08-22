# -*- coding: utf-8 -*-

# AGG-Table Validation & Feature Audit
# ------------------------------------
# Validiert dbo.CD_FA_AGG_KENNZAHLEN (oder CSV-Export) vor ML-Training.

# Funktionen:
# - Einlesen aus DB (SQLAlchemy URI) ODER CSV
# - Eindeutigkeit (Depotnummer, Score_Datum)
# - Nullwerte, Konstanten, basic Stats
# - Bereichsprüfungen für Ratio-Features
# - Ausreißer-Screening (z-Score)
# - Hohe Korrelationen
# - (Optional) Zeitreihenplots zur Drift-Erkennung
# - (Optional) Grobe Feature-Importance (falls Label vorhanden)

# Benutzung:
# 1) Verbindung/CSV unten konfigurieren.
# 2) python agg_validation_starter.py
# 3) Report & Artefakte erscheinen im ./out Ordner.

# scripts/add_validation.py
# -*- coding: utf-8 -*-
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import text

# Projekt-Root in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connection import get_engine, get_engine_source

ROWCOUNT_TOL_PCT = 2.0     # Abweichung vs. Vortag in %
LEAD_MIN_RATE    = 5.0     # Mindest-Leads gesamt in % der AGG-Zahl
VOL_Q80_TOL_PCT  = 30.0    # Verteilungsdrift Toleranz in % (P80 Vol 90d)
SCORE_Q80_TOL_PCT= 25.0    # Verteilungsdrift Toleranz in % (P80 InvestScore)
CASH_Q80_TOL_ABS = 0.10    # absolute Toleranz (P80 Cashquote_true)

def _parse_date(dstr: str) -> pd.Timestamp:
    if dstr.lower() in ("today","heute"):
        return pd.Timestamp(datetime.today().date())
    return pd.Timestamp(dstr)

def _conn_test(engine, name: str) -> None:
    try:
        with engine.begin() as c:
            dbn = c.execute(text("SELECT DB_NAME()")).scalar()
        print(f"[OK] Verbindung {name}: {dbn}")
    except Exception as e:
        print(f"[FATAL] {name} Verbindung fehlgeschlagen: {e}")
        raise

def _load_by_date(engine, table: str, d: pd.Timestamp, datecol="Score_Datum", cols="*"):
    q = text(f"SELECT {cols} FROM {table} WHERE {datecol} = :d")
    return pd.read_sql(q, engine, params={"d": d.date()}, parse_dates=[datecol])

def _prev_date(engine, table: str, d: pd.Timestamp, datecol="Score_Datum"):
    q = text(f"SELECT MAX({datecol}) FROM {table} WHERE {datecol} < :d")
    with engine.begin() as c:
        prev = c.execute(q, {"d": d.date()}).scalar()
    return pd.Timestamp(prev) if prev else None

def _quantiles(series: pd.Series, qs=(0.5, 0.8)):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {q: np.nan for q in qs}
    qv = s.quantile(list(qs))
    return {float(k): float(v) for k, v in qv.items()}

def main():
    ap = argparse.ArgumentParser(description="QA/Validation inkl. Drift und Coverage")
    ap.add_argument("--date", default="today", help="YYYY-MM-DD oder 'today'")
    args = ap.parse_args()

    as_of = _parse_date(args.date)
    print(f"[INFO] Validation fuer {as_of.date()}")

    fa_engine = get_engine()
    _conn_test(fa_engine, "FA")

    # Heute laden
    agg = _load_by_date(fa_engine, "dbo.CD_FA_AGG_KENNZAHLEN", as_of)
    ib  = _load_by_date(fa_engine, "dbo.CD_FA_INVEST_BEREITSCHAFT", as_of)
    sc  = _load_by_date(fa_engine, "dbo.CD_FA_SCORING", as_of)
    seg = _load_by_date(fa_engine, "dbo.CD_FA_SEGMENTE", as_of)

    # Leads (falls Tabelle vorhanden)
    try:
        leads = _load_by_date(fa_engine, "dbo.CD_FA_LEADS", as_of, datecol="Score_Datum")
    except Exception:
        leads = pd.DataFrame(columns=["Depotnummer","Score_Datum","Next_Action"])

    # Basis-QC
    print(f"[QC] AGG rows: {len(agg)}, IB rows: {len(ib)}, SC rows: {len(sc)}, SEG rows: {len(seg)}")
    for col in ("Cashquote_true","Cashquote"):
        if col in agg.columns:
            s = pd.to_numeric(agg[col], errors="coerce")
            bad = ((s < 0) | (s > 1)).sum()
            print(f"[QC] {col} out-of-range: {bad}")
    for col in ("Last_Buy_Days","Last_Trade_Days","Last_Cashflow_Days"):
        if col in agg.columns:
            s = pd.to_numeric(agg[col], errors="coerce")
            neg = (s < 0).sum()
            print(f"[QC] {col} negative: {neg}")

    # Drift vs. voriger Stichtag
    prev_d = _prev_date(fa_engine, "dbo.CD_FA_AGG_KENNZAHLEN", as_of)
    notes = []
    rowcount_prev = None
    rowcount_delta_pct = None
    p50_vol = p80_vol = p50_vol_prev = p80_vol_prev = np.nan
    p50_cash = p80_cash = p50_cash_prev = p80_cash_prev = np.nan
    p50_score = p80_score = p50_score_prev = p80_score_prev = np.nan
    drift_flag = 0

    # Heutige Quantile
    if "ØOrdervolumen_90d" in agg.columns:
        q_vol = _quantiles(agg["ØOrdervolumen_90d"])
        p50_vol, p80_vol = q_vol.get(0.5, np.nan), q_vol.get(0.8, np.nan)
    if "Cashquote_true" in agg.columns:
        q_cash = _quantiles(agg["Cashquote_true"])
        p50_cash, p80_cash = q_cash.get(0.5, np.nan), q_cash.get(0.8, np.nan)
    if "InvestScore" in ib.columns:
        q_sc = _quantiles(ib["InvestScore"])
        p50_score, p80_score = q_sc.get(0.5, np.nan), q_sc.get(0.8, np.nan)

    if prev_d is not None:
        prev_agg = _load_by_date(fa_engine, "dbo.CD_FA_AGG_KENNZAHLEN", prev_d)
        rowcount_prev = len(prev_agg)
        if rowcount_prev and rowcount_prev > 0:
            rowcount_delta_pct = (len(agg) - rowcount_prev) / rowcount_prev * 100.0
            if abs(rowcount_delta_pct) > ROWCOUNT_TOL_PCT:
                drift_flag = 1
                notes.append(f"Rowcount drift {rowcount_delta_pct:.2f}% > {ROWCOUNT_TOL_PCT}%")

        # Vor-Tages-Quantile
        if "ØOrdervolumen_90d" in prev_agg.columns:
            q_vol_prev = _quantiles(prev_agg["ØOrdervolumen_90d"])
            p50_vol_prev, p80_vol_prev = q_vol_prev.get(0.5, np.nan), q_vol_prev.get(0.8, np.nan)
            if p80_vol_prev and p80_vol and p80_vol_prev > 0:
                diff_pct = abs(p80_vol - p80_vol_prev) / p80_vol_prev * 100.0
                if diff_pct > VOL_Q80_TOL_PCT:
                    drift_flag = 1
                    notes.append(f"P80 Vol90d drift {diff_pct:.1f}% > {VOL_Q80_TOL_PCT}%")

        if "Cashquote_true" in prev_agg.columns:
            q_cash_prev = _quantiles(prev_agg["Cashquote_true"])
            p50_cash_prev, p80_cash_prev = q_cash_prev.get(0.5, np.nan), q_cash_prev.get(0.8, np.nan)
            if not np.isnan(p80_cash_prev) and not np.isnan(p80_cash):
                if abs(p80_cash - p80_cash_prev) > CASH_Q80_TOL_ABS:
                    drift_flag = 1
                    notes.append(f"P80 Cashquote drift {abs(p80_cash-p80_cash_prev):.2f} > {CASH_Q80_TOL_ABS}")

        prev_ib = _load_by_date(fa_engine, "dbo.CD_FA_INVEST_BEREITSCHAFT", prev_d)
        if "InvestScore" in prev_ib.columns and not prev_ib.empty and not ib.empty:
            q_sc_prev = _quantiles(prev_ib["InvestScore"])
            p50_score_prev, p80_score_prev = q_sc_prev.get(0.5, np.nan), q_sc_prev.get(0.8, np.nan)
            if p80_score_prev and p80_score and p80_score_prev > 0:
                diff_pct = abs(p80_score - p80_score_prev) / p80_score_prev * 100.0
                if diff_pct > SCORE_Q80_TOL_PCT:
                    drift_flag = 1
                    notes.append(f"P80 InvestScore drift {diff_pct:.1f}% > {SCORE_Q80_TOL_PCT}%")

    # Lead Coverage
    leads_rows = len(leads)
    leads_call = int((leads["Next_Action"] == "Call").sum()) if "Next_Action" in leads.columns else 0
    leads_email = int((leads["Next_Action"] == "Email").sum()) if "Next_Action" in leads.columns else 0
    coverage_flag = 0
    if len(agg) > 0:
        lead_rate = leads_rows / len(agg) * 100.0
        if lead_rate < LEAD_MIN_RATE:
            coverage_flag = 1
            notes.append(f"Lead coverage {lead_rate:.2f}% < {LEAD_MIN_RATE}%")

    # Zusammenfassung
    print(f"[QC] Leads rows: {leads_rows} (Call={leads_call}, Email={leads_email})")
    if rowcount_delta_pct is not None:
        print(f"[QC] Rowcount delta pct vs prev: {rowcount_delta_pct:.2f}%")
    print(f"[QC] P50/P80 Vol90d: {p50_vol:.2f}/{p80_vol:.2f}")
    print(f"[QC] P50/P80 Cashquote: {p50_cash:.4f}/{p80_cash:.4f}")
    print(f"[QC] P50/P80 InvestScore: {p50_score:.2f}/{p80_score:.2f}")
    if prev_d is not None:
        print(f"[QC] Prev date: {prev_d.date()}")

    # In Log-Tabelle schreiben (idempotent fuer den Tag)
    row = pd.DataFrame([{
        "Log_Datum": as_of.date(),
        "Agg_Rows": len(agg),
        "IB_Rows": len(ib),
        "SC_Rows": len(sc),
        "SEG_Rows": len(seg),
        "Leads_Rows": leads_rows,
        "Leads_Call": leads_call,
        "Leads_Email": leads_email,
        "Rowcount_Prev": rowcount_prev,
        "Rowcount_Delta_Pct": rowcount_delta_pct,
        "P50_Vol90d": p50_vol, "P80_Vol90d": p80_vol,
        "P50_Vol90d_Prev": p50_vol_prev, "P80_Vol90d_Prev": p80_vol_prev,
        "P50_Cashquote": p50_cash, "P80_Cashquote": p80_cash,
        "P50_Cashquote_Prev": p50_cash_prev, "P80_Cashquote_Prev": p80_cash_prev,
        "P50_InvestScore": p50_score, "P80_InvestScore": p80_score,
        "P50_InvestScore_Prev": p50_score_prev, "P80_InvestScore_Prev": p80_score_prev,
        "Drift_Flag": int(drift_flag),
        "Coverage_Flag": int(coverage_flag),
        "Notes": "; ".join(notes)[:3800] if notes else None,
    }])

    with fa_engine.begin() as conn:
        conn.execute(text("DELETE FROM dbo.CD_FA_VALIDATION_LOG WHERE Log_Datum = :d"),
                     {"d": as_of.date()})
    row.to_sql("CD_FA_VALIDATION_LOG", fa_engine, if_exists="append", index=False)

    status = "OK"
    if drift_flag or coverage_flag:
        status = "WARN"
    print(f"[{status}] Validation abgeschlossen.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[FATAL] Unerwarteter Fehler: {e}")
        sys.exit(2)
