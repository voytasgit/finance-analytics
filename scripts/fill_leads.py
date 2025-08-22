# # scripts/fill_leads.py
# Hier ist, wie du systematisch tunen solltest, mit konkreten Beispielen.
# 1) Ziel definieren
# Kapazität/Tag: z. B. 40 Calls, 400 E-Mails.
# Qualitätsziel: z. B. ≥70 % der Calls sollen „ready“ sein (genug Cash/Einzahlung/Recency).
# 2) Stellschrauben verstehen
# A. Mengensteuerung (wie viele):
# QCALL_PCT: Quantilschwelle für Calls (z. B. 0.97 = Top 3 %).
# ↓ senken ⇒ mehr Calls · ↑ erhöhen ⇒ weniger Calls.
# QEMAIL_PCT: Quantilschwelle für Mails (z. B. 0.80 = Top 20 %).
# Praktisch relevant, wenn du „None“ irgendwann ausschließen willst.
# MIN_CALLS: Mindestzahl Calls pro Tag (überstimmt QCALL_PCT, wenn die Population kalt ist).
# B. Qualitätssteuerung (wer):
# READY_CASH_ABS_MIN (absoluter Cash), READY_INFLOW_30D_MIN (Einzahlung),
# READY_LAST_BUY_DAYSMAX (Kauf in ≤ X Tagen), READY_RECENT_ACTIVITY (Trade in ≤ X Tagen).
# → strenger ⇒ höhere Qualität, aber weniger Calls (weil mehr „Call→Email“-Downgrades).
# 3) Startwerte (empfohlen)
# QCALL_PCT = 0.97, MIN_CALLS = 60 (liefert ~Top 3 % oder mind. 60 Calls).
# READY_*: wie bei dir (1000 € Cash, 500 € Einzahlung, Kauf ≤14 Tage, Trade ≤180 Tage).
# 4) Tuning nach Beobachtung
# Fall A: Zu wenige Calls
# Symptom: Calls < Kapazität (z. B. nur 25 statt 40).
# Schritt 1: Senke die Schwelle oder nutze die Ziel-Formel:
# Formel: QCALL_PCT = 1 - (desired_calls / n)
# Beispiel: 40 Calls bei n=6000 ⇒ 1 - 40/6000 = 0.9933 (Top 0,67 %).
# Lass MIN_CALLS stehen (Sicherheitsnetz).
# Schritt 2 (optional): Lockere READY_*, z. B.
# READY_CASH_ABS_MIN 1000 → 500, READY_INFLOW_30D_MIN 500 → 300,
# READY_LAST_BUY_DAYSMAX 14 → 21, READY_RECENT_ACTIVITY 180 → 270.
# Fall B: Zu viele Calls
# Schritt 1: Erhöhe QCALL_PCT (z. B. 0.97 → 0.98).
# Schritt 2: Senke MIN_CALLS (z. B. 60 → 40).
# Schritt 3: Mache READY_* strenger (Cash 1000 → 2000, Kauf ≤14 → ≤7, Trade ≤180 → ≤90).
# Fall C: Calls sind nicht „ready“ genug (Qualität zu niedrig)
# READY_* strenger stellen (wie oben).
# Optional: Segmentfilter nur für Call anwenden (z. B. nur „Cash-Ready“, „Vermögend aktiv“).
# → In fill_leads.py: df["Next_Action"]="Email" setzen, wenn Segment nicht in erlaubter Liste ist.
# Fall D: InvestScore sehr flach (viele gleiche Werte)
# MIN_CALLS sorgt zwar für Menge, aber Qualität schwankt.
# Tie-Breaker einführen: sortiere Top-Kandidaten zusätzlich nach Cash_abs (absteigend) und Last_Buy_Days (aufsteigend).
# (Kleiner Code-Patch möglich, sag Bescheid wenn du ihn willst.)
# 5) Konkrete Beispiel-Presets
# Preset „Kalt, wir brauchen mehr Calls“
# QCALL_PCT = 0.99
# MIN_CALLS = 60
# READY_CASH_ABS_MIN = 500, READY_INFLOW_30D_MIN = 300, READY_LAST_BUY_DAYSMAX = 21, READY_RECENT_ACTIVITY = 270
# Preset „Qualität vor Quantität“
# QCALL_PCT = 0.97
# MIN_CALLS = 30
# READY_CASH_ABS_MIN = 2000, READY_INFLOW_30D_MIN = 1000, READY_LAST_BUY_DAYSMAX = 7, READY_RECENT_ACTIVITY = 90
# Preset „Ziel: exakt ~40 Calls/Tag“
# Formelbasiert: bei n=6 000
# QCALL_PCT = 1 - 40/6000 = 0.9933
# MIN_CALLS = 40 (Sicherheitsnetz)
# READY_* unverändert; bei Bedarf vorsichtig anziehen.
# 6) Was täglich prüfen (aus Logs/LEADS/QC)
# Counts: #Call, #Email vs. Kapazität.
# Readiness-Rate: Anteil der Calls, die ready==True waren (vor dem Downgrade).
# P80/P95 InvestScore (nur Calls) → ob du „zu tief schneiden“ musst.
# Cash-Profil der Calls: Median/P80 von Cash_abs.
# Segment-Mix der Calls (Anteil „Cash-Ready“, „Vermögend aktiv“, …).
# 7) Kleiner Praxis-Workflow (5-Minuten-Routine)
# Pipeline laufen lassen → LEADS anschauen.
# Sind Calls < Kapazität?
# → QCALL_PCT runter oder MIN_CALLS hoch; ggf. READY_* leicht lockern.
# Sind Calls qualitativ schwach?
# → READY_* strenger; optional Segmentfilter für Calls.
# Änderungen als kleine Schritte (eine Schraube pro Tag), Effekte beobachten.
# scripts/fill_leads.py
# Purpose:
# - Build daily leads (Call vs Email) from aggregated tables.
# - Control both quantity (how many calls) and quality (who gets a call).
#
# Tuning knobs:
#   QCALL_PCT: quantile on InvestScore for Call bucket (e.g. 0.97 => top 3%)
#   MIN_CALLS: ensure at least this many Calls after readiness/segment filters
#   READY_*  : readiness gates to avoid low-quality calls
#   ENFORCE_SEGMENT_FOR_CALL / ALLOWED_CALL_SEGMENTS: optional segment filter

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
from datetime import date, datetime
from sqlalchemy import text
from db_connection import get_engine

# -------------------------
# Tuning
# -------------------------
QCALL_PCT   = 0.99 # 0.97   # top 3% by InvestScore -> Call
QEMAIL_PCT  = 0.80   # not used to cut emails yet, kept for future
MIN_CALLS   = 60     # hard minimum number of Calls per day

READY_CASH_ABS_MIN     = 1000   # EUR cash
READY_INFLOW_30D_MIN   = 300 #500    # EUR effective inflow last 30d
READY_LAST_BUY_DAYSMAX = 21 #14     # last buy in <= 14 days
READY_RECENT_ACTIVITY  = 270 # 180    # or last trade in <= 180 days if 0 active months

# Optional: restrict which segments may receive Calls
ENFORCE_SEGMENT_FOR_CALL = True
ALLOWED_CALL_SEGMENTS = {
    "Vermoegend aktiv",  # note: if your DB holds umlauts, keep original: "Vermögend aktiv"
    "Cash-Ready",
    "Trader",
    "Reaktiviert",
    "Tech-Affin"
}

def parse_date(s: str) -> date:
    s_low = s.lower()
    if s_low in ("today", "heute", "now"):
        return date.today()
    return datetime.strptime(s, "%Y-%m-%d").date()

def main(d: date):
    eng = get_engine()

    # Load minimal columns needed
    agg = pd.read_sql(text("""
        SELECT Depotnummer, Score_Datum,
               Cashquote_true, Cash_abs, Depotwert,
               Einzahlung_30d_eff, Orderfrequenz_30d, Orderfrequenz_90d,
               Last_Buy_Days, Last_Trade_Days, Activity_Months_6m,
               Wertlos_Flag, Nettoflow_30d
        FROM dbo.CD_FA_AGG_KENNZAHLEN
        WHERE Score_Datum = :d
    """), eng, params={"d": d})

    ib = pd.read_sql(text("""
        SELECT Depotnummer, Score_Datum, InvestScore
        FROM dbo.CD_FA_INVEST_BEREITSCHAFT
        WHERE Score_Datum = :d
    """), eng, params={"d": d})

    sc = pd.read_sql(text("""
        SELECT Depotnummer, Score_Datum, Score AS GeneralScore
        FROM dbo.CD_FA_SCORING
        WHERE Score_Datum = :d
    """), eng, params={"d": d})

    seg = pd.read_sql(text("""
        SELECT Depotnummer, Score_Datum, Segment
        FROM dbo.CD_FA_SEGMENTE
        WHERE Score_Datum = :d
    """), eng, params={"d": d})

    # Join
    df = agg.merge(ib, on=["Depotnummer","Score_Datum"], how="left") \
            .merge(sc, on=["Depotnummer","Score_Datum"], how="left") \
            .merge(seg, on=["Depotnummer","Score_Datum"], how="left")

    if df.empty:
        print(f"[LEADS] WARN: no data for {d}.")
        return

    # Robust numerics
    def num(col, clip=None):
        s = pd.to_numeric(df.get(col), errors="coerce").fillna(0)
        if clip is not None:
            lo, hi = clip
            s = s.clip(lo, hi)
        return s

    invest = num("InvestScore")
    gen    = num("GeneralScore")
    cash   = num("Cash_abs")
    inflow = num("Einzahlung_30d_eff")
    lbuy   = num("Last_Buy_Days")
    ltrade = num("Last_Trade_Days")
    act6m  = num("Activity_Months_6m")
    wertlos= num("Wertlos_Flag")
    netto30= num("Nettoflow_30d")
    cqtrue = num("Cashquote_true", clip=(0,1))

    # Readiness gate
    ready_basic    = (cash >= READY_CASH_ABS_MIN) | (inflow >= READY_INFLOW_30D_MIN) | (lbuy <= READY_LAST_BUY_DAYSMAX)
    ready_activity = (act6m > 0) | (ltrade <= READY_RECENT_ACTIVITY)
    df["ready"]    = ready_basic & ready_activity

    # Dynamic thresholds (quantiles)
    if invest.notna().any():
        q_call  = float(invest.quantile(QCALL_PCT))
        q_email = float(invest.quantile(QEMAIL_PCT))
    else:
        q_call, q_email = 999.0, 0.0

    # Ensure MIN_CALLS irrespective of flat distributions
    n = len(df)
    if n > 0 and MIN_CALLS > 0 and MIN_CALLS <= n:
        # nlargest(MIN_CALLS).min() is a lower bound for the Call threshold
        thresh_min_calls = float(invest.nlargest(MIN_CALLS).min())
        q_call = min(q_call, thresh_min_calls)

    # Initial Next_Action by quantile
    df["Next_Action"] = np.where(invest >= q_call, "Call", "Email")

    # Enforce readiness: non-ready Calls -> Email
    df.loc[(df["Next_Action"] == "Call") & (~df["ready"]), "Next_Action"] = "Email"

    # Optional: enforce segment allowlist for Calls
    if ENFORCE_SEGMENT_FOR_CALL:
        seg_series = df.get("Segment").fillna("")
        bad_seg = ~seg_series.isin(ALLOWED_CALL_SEGMENTS)
        df.loc[(df["Next_Action"] == "Call") & bad_seg, "Next_Action"] = "Email"

    # Top-up to MIN_CALLS after readiness/segment downgrade
    call_count = int((df["Next_Action"] == "Call").sum())
    need = max(0, MIN_CALLS - call_count)
    if need > 0:
        cands = (df["Next_Action"] == "Email") & (df["ready"])
        if ENFORCE_SEGMENT_FOR_CALL:
            cands &= df.get("Segment").fillna("").isin(ALLOWED_CALL_SEGMENTS)

        promo = df.loc[cands].copy()
        if not promo.empty:
            # Tie-breaker: InvestScore desc, Cash_abs desc, Last_Buy_Days asc, GeneralScore desc
            promo = promo.sort_values(
                by=["InvestScore", "Cash_abs", "Last_Buy_Days", "GeneralScore"],
                ascending=[False, False, True, False]
            ).head(need)
            df.loc[promo.index, "Next_Action"] = "Call"

    # Composite score (same as before)
    comp = (
        invest * 0.6
      + gen    * 0.4
      + np.where(wertlos >= 1, -5.0, 0.0)
      + np.where(act6m == 0,    -4.0, 0.0)
    ).astype(float)

    # Prepare output
    out = pd.DataFrame({
        "Depotnummer":    df["Depotnummer"].astype(str),
        "Score_Datum":    df["Score_Datum"],
        "InvestScore":    invest.fillna(0).astype("int64"),
        "GeneralScore":   gen.fillna(0).astype("int64"),
        "CompositeScore": comp,
        "Segment":        df.get("Segment"),
        "Next_Action":    df["Next_Action"]
    })

    # Replace today's leads
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM dbo.CD_FA_LEADS WHERE Score_Datum = :d"), {"d": d})
    out.to_sql("CD_FA_LEADS", eng, if_exists="append", index=False)

    # Logging
    calls  = int((out["Next_Action"] == "Call").sum())
    emails = int((out["Next_Action"] == "Email").sum())
    # Ready rate among final calls (should be 1.0 if gates worked, but we still compute)
    final_calls_idx = out["Next_Action"] == "Call"
    ready_call_rate = float(df.loc[final_calls_idx, "ready"].mean() or 0)
    med_cash_calls  = float(df.loc[final_calls_idx, "Cash_abs"].median() or 0)
    print(f"[LEADS] {d}: total={len(out)}, Call={calls}, Email={emails}, q_call={q_call:.2f}")
    print(f"[LEADS] quality: ready_call_rate={ready_call_rate:.2%}, median_cash_calls={med_cash_calls:,.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="today")
    d = parse_date(ap.parse_args().date)
    main(d)
