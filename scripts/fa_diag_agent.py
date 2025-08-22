# -*- coding: utf-8 -*-
# fa_diag_agent.py
import os, sys, json, textwrap, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
# Eure Connectoren verwenden:
from db_connection import get_engine_source, get_engine


# ---- Einstellungen ----
AS_OF = os.getenv("FA_AS_OF", "2025-08-12")            # z.B. "2025-08-12"
DEPOT_TEST = os.getenv("FA_DEPOT", "118742221500")       # optional: einzelnes Depot für Stichprobe
OUTDIR = os.getenv("FA_OUTDIR", ".")         # wohin Reports/CSVs geschrieben werden

# exakt die Spalten, die AGG in eurer DB aktuell hat
AGG_EXPECTED_COLS = [
    "Depotnummer","Score_Datum","Letzter_Kauf","Inaktiv_seit_Kauf","Last_Buy_Days","Last_Trade_Days",
    "Einzahlung_30d","Einzahlung_90d","Sparplan_Klein","Einzahlung_30d_eff",
    "Auszahlung_30d","Auszahlung_90d","Nettoflow_30d","Nettoflow_90d",
    "Sparplan_Strength_6m","Deposit_Volatility_6m","Last_Cashflow_Days",
    "Orderfrequenz","Orderfrequenz_30d","Orderfrequenz_90d",
    "ØOrdervolumen","ØOrdervolumen_30d","ØOrdervolumen_90d",
    "TechAffin","Cash_abs","Bestandwert","Depotwert","Pos_Count","ZeroPos","Wertlos_Flag",
    "HHI_Concentration","DeepLossShare_50","WinLoss_Skew","TechWeight",
    "Cashquote","Aktienquote","Activity_Months_6m","Streak_Inactive_Months","DIY_FrequentSmall",
    # Falls ihr "Inflow_Strength_30d" doch verwendet, HIER hinzufügen
]

def today_str():
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def fetch_df(conn, sql, **params):
    return pd.read_sql(text(sql), conn, params=params)

def classify_buchungtyp(s: str) -> str:
    t = (s or "").lower()
    if ("einlage" in t) or ("eingang" in t) or ("überweisung" in t) or ("ueberweisung" in t):
        return "in"
    if ("entnahme" in t) or ("ausgang" in t) or ("auszahlung" in t):
        return "out"
    if ("zins" in t) or ("gebühr" in t) or ("gebuehr" in t) or ("entgelt" in t) or ("spesen" in t):
        return "fee_out"
    if ("steuer" in t) or ("steuerr" in t):
        return "tax_in"
    if ("ertrag" in t) or (t.startswith("div")):
        return "yield_in"
    return "ignore"

def main():
    ensure_outdir()
    eng_src = get_engine_source()
    eng_fa  = get_engine()

    with eng_src.begin() as csrc, eng_fa.begin() as cfa:
        # 1) Letztes Datum ermitteln (oder AS_OF verwenden)
        if AS_OF:
            as_of = pd.Timestamp(AS_OF).normalize()
        else:
            maxd = csrc.execute(text("SELECT MAX(Datum) FROM Provisionen.dbo.cd_Datei WHERE Pruefziffer>0")).scalar()
            if not maxd:
                print("Kein Datum in Quelle gefunden."); sys.exit(2)
            as_of = pd.Timestamp(maxd).normalize()
        print(f"- Prüfe Stichtag: {as_of.date()}")

        # 2) Schema-Check AGG
        agg_cols = fetch_df(cfa, """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME='CD_FA_AGG_KENNZAHLEN'
            ORDER BY ORDINAL_POSITION
        """)
        cols_db = agg_cols["COLUMN_NAME"].tolist()
        missing = [c for c in AGG_EXPECTED_COLS if c not in cols_db]
        extra   = [c for c in cols_db if c not in AGG_EXPECTED_COLS]
        print("-- Schema CD_FA_AGG_KENNZAHLEN:")
        print("  - fehlend:", missing)
        print("  - zusätzlich:", extra)

        # 3) Daten ziehen
        agg = fetch_df(cfa, """
            SELECT * FROM dbo.CD_FA_AGG_KENNZAHLEN WHERE Score_Datum=:d
        """, d=as_of.date())

        # Depotwahl
        depot = DEPOT_TEST or (agg["Depotnummer"].astype(str).iloc[0] if not agg.empty else "")
        if not depot:
            print("Keine Depotnummer gefunden."); sys.exit(3)
        print("-- Test-Depot:", depot)

        # 4) Rohdaten (Buchung inkl. Typ/Bemerkung)
        # versuche beide Varianten
        try:
            b = fetch_df(csrc, """
                SELECT Kontoreferenz, Betrag, Valutadatum, Buchungtyp, Bemerkung
                FROM Provisionen.dbo.cd_Buchung
                WHERE Valutadatum <= :d
            """, d=as_of.date())
        except Exception:
            b = fetch_df(csrc, """
                SELECT Kontoreferenz, Betrag, Valutadatum, Buchungstyp AS Buchungtyp, Bemerkung
                FROM Provisionen.dbo.cd_Buchung
                WHERE Valutadatum <= :d
            """, d=as_of.date())

        # Saldo und Bestand (für Cashquote)
        saldo = fetch_df(csrc, """
            SELECT * FROM Provisionen.dbo.cd_Saldo WHERE Datum=:d
        """, d=as_of.date())
        best  = fetch_df(csrc, """
            SELECT * FROM Provisionen.dbo.cd_Bestand WHERE Datum=:d
        """, d=as_of.date())

        # 5) Flows klassifizieren (Python-Seite)
        b["Valutadatum"] = pd.to_datetime(b["Valutadatum"], errors="coerce")
        b["Betrag"] = pd.to_numeric(b["Betrag"], errors="coerce").fillna(0.0)
        b["klasse"] = b["Buchungtyp"].fillna("").map(classify_buchungtyp)
        start_30 = as_of - pd.Timedelta(days=30)
        start_90 = as_of - pd.Timedelta(days=90)
        b30 = b[b["Valutadatum"]>=start_30]
        b90 = b[b["Valutadatum"]>=start_90]
        inflow_classes = {"in"}                # ggf {"in","tax_in"}
        outflow_classes= {"out","fee_out"}

        inflow_30  = b30[b30["klasse"].isin(inflow_classes)].groupby("Kontoreferenz")["Betrag"].sum()
        outflow_30 = b30[b30["klasse"].isin(outflow_classes)].groupby("Kontoreferenz") ["Betrag"].sum()
        inflow_90  = b90[b90["klasse"].isin(inflow_classes)].groupby("Kontoreferenz")["Betrag"].sum()
        outflow_90 = b90[b90["klasse"].isin(outflow_classes)].groupby("Kontoreferenz")["Betrag"].sum()

        # Map über alle möglichen Referenzen
        # (Depot~Konto-Mapping über Depotreferenz/Kontoreferenz = identisch in euren Daten)
        depot_str = str(depot)
        inflow30_py  = float(inflow_30.get(depot_str, 0.0))
        outflow30_py = float(outflow_30.get(depot_str, 0.0))
        inflow90_py  = float(inflow_90.get(depot_str, 0.0))
        outflow90_py = float(outflow_90.get(depot_str, 0.0))
        netto30_py   = inflow30_py - outflow30_py
        netto90_py   = inflow90_py - outflow90_py

        # 6) AGG-Werte lesen
        row = agg.loc[agg["Depotnummer"].astype(str)==depot_str].copy()
        if row.empty:
            print("Depot im AGG nicht gefunden."); sys.exit(4)
        row = row.iloc[0]

        # 7) Cashquote SQL-seitig nachstellen
        # (Saldo letzter Schnappschuss / Bestandwert)
        s_ = saldo[saldo["Kontoreferenz"].astype(str)==depot_str].copy()
        best_sum = best[best["Depotreferenz"].astype(str)==depot_str]["Kurswert"].sum()
        cash_abs_sql = float(s_["Saldosollwert"].sum()) if not s_.empty else 0.0
        cashquote_sql = cash_abs_sql / (best_sum + 1e-6)

        # 8) Ergebnisse/Abweichungen
        res = {
            "Depotnummer": depot_str,
            "Score_Datum": as_of.date(),
            "Einzahlung_30d": float(row.get("Einzahlung_30d", 0.0)),
            "Auszahlung_30d": float(row.get("Auszahlung_30d", 0.0)),
            "Nettoflow_30d":  float(row.get("Nettoflow_30d", 0.0)),
            "Einzahlung_90d": float(row.get("Einzahlung_90d", 0.0)),
            "Auszahlung_90d": float(row.get("Auszahlung_90d", 0.0)),
            "Nettoflow_90d":  float(row.get("Nettoflow_90d", 0.0)),
            "Cashquote":      float(row.get("Cashquote", 0.0)),
            "inflow_30_sql":  inflow30_py,
            "outflow_30_sql": outflow30_py,
            "netto_30_sql":   netto30_py,
            "inflow_90_sql":  inflow90_py,
            "outflow_90_sql": outflow90_py,
            "netto_90_sql":   netto90_py,
            "cash_abs_sql":   cash_abs_sql,
            "cashquote_sql":  cashquote_sql,
        }
        for k in ["inflow_30","outflow_30","netto_30","inflow_90","outflow_90","netto_90"]:
            res[f"Δ_{k}"] = res.get(k.replace("_","_")+"")  # Platzhalter, wird unten gesetzt

        res["Δ_inflow_30"]  = res["inflow_30_sql"]  - res["Einzahlung_30d"]
        res["Δ_outflow_30"] = res["outflow_30_sql"] - res["Auszahlung_30d"]
        res["Δ_netto_30"]   = res["netto_30_sql"]   - res["Nettoflow_30d"]
        res["Δ_inflow_90"]  = res["inflow_90_sql"]  - res["Einzahlung_90d"]
        res["Δ_outflow_90"] = res["outflow_90_sql"] - res["Auszahlung_90d"]
        res["Δ_netto_90"]   = res["netto_90_sql"]   - res["Nettoflow_90d"]
        res["Δ_cashquote"]  = res["cashquote_sql"]  - res["Cashquote"]

        df = pd.DataFrame([res])
        outcsv = os.path.join(OUTDIR, f"fa_diag_{today_str()}.csv")
        df.to_csv(outcsv, index=False)
        print("-- Report geschrieben:", outcsv)

        # 9) Spalten-Mismatch-Report
        diff = pd.DataFrame({
            "in_db": cols_db,
            "expected?": [c in AGG_EXPECTED_COLS for c in cols_db]
        })
        diff["missing_expected"] = diff["in_db"].apply(lambda x: x in missing)
        diff.to_csv(os.path.join(OUTDIR, f"fa_schema_{today_str()}.csv"), index=False)
        print("-- Schema-Report geschrieben.")

if __name__ == "__main__":
    main()
