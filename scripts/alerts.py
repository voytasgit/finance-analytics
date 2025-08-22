# alerts.py (angepasst)
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import text
from db_connection import get_engine


engine = get_engine()
heute = pd.Timestamp(datetime.today().date())

# ----------------------------------------
# 📥 1) Daten laden (+ optional InvestScore)
# ----------------------------------------
agg = pd.read_sql(
    text("SELECT * FROM CD_FA_AGG_KENNZAHLEN WHERE Score_Datum = :d"),
    engine, params={"d": heute.date()}, parse_dates=["Score_Datum"]
).dropna(subset=["Depotnummer"]).copy()

# Typisieren & Defaults
num_cols = [
    "Cashquote_true","Cash_abs","Depotwert","Bestandwert",
    "Last_Buy_Days","Last_Trade_Days","Last_Cashflow_Days",
    "Einzahlung_30d","Einzahlung_30d_eff","Nettoflow_30d",
    "Orderfrequenz_30d","Orderfrequenz_90d","Activity_Months_6m",
    "Wertlos_Flag","Sparplan_Klein","DeepLossShare_50","DIY_FrequentSmall"
]
for c in num_cols:
    if c in agg.columns:
        agg[c] = pd.to_numeric(agg[c], errors="coerce")

agg["Cashquote_true"] = agg.get("Cashquote_true", 0).clip(0, 1).fillna(0)
agg["Cash_abs"]       = agg.get("Cash_abs", 0).fillna(0.0)
agg["Depotwert"]      = agg.get("Depotwert", 0).fillna(0.0)
agg["Last_Buy_Days"]  = agg.get("Last_Buy_Days", np.nan).fillna(9999).astype(int)
agg["Last_Trade_Days"]= agg.get("Last_Trade_Days", np.nan).fillna(9999).astype(int)
agg["Last_Cashflow_Days"] = agg.get("Last_Cashflow_Days", np.nan).fillna(9999).astype(int)
agg["Orderfrequenz_30d"]  = agg.get("Orderfrequenz_30d", 0).fillna(0).astype(int)
agg["Orderfrequenz_90d"]  = agg.get("Orderfrequenz_90d", 0).fillna(0).astype(int)
agg["Activity_Months_6m"] = agg.get("Activity_Months_6m", 0).fillna(0).astype(int)
agg["Wertlos_Flag"]       = agg.get("Wertlos_Flag", 0).fillna(0).astype(int)
agg["Sparplan_Klein"]     = agg.get("Sparplan_Klein", 0).fillna(0).astype(int)
agg["Einzahlung_30d"]     = agg.get("Einzahlung_30d", 0).fillna(0.0)
agg["Einzahlung_30d_eff"] = agg.get("Einzahlung_30d_eff", 0).fillna(0.0)
agg["Nettoflow_30d"]      = agg.get("Nettoflow_30d", 0).fillna(0.0)
agg["DeepLossShare_50"]   = agg.get("DeepLossShare_50", 0).fillna(0.0)
agg["DIY_FrequentSmall"]  = agg.get("DIY_FrequentSmall", 0).fillna(0).astype(int)

# Optional: InvestScore dazuholen (wenn Tabelle existiert)
try:
    ib = pd.read_sql(
        text("SELECT Depotnummer, InvestScore, Score_Datum FROM CD_FA_INVEST_BEREITSCHAFT WHERE Score_Datum = :d"),
        engine, params={"d": heute.date()}, parse_dates=["Score_Datum"]
    )
    agg = agg.merge(ib[["Depotnummer","InvestScore"]], on="Depotnummer", how="left")
except Exception:
    agg["InvestScore"] = np.nan

# Hilfsgrößen
rel_inflow = np.where(agg["Depotwert"] > 0, agg["Einzahlung_30d_eff"] / agg["Depotwert"], 0.0)
rel_netflow= np.where(agg["Depotwert"] > 0, agg["Nettoflow_30d"] / agg["Depotwert"], 0.0)

# ----------------------------------------
# 🚨 2) Regeln (vektorisiert)
# ----------------------------------------
alerts = []

def add_alert(mask, code, text, severity="INFO"):
    if mask.any():
        df = agg.loc[mask, ["Depotnummer"]].copy()
        df["Alert_Code"]  = code
        df["Alert_Text"]  = text
        df["Severity"]    = severity
        df["Alert_Datum"] = heute.date()
        alerts.append(df)

# 2.1 Inaktivität (Trade) > 90 Tage, aber Depotwert relevant
add_alert(
    (agg["Last_Trade_Days"] > 90) & (agg["Depotwert"] >= 1000),
    "INAKTIV_90T",
    "Keine Trades seit > 90 Tagen bei relevantem Depotvolumen",
    "WARN"
)

# 2.2 Hohe Cashquote & keine Käufe (Kapazität ungenutzt)
add_alert(
    (agg["Cashquote_true"] >= 0.40) & (agg["Last_Buy_Days"] > 30) & (agg["Cash_abs"] >= 1000),
    "CASH_40_NO_BUY_30D",
    "Hohe Cashquote (≥40%), aber keine Käufe in 30 Tagen",
    "INFO"
)

# 2.3 Einzahlungen ohne Invest (frisches Geld)
add_alert(
    (agg["Einzahlung_30d_eff"] >= 5000) & (agg["Last_Buy_Days"] > 20) & (agg["Sparplan_Klein"] == 0),
    "INFLOW_NO_BUY",
    "Hohe Einzahlung (≥5.000) ohne nachfolgenden Kauf",
    "WARN"
)
# Relativ zum Depotwert (z. B. ≥5%)
add_alert(
    (rel_inflow >= 0.05) & (agg["Last_Buy_Days"] > 20) & (agg["Sparplan_Klein"] == 0),
    "INFLOW_NO_BUY_REL",
    "Einzahlung ≥5% des Depotwerts ohne nachfolgenden Kauf",
    "WARN"
)

# 2.4 Nettoabflüsse (Abwanderungsrisiko)
add_alert(
    (agg["Nettoflow_30d"] <= -2000) | (rel_netflow <= -0.05),
    "NETTO_ABFLUSS_30D",
    "Signifikanter Nettoabfluss in 30 Tagen",
    "WARN"
)

# 2.5 Pleite-/Bereinigungsbedarf: wertlose Positionen + Cash vorhanden
add_alert(
    (agg["Wertlos_Flag"] == 1) & (agg["Cash_abs"] >= 500),
    "WERTLOS_BEREINIGEN",
    "Wertlose Position(en) vorhanden; Cash verfügbar – Depot bereinigen?",
    "INFO"
)

# 2.6 „Buy-the-dip“-Hinweis: tiefe Verluste, aber Aktivität/Zufluss
add_alert(
    (agg["DeepLossShare_50"] >= 0.30) & ((agg["Einzahlung_30d_eff"] > 0) | (agg["Last_Buy_Days"] <= 14)),
    "DEEPLOSS_ACTIVE",
    "Tiefe Verluste, aber zuletzt aktiv/Zufluss – gezielte Idee anbieten",
    "INFO"
)

# 2.7 Re-Aktivierung: lange inaktiv, jetzt frischer Cashflow
add_alert(
    (agg["Activity_Months_6m"] == 0) & (agg["Last_Cashflow_Days"] <= 7),
    "REACTIVATED_CASHFLOW",
    "Nach Inaktivität wieder Cashflow in ≤7 Tagen",
    "INFO"
)

# 2.8 Score-basierte Priorisierung (optional)
add_alert(
    (agg["InvestScore"] >= 16) & (agg["Cashquote_true"] >= 0.25),
    "HOT_LEAD_SCORE",
    "Hohe Investitionsbereitschaft laut Score",
    "CRIT"
)

# ----------------------------------------
# 💾 3) In Tabelle schreiben (idempotent pro Tag)
# ----------------------------------------
if alerts:
    alerts_df = pd.concat(alerts, ignore_index=True).drop_duplicates(
        subset=["Depotnummer","Alert_Code","Alert_Datum"]
    )
    alerts_df["Status"] = "offen"

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM CD_FA_ALERTS WHERE Alert_Datum = :d"), {"d": heute.date()})

    alerts_df.to_sql("CD_FA_ALERTS", engine, if_exists="append", index=False)
    print(f"- {len(alerts_df)} Alerts gespeichert.")
else:
    print("- Keine neuen Alerts gefunden.")
