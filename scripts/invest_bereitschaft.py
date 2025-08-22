# -*- coding: utf-8 -*-
# invest_bereitschaft.py scoring

# Feintuning (wenn du eine Ziel-Callquote willst)
# Mehr Calls nötig? Senke leicht die Kapazitätsgrenzen (z. B. Cash_abs ≥ 500 + Cashquote_true ≥ 0.20) 
# oder erhöhe Momentum-Gewicht (+1 auf freq_30/freq_90).
# Zu viele Calls? Erhöhe inflow_strength von 0.02 → 0.03 oder setze last_buy_days Bonus 5 → 4.


import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from sqlalchemy import create_engine, text, String, Integer, Numeric, Date
import urllib
from db_connection import get_engine


engine = get_engine()

# ------------------------------------------
# 📥 2. Daten einlesen
# ------------------------------------------

heute = pd.Timestamp(datetime.today().date())

# ------------------------------------------
# 📥 2. Aggregierte Daten laden
# ------------------------------------------
heute = pd.Timestamp(datetime.today().date())

query = """
SELECT *
FROM CD_FA_AGG_KENNZAHLEN
WHERE Score_Datum = :heute
"""

agg = pd.read_sql(text(query), engine, params={"heute": heute.date()})


# --- Zusätzliche Felder aus AGG typisieren (falls vorhanden)
num = ["Cashquote_true","Cash_abs","Bestandwert","Wertlos_Flag","ZeroPos",
       "DeepLossShare_50","WinLoss_Skew","Inflow_Strength_30d","Nettoflow_30d",
       "Last_Cashflow_Days","Activity_Months_6m","Orderfrequenz_30d","Orderfrequenz_90d",
       "Last_Buy_Days","Einzahlung_30d_eff"]
for c in num:
    if c in agg.columns:
        agg[c] = pd.to_numeric(agg.get(c), errors="coerce")

# Mapping & Typen
agg["Cashquote_true"]     = pd.to_numeric(agg.get("Cashquote_true"), errors="coerce").fillna(0).clip(0,1)
agg["Cashquote"]          = agg["Cashquote_true"]  # Abwärtskompatibilität für IB-Export
agg["Cash_abs"]           = pd.to_numeric(agg.get("Cash_abs"), errors="coerce").fillna(0.0)
agg["Einzahlung_30d_eff"] = pd.to_numeric(agg.get("Einzahlung_30d_eff"), errors="coerce").fillna(0.0)
agg["Inflow_Strength_30d"]= pd.to_numeric(agg.get("Inflow_Strength_30d"), errors="coerce").fillna(0.0)
agg["Last_Buy_Days"]      = pd.to_numeric(agg.get("Last_Buy_Days"), errors="coerce").fillna(9999).astype(int)
agg["Orderfrequenz_30d"]  = pd.to_numeric(agg.get("Orderfrequenz_30d"), errors="coerce").fillna(0).astype(int)
agg["Orderfrequenz_90d"]  = pd.to_numeric(agg.get("Orderfrequenz_90d"), errors="coerce").fillna(0).astype(int)
agg["Wertlos_Flag"]       = pd.to_numeric(agg.get("Wertlos_Flag"), errors="coerce").fillna(0).astype(int)
agg["DeepLossShare_50"]   = pd.to_numeric(agg.get("DeepLossShare_50"), errors="coerce").fillna(0.0)
agg["WinLoss_Skew"]       = pd.to_numeric(agg.get("WinLoss_Skew"), errors="coerce").fillna(0.0)
agg["Activity_Months_6m"] = pd.to_numeric(agg.get("Activity_Months_6m"), errors="coerce").fillna(0).astype(int)
agg["Last_Cashflow_Days"] = pd.to_numeric(agg.get("Last_Cashflow_Days"), errors="coerce").fillna(9999).astype(int)
agg["Nettoflow_30d"]      = pd.to_numeric(agg.get("Nettoflow_30d"), errors="coerce").fillna(0.0)


def invest_score(row):
    # --- robuste Getter (NaN -> default) ---
    def g(key, default=0.0):
        v = row.get(key, default)
        try:
            return float(v) if v is not None and not pd.isna(v) else float(default)
        except Exception:
            return float(default)

    cash_abs            = g("Cash_abs", 0)
    cash_true           = g("Cashquote_true", g("Cashquote", 0))
    inflow30_eff        = g("Einzahlung_30d_eff", 0)
    inflow_strength30   = g("Inflow_Strength_30d", 0)
    last_buy_days       = g("Last_Buy_Days", 9_999)
    last_trade_days     = g("Last_Trade_Days", 9_999)
    freq_30             = g("Orderfrequenz_30d", 0)
    freq_90             = g("Orderfrequenz_90d", 0)
    tech_affin          = g("TechAffin", 0)
    wertlos_flag        = g("Wertlos_Flag", 0)
    deep_loss_share     = g("DeepLossShare_50", 0)
    winloss_skew        = g("WinLoss_Skew", 0)
    act_m6              = g("Activity_Months_6m", 0)
    last_cf_days        = g("Last_Cashflow_Days", 9_999)
    inflow90            = g("Einzahlung_90d", 0)
    nettoflow_30d       = g("Nettoflow_30d", 0)

    score = 0

    # 1) Kapazität (Cash + frisches Geld)
    if cash_abs >= 1000 and cash_true >= 0.25: score += 2
    if cash_abs >= 3000 and cash_true >= 0.50: score += 2
    if inflow30_eff > 0:                        score += 3
    if inflow_strength30 >= 0.02:               score += 2   # ~2% Depotwert in 30d

    # 2) Engagement / Momentum
    if last_buy_days <= 14:   score += 5
    elif last_buy_days <= 30: score += 2

    if freq_30 >= 2: score += 3
    if freq_90 >= 4: score += 2

    # 3) Präferenz
    if tech_affin >= 1: score += 1

    # 4) Malus (einmalig berechnen, dann abziehen)
    malus = 0
    # „ewig inaktiv & ohne Zufluss/Cash“
    if act_m6 == 0 and last_trade_days > 180 and inflow90 == 0 and cash_abs < 100:
        malus += 6
    # wertlose Positionen
    if wertlos_flag >= 1:
        malus += 5
    # tiefe Verluste / Schieflage
    if deep_loss_share >= 0.30:
        malus += 3
    if winloss_skew <= -0.40:
        malus += 2
    # anhaltende Passivität
    if act_m6 == 0 and last_cf_days > 60:
        malus += 4
    # negativer Nettoflow zuletzt
    if nettoflow_30d < 0:
        malus += 2

    # 5) Reaktivierungs-Entlastung (wenn trotz Risiken wieder „Leben“ drin ist)
    if inflow30_eff >= 500 or last_buy_days <= 14:
        # mildert v. a. wertlos/altlasten
        malus = max(0, malus - 3)

    score = max(0, score - malus)

    # 6) Soft-Caps (optional, um Ausreißer zu begrenzen)
    if wertlos_flag >= 1 and act_m6 == 0:
        score = min(score, 12)

    # Final clamp (0..20)
    return int(min(20, max(0, round(score))))

agg["InvestScore"] = agg.apply(invest_score, axis=1)



agg["Score_Datum"] = heute

# ------------------------------------------
# 💾 5. Speichern
# ------------------------------------------

sql_dtypes = {
    "Depotnummer": String(50),
    "Cashquote": Numeric(10, 6),
    "Einzahlung_30d": Numeric(18, 2),
    "Inaktiv_seit_Kauf": Integer(),
    "Orderfrequenz": Integer(),
    "TechAffin": Integer(),
    "ØOrdervolumen": Numeric(18, 2),
    "Letzter_Kauf": Date(),
    "InvestScore": Integer(),
    "Score_Datum": Date()
}
# sql_dtypes.update({"Einzahlung_30d_eff": Numeric(18, 2)})
export_cols = [
    "Depotnummer", "Cashquote", "Einzahlung_30d", "Letzter_Kauf",
    "Inaktiv_seit_Kauf", "Orderfrequenz", "TechAffin",
    "ØOrdervolumen", "InvestScore", "Score_Datum"
]

# Export
with engine.begin() as conn:
    conn.execute(text("DELETE FROM CD_FA_INVEST_BEREITSCHAFT WHERE Score_Datum = :heute"), {"heute": heute.date()})
    #conn.execute(text("DELETE FROM CD_FA_INVEST_BEREITSCHAFT"))

#depots.to_sql("CD_FA_INVEST_BEREITSCHAFT", engine, if_exists="append", index=False, dtype=sql_dtypes)
agg[export_cols].to_sql(
    "CD_FA_INVEST_BEREITSCHAFT",
    engine,
    if_exists="append",
    index=False,
    dtype=sql_dtypes,
    method="multi",
    chunksize=200
)

print("- Investitionsbereitschaft erfolgreich berechnet und gespeichert.")
