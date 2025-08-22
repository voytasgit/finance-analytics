# segmentierung.py (aktualisiert)
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime
from db_connection import get_engine


engine = get_engine()
heute = pd.Timestamp(datetime.today().date())

# -------------------------------
# 1) Daten laden (heute)
# -------------------------------
df = pd.read_sql(
    text("SELECT * FROM CD_FA_AGG_KENNZAHLEN WHERE Score_Datum = :heute"),
    engine,
    params={"heute": heute.date()},
    parse_dates=["Score_Datum"]
).dropna(subset=["Depotnummer"])

# -------------------------------
# 2) Helper für robustes Casting
# -------------------------------
def num(df, col, default=0.0, as_int=False, clip=None):
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series(default, index=df.index)
    s = s.fillna(default)
    if clip is not None:
        s = s.clip(*clip)
    if as_int:
        s = s.astype(int)
    return s

# Kernfelder (bestehend)
df["Cashquote"]           = num(df, "Cashquote", default=0.0, clip=(0,1))
df["Orderfrequenz_30d"]   = num(df, "Orderfrequenz_30d", default=0, as_int=True)
df["Orderfrequenz_90d"]   = num(df, "Orderfrequenz_90d", default=0, as_int=True)
df["ØOrdervolumen_90d"]   = num(df, "ØOrdervolumen_90d", default=0.0)
df["TechAffin"]           = num(df, "TechAffin", default=0, as_int=True)
df["Last_Buy_Days"]       = num(df, "Last_Buy_Days", default=9999, as_int=True)
df["Last_Trade_Days"]     = num(df, "Last_Trade_Days", default=9999, as_int=True)
df["Einzahlung_30d_eff"]  = num(df, "Einzahlung_30d_eff", default=0.0)
df["Einzahlung_90d"]      = num(df, "Einzahlung_90d", default=0.0)

# NEUE Felder (robust, falls (noch) nicht vorhanden)
df["Cashquote_true"] = num(df, "Cashquote_true", default=np.nan)
df["Depotwert"]              = num(df, "Depotwert", default=0.0)
df["Cash_abs"]               = num(df, "Cash_abs", default=0.0)
df["Pos_Count"]              = num(df, "Pos_Count", default=0, as_int=True)
df["ZeroPos"]                = num(df, "ZeroPos", default=0, as_int=True)
df["Wertlos_Flag"]           = num(df, "Wertlos_Flag", default=0, as_int=True)
df["HHI_Concentration"]      = num(df, "HHI_Concentration", default=0.0, clip=(0,1))
df["DeepLossShare_50"]       = num(df, "DeepLossShare_50", default=0.0, clip=(0,1))
df["WinLoss_Skew"]           = num(df, "WinLoss_Skew", default=0.0)
df["TechWeight"]             = num(df, "TechWeight", default=0.0, clip=(0,1))
df["Activity_Months_6m"]     = num(df, "Activity_Months_6m", default=0, as_int=True)
df["Streak_Inactive_Months"] = num(df, "Streak_Inactive_Months", default=0, as_int=True)
df["DIY_FrequentSmall"]      = num(df, "DIY_FrequentSmall", default=0, as_int=True)
df["Deposit_Volatility_6m"]  = num(df, "Deposit_Volatility_6m", default=0.0)
df["Sparplan_Strength_6m"]   = num(df, "Sparplan_Strength_6m", default=0, as_int=True)
df["Last_Cashflow_Days"]     = num(df, "Last_Cashflow_Days", default=9999, as_int=True)

## Cashquote ggf. aus Cash_abs/Depotwert herleiten (fallback)
#mask_need_cash = (df["Depotwert"] > 0) & (df["Cashquote"] == 0)
#df.loc[mask_need_cash, "Cashquote"] = (df.loc[mask_need_cash, "Cash_abs"] / df.loc[mask_need_cash, "Depotwert"]).clip(0,1)

# Effektive Cashquote: bevorzugt 'true', sonst aus Cash_abs/Depotwert
df["Cashquote_eff"] = df["Cashquote_true"]
mask_need = df["Cashquote_eff"].isna() & (df["Depotwert"] > 0)
df.loc[mask_need, "Cashquote_eff"] = (df.loc[mask_need, "Cash_abs"] / df.loc[mask_need, "Depotwert"]).clip(0, 1)
df["Cashquote_eff"] = df["Cashquote_eff"].fillna(0).clip(0, 1)

# -------------------------------
# 3) dynamische Schwellen
# -------------------------------
s = df["ØOrdervolumen_90d"].astype(float)
s = s[s > 0]
vol_cut_90 = float(s.quantile(0.80)) if not s.empty else 5000.0
print(f"vol_cut_90 (80%-Quantil): {vol_cut_90:.2f}")

# Vermögensschwellen (dynamisch + Mindestfix)
dep = df["Depotwert"].astype(float)
dep_pos = dep[dep > 0]
wealth_cut = float(dep_pos.quantile(0.75)) if not dep_pos.empty else 25000.0
wealth_cut = max(wealth_cut, 25000.0)  # Mindestsinnhaftigkeit

# Zufluss-Relationen
df["rel_inflow_30d"] = np.where(df["Depotwert"] > 0, df["Einzahlung_30d_eff"] / df["Depotwert"], 0.0)
# Nettoflow kannst du bei Bedarf auch ziehen, falls im AGG vorhanden:
df["Nettoflow_30d"] = num(df, "Nettoflow_30d", default=0.0)
df["rel_netto_30d"] = np.where(df["Depotwert"] > 0, df["Nettoflow_30d"] / df["Depotwert"], 0.0)

print(f"wealth_cut (ca. P75, min 25k): {wealth_cut:,.0f}")


# Tuning-Konstanten
INAKT_TRADE_DAYS   = 180
INAKT_STREAK       = 6
TRADER_30D         = 8
TRADER_90D         = 10
TECH_WEIGHT_MIN    = 0.30
HHI_HIGH           = 0.50
DEEPLOSS_HIGH      = 0.30
SPARPLAN_MIN       = 3

# -------------------------------
# 4) Segment-Logik (priorisiert)
# -------------------------------
def segment_logik(row):
    # 0) Bereinigen nötig – wertlose Positionen dominieren
    if row.get("Wertlos_Flag", 0) == 1 or (row["Depotwert"] <= 1 and row["Pos_Count"] > 0):
        return "Bereinigen nötig"

    # 1) Inaktiv (klarer Winback-Fokus)
    if (row["Streak_Inactive_Months"] >= INAKT_STREAK) or \
       (row["Orderfrequenz_90d"] == 0 and row["Orderfrequenz_30d"] == 0 and
        row["Last_Trade_Days"] > INAKT_TRADE_DAYS and row["Einzahlung_90d"] == 0):
        return "Inaktiv"

    # 2) Reaktiviert – vorher inaktiv, jetzt frischer Cashflow/Einzahlung
    if (row["Activity_Months_6m"] <= 1) and (row["Last_Cashflow_Days"] <= 7 or row["Einzahlung_30d_eff"] > 0):
        return "Reaktiviert"

    # 3) Vermögend aktiv – hoher Depotwert + Momentum
    if (row["Depotwert"] >= wealth_cut) and \
       ((row["ØOrdervolumen_90d"] >= vol_cut_90 and row["Last_Trade_Days"] <= 60) or
        (row["Orderfrequenz_90d"] >= 4 and row["Last_Trade_Days"] <= 60) or
        (row["Activity_Months_6m"] >= 3)):
        return "Vermögend aktiv"

    # 4) Cash-Ready – hohe Cashquote/absoluter Cash + kein Kauf zuletzt
    if (row["Cashquote_eff"] >= 0.40 and row["Cash_abs"] >= 1000 and row["Last_Buy_Days"] > 30) or \
       (row["rel_inflow_30d"] >= 0.05 and row["Last_Buy_Days"] > 20):
        return "Cash-Ready"

    # 5) Trader – sehr hohe Frequenz aktuell/zuletzt
    if (row["Orderfrequenz_90d"] >= TRADER_90D or row["Orderfrequenz_30d"] >= TRADER_30D) and \
       (row["Last_Trade_Days"] <= 60):
        return "Trader"

    # 6) Tech-Affin – gewichtet (nicht nur Trefferzahl)
    if row["TechWeight"] >= TECH_WEIGHT_MIN and row["Activity_Months_6m"] >= 2:
        return "Tech-Affin"

    # 7) Vorsichtig – tiefe Verluste ODER viel Cash + keine Trades trotz Zufluss
    if (row["DeepLossShare_50"] >= DEEPLOSS_HIGH) or \
       (row["Cashquote_eff"] > 0.70 and row["Orderfrequenz_30d"] == 0 and row["Einzahlung_30d_eff"] > 0):
        return "Vorsichtig"

    # 8) DIY-Sparer – viele kleine Orders/Sparplan-Muster
    if (row["DIY_FrequentSmall"] == 1) or (row["Sparplan_Strength_6m"] >= SPARPLAN_MIN):
        return "DIY-Sparer"

    # 9) Rest
    return "Standard"


df["Segment"] = df.apply(segment_logik, axis=1)
df["Score_Datum"] = heute
print(df["Segment"].value_counts(dropna=False).sort_values(ascending=False))

# -------------------------------
# 5) Speichern
# -------------------------------
segments_df = df[["Depotnummer", "Segment", "Score_Datum"]].copy()

with engine.begin() as conn:
    conn.execute(text("DELETE FROM CD_FA_SEGMENTE WHERE Score_Datum = :heute"),
                 {"heute": heute.date()})
segments_df.to_sql("CD_FA_SEGMENTE", engine, if_exists="append", index=False)

print(f"V {len(segments_df)} Kundensegmente gespeichert.")
