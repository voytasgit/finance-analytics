# -*- coding: utf-8 -*-
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, String, Date, Numeric
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import urllib
import numpy as np
import re
from db_connection import get_engine, get_engine_source
from backfill_update_kennzahlen import load_inputs_for_day


engine = get_engine()
# Engine für Quelle (Provisionen)
engine_src = get_engine_source()

# ------------------------------------------
# 📥 2. Tabellen abrufen aus Provisionen.dbo
# ------------------------------------------

heute = pd.Timestamp(datetime.today().date())
as_of_date = pd.Timestamp.today().normalize()  

orders, saldo, bestand, buchungen = load_inputs_for_day(engine_src, as_of_date)

# ---- NEU: Nones → leere DFs mit Minimalspalten ----
def ensure_df(df, cols):
    if isinstance(df, pd.DataFrame):
        # fehlende Spalten anlegen (richtige Reihenfolge egal)
        for c in cols:
            if c not in df.columns:
                df[c] = pd.Series(dtype="object")
        return df
    # komplett leer anlegen
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

orders  = ensure_df(orders,  ["Depotnummer","Orderdatum","Cashflow"])
saldo   = ensure_df(saldo,   ["Kontoreferenz","Datum","Saldosollwert"])
bestand = ensure_df(bestand, ["Depotreferenz","Datum","Wertpapiername","Wertpapiertyp","Kurswert"])
buchungen = ensure_df(buchungen, ["Kontoreferenz","Valutadatum","Betrag"])

# Typisierung (schadet auch bei leerem DF nicht)
orders["Orderdatum"] = pd.to_datetime(orders["Orderdatum"], errors="coerce")
orders["Cashflow"]   = pd.to_numeric(orders["Cashflow"], errors="coerce")
saldo["Datum"]       = pd.to_datetime(saldo["Datum"], errors="coerce")
saldo["Saldosollwert"]= pd.to_numeric(saldo["Saldosollwert"], errors="coerce")
bestand["Datum"]     = pd.to_datetime(bestand["Datum"], errors="coerce")
bestand["Kurswert"]  = pd.to_numeric(bestand["Kurswert"], errors="coerce")

# Orders
if orders is not None:
    print(f"Orders: {len(orders)} Zeilen, min Datum = {orders['Orderdatum'].min()}, max Datum = {orders['Orderdatum'].max()}")

# Saldo
if saldo is not None:
    print(f"Saldo: {len(saldo)} Zeilen, min Datum = {saldo['Datum'].min()}, max Datum = {saldo['Datum'].max()}")

# Bestand
if bestand is not None:
    print(f"Bestand: {len(bestand)} Zeilen, min Datum = {bestand['Datum'].min()}, max Datum = {bestand['Datum'].max()}")

# Buchungen (optional)
if buchungen is not None:
    print(f"Buchungen: {len(buchungen)} Zeilen, min Datum = {buchungen['Valutadatum'].min()}, max Datum = {buchungen['Valutadatum'].max()}")

# ------------------------------------------
# 🧮 3. Scoring-Berechnung
# ------------------------------------------


# FIX: Bestand auf 'heute' oder letzten verfügbaren Tag davor snappen
if "Datum" in bestand.columns:
    bestand = bestand[bestand["Datum"] <= heute].copy()
    idx = bestand.groupby("Depotreferenz")["Datum"].transform("max") == bestand["Datum"]
    bestand_snapshot = bestand[idx].copy()
else:
    bestand_snapshot = bestand.copy()

orders["Monat"] = orders["Orderdatum"].dt.to_period("M")

score_df = pd.DataFrame()

alle_depots = (
    pd.Index(orders["Depotnummer"].dropna().astype(str).unique())
    .union(pd.Index(bestand_snapshot["Depotreferenz"].dropna().astype(str).unique()))
    .union(pd.Index(saldo.sort_values("Datum")
            .drop_duplicates("Kontoreferenz", keep="last")["Kontoreferenz"].dropna().astype(str).unique()))
)
score_df = pd.DataFrame({"Depotnummer": alle_depots})
score_df["Score"] = 0

# 1️⃣ Orderfrequenz
order_freq = orders.groupby("Depotnummer")["Monat"].nunique()
score_df["Orderfrequenz"] = score_df["Depotnummer"].map(order_freq)

############## score_df["Score"] += score_df["Orderfrequenz"].apply(lambda x: 10 if x > 2 else 0) ################
# 2️⃣ Ø Ordervolumen
cut30 = heute - pd.Timedelta(days=30)
cut90 = heute - pd.Timedelta(days=90)
orders["Ordervolumen"] = orders["Cashflow"].abs()

freq_30 = orders[orders["Orderdatum"] >= cut30].groupby("Depotnummer").size()
freq_90 = orders[orders["Orderdatum"] >= cut90].groupby("Depotnummer").size()
avg_vol_90 = orders[orders["Orderdatum"] >= cut90].groupby("Depotnummer")["Ordervolumen"].mean()
score_df["ØOrdervolumen"]     = score_df["Depotnummer"].map(avg_vol_90).fillna(0.0)  # ersetzt Allzeit
score_df.loc[:, "Orderfrequenz_30d"] = (
    score_df["Depotnummer"].map(freq_30).fillna(0).astype("int64")
)
score_df.loc[:, "Orderfrequenz_90d"] = (
    score_df["Depotnummer"].map(freq_90).fillna(0).astype("int64")
)

################# score_df["Score"] += score_df["ØOrdervolumen"].apply(lambda x: 10 if x > 3000 else 0) ######################

# 3️⃣ Aktienquote

# Aktiendichte/gesamt jetzt auf snapshot:
aktien_only = bestand_snapshot[bestand_snapshot["Wertpapiertyp"] == "AKTIE"]
aktien_sum  = aktien_only.groupby("Depotreferenz")["Kurswert"].sum()
gesamt_sum  = bestand_snapshot.groupby("Depotreferenz")["Kurswert"].sum()

aktienquote = aktien_sum / gesamt_sum
score_df["Aktienquote"] = score_df["Depotnummer"].map(aktienquote)
########################### score_df["Score"] += score_df["Aktienquote"].apply(lambda x: 6 if x > 0.8 else 0)

# 4️⃣ Cashquote
saldo_latest = saldo.sort_values("Datum").drop_duplicates("Kontoreferenz", keep="last")

# Cashquote: saldo_latest (wie gehabt) mit *snapshot* der Bestände
bestand_sum = bestand_snapshot.groupby("Depotreferenz")["Kurswert"].sum()
saldo_merged = saldo_latest.merge(bestand_sum, left_on="Kontoreferenz", right_on="Depotreferenz", how="left")
saldo_merged["cash_to_total"] = saldo_merged["Saldosollwert"] / (saldo_merged["Saldosollwert"] + saldo_merged["Kurswert"] + 1e-6)
cash_map = saldo_merged.set_index("Kontoreferenz")["cash_to_total"]
score_df["Cashquote"] = score_df["Depotnummer"].map(cash_map).fillna(0).clip(0, 1)

################## score_df["Score"] += score_df["Cashquote"].apply(lambda x: -6 if x > 0.3 else 0) ################################

# 5️⃣ Tech-Affinität
#bestand["TechTreffer"] = bestand["Wertpapiername"].str.contains("NVIDIA|APPLE|TESLA|AI|TECH", case=False, na=False)
TECH_NAMES = ["APPLE","MICROSOFT","AMAZON","ALPHABET","GOOGLE","META PLATFORMS","FACEBOOK",
                "TESLA","NETFLIX","NVIDIA","ADVANCED MICRO DEVICES","AMD","INTEL","BROADCOM",
                "QUALCOMM","ASML","TSMC","TAIWAN SEMICONDUCTOR","MICRON","MARVELL","NXP",
                "STMICROELECTRONICS","INFINEON","TEXAS INSTRUMENTS","ANALOG DEVICES",
                "ON SEMICONDUCTOR","ONSEMI","ADOBE","SALESFORCE","ORACLE","SERVICENOW",
                "SNOWFLAKE","DATADOG","MONGODB","ATLASSIAN","ELASTIC","INTUIT","SHOPIFY",
                "PALANTIR","CLOUDFLARE","PALO ALTO NETWORKS","CROWDSTRIKE","ZSCALER",
                "FORTINET","OKTA","SAMSUNG ELECTRONICS","TENCENT","ALIBABA","BAIDU","SAP"]

TECH_REGEX = re.compile(
    r"(?:^|\b)(?:" + "|".join(map(re.escape, TECH_NAMES)) + r")(?:\b|$)",
    flags=re.IGNORECASE
)

# Falls du das ETF-Pattern verwendest: hier auch non-capturing benutzen
ETF_REGEX = re.compile(
    r"(?i)(?=.*\betf\b)(?=.*\b(?:technology|technologie|software|cloud|semiconductor|"
    r"halbleiter|robotics|robotik|cybersecurity)\b)"
)

# 1) Sicherstellen, dass wir wirklich einen Snapshot verwenden
if "Datum" in bestand.columns:
    # Prüfen, ob 'bestand' evtl. bereits nur 1 Stichtag enthält (z.B. durch Datei_ID-Filter)
    if bestand["Datum"].nunique() == 1:
        portfolio = bestand.copy()          # ist bereits Snapshot
    else:
        # Snapshot je Depotreferenz: letzter Stand <= heute
        tmp = bestand[bestand["Datum"] <= heute].copy()
        idx = tmp.groupby("Depotreferenz")["Datum"].transform("max") == tmp["Datum"]
        portfolio = tmp[idx].copy()
else:
    portfolio = bestand.copy()              # kein Datumsfeld => als Snapshot behandeln

# 2) Tech-Maske NUR auf 'portfolio' berechnen (nicht auf 'bestand' UND nicht gemischt)
names = portfolio["Wertpapiername"].fillna("")
hit_names = names.str.contains(TECH_REGEX, na=False)
hit_etfs  = names.str.contains(ETF_REGEX,  na=False)
portfolio["TechTreffer"] = hit_names | hit_etfs

# 3) TechAffin / TechWeight aus dem Snapshot ableiten
tech_cnt = portfolio.groupby("Depotreferenz")["TechTreffer"].sum()
score_df["TechFokus"] = score_df["Depotnummer"].map(tech_cnt).fillna(0).astype(int)

tech_val = portfolio.loc[portfolio["TechTreffer"]].groupby("Depotreferenz")["Kurswert"].sum()
tot_val  = portfolio.groupby("Depotreferenz")["Kurswert"].sum()
tech_w   = (tech_val / (tot_val + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

################## score_df["Score"] += score_df["TechFokus"].apply(lambda x: 8 if x > 0 else 0)

# ===================== PP5: Punktelogik (nachdem Cashquote & TechFokus existieren) =====================
MODE = "bereitschaft"   # oder "aktiv"

# Score sauber neu aufbauen (verhindert Doppelzählung)
score_df["Score"] = 0

# Momentum/Aktivität (30/90 Tage)
score_df["Score"] += np.where(score_df["Orderfrequenz_30d"] >= 2, 8, 0)
score_df["Score"] += np.where(score_df["Orderfrequenz_90d"] >= 4, 6, 0)
score_df["Score"] += np.where(score_df["ØOrdervolumen"] > 2000, 2, 0)

# Cash je nach Ziel
if MODE == "bereitschaft":
    score_df["Score"] += np.where(score_df["Cashquote"] >= 0.25, 4, 0)
    score_df["Score"] += np.where(score_df["Cashquote"] >= 0.50, 2, 0)
else:
    # Beispiel für "aktiv": sehr hoher Cash leicht negativ
    # score_df["Score"] += np.where(score_df["Cashquote"] > 0.80, -2, 0)
    pass

# Tech-Bonus (moderater Aufschlag)
score_df["Score"] += np.where(score_df["TechFokus"] > 0, 4, 0)

# Optional: wenn du Aktienquote belohnen willst, dann hier – nicht oben!
# score_df["Score"] += np.where(score_df["Aktienquote"] > 0.80, 2, 0)
# =====================================================================================

# ===================== PP6: Guardrails (Abschläge & Soft-Cap) =====================
# Wertlos_Flag: Bestand ≈ 0 und >=1 Position mit Kurswert <= 0
zero_pos = (portfolio.assign(KW0=portfolio["Kurswert"] <= 0)
            .groupby("Depotreferenz")["KW0"].sum())
best_sum = portfolio.groupby("Depotreferenz")["Kurswert"].sum()
wertlos_flag_map = ((best_sum.fillna(0) <= 1) & (zero_pos.fillna(0) >= 1)).astype(int)

score_df["Wertlos_Flag"] = score_df["Depotnummer"].map(wertlos_flag_map).fillna(0).astype(int)
score_df["Score"] -= np.where(score_df["Wertlos_Flag"] == 1, 5, 0)

# Inaktivität 6M: Monate mit >=1 Order
six_m = heute - pd.DateOffset(months=6)
act6 = (orders[(orders["Orderdatum"] >= six_m) & (orders["Orderdatum"] <= heute)]
        .assign(mon=lambda d: d["Orderdatum"].dt.to_period("M"))
        .groupby(["Depotnummer","mon"]).size().reset_index()
        .groupby("Depotnummer")["mon"].nunique())

score_df["Activity_Months_6m"] = score_df["Depotnummer"].map(act6).fillna(0).astype(int)
score_df["Score"] -= np.where(score_df["Activity_Months_6m"] == 0, 4, 0)

# Soft-Cap für "hart negativ": wertlos + keine Aktivität
mask_hart = (score_df["Wertlos_Flag"] == 1) & (score_df["Activity_Months_6m"] == 0)
score_df.loc[mask_hart, "Score"] = np.minimum(score_df.loc[mask_hart, "Score"], 12)
# ================================================================================


# Aufbereitung
score_df["Score"] = score_df["Score"].clip(0, 100)
score_df["Score_Datum"] = heute


# ------------------------------------------
# 🧹 4. Präzise Rundung mit Decimal
# ------------------------------------------
def round_decimal(val, digits):
    try:
        return Decimal(str(val)).quantize(Decimal(f'1.{"0"*digits}'), rounding=ROUND_HALF_UP)
    except:
        return Decimal('0.0')

score_df["ØOrdervolumen"] = score_df["ØOrdervolumen"].fillna(0).apply(lambda x: round_decimal(x, 4))
score_df["Aktienquote"] = score_df["Aktienquote"].fillna(0).apply(lambda x: round_decimal(x, 6))
score_df["Cashquote"] = score_df["Cashquote"].fillna(0).apply(lambda x: round_decimal(x, 6))
score_df["TechFokus"] = score_df["TechFokus"].fillna(0).astype(int)

# ✅ HIER: Konvertiere Decimal → float für SQL Insert
score_df["ØOrdervolumen"] = score_df["ØOrdervolumen"].astype(float)
score_df["Aktienquote"] = score_df["Aktienquote"].astype(float)
score_df["Cashquote"] = score_df["Cashquote"].astype(float)

# ------------------------------------------
# 💾 4. In Tabelle CD_FA_SCORING speichern
# ------------------------------------------

score_df_out = score_df.loc[:, [
    "Depotnummer", "Orderfrequenz", "ØOrdervolumen", "Aktienquote",
    "Cashquote", "TechFokus", "Score", "Score_Datum"
]].copy()

# Optional: vorherige Daten löschen

# Nur den heutigen Tag löschen
with engine.begin() as conn:
    conn.execute(
        text("DELETE FROM CD_FA_SCORING WHERE Score_Datum = :heute"),
        {"heute": heute.date()}
    )

sql_dtypes = {
    "Depotnummer": String(50),
    "Orderfrequenz": Integer(),
    "ØOrdervolumen": Numeric(18, 4),
    "Aktienquote": Numeric(10, 6),
    "Cashquote": Numeric(10, 6),
    "TechFokus": Integer(),
    "Score": Integer(),
    "Score_Datum": Date()
}
# 👉 Korrekte Datentypen setzen

score_df_out.loc[:, "ØOrdervolumen"] = pd.to_numeric(score_df_out["ØOrdervolumen"], errors="coerce")
score_df_out.loc[:, "Aktienquote"]   = pd.to_numeric(score_df_out["Aktienquote"], errors="coerce")
score_df_out.loc[:, "Cashquote"]     = pd.to_numeric(score_df_out["Cashquote"], errors="coerce")
score_df_out.loc[:, "TechFokus"]     = score_df_out["TechFokus"].astype("int64")
score_df_out.loc[:, "Score"]         = pd.to_numeric(score_df_out["Score"], errors="coerce")
score_df_out.loc[:, "Score_Datum"]   = pd.to_datetime(score_df_out["Score_Datum"], errors="coerce")

#print(score_df_out)
#print(score_df_out.dtypes)

# Ergebnisse schreiben
score_df_out.to_sql("CD_FA_SCORING", engine, if_exists="append", index=False, dtype=sql_dtypes)

print("- Scoring erfolgreich in CD_FA_SCORING gespeichert.")

