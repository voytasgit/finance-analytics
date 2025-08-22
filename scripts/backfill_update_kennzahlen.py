# -*- coding: utf-8 -*-
# Feature MAtrix
"""
backfill_update_kennzahlen.py

Zweck
-----
- Einmaliger (oder wiederholbarer) Backfill von CD_FA_AGG_KENNZAHLEN über einen Datumsbereich.
- Quellen-DB (Provisionen): lädt Files je Tag (U=Orders kumulativ bis Datei_ID, B=Saldo/Bestand Snapshot).
- Ziel-DB (FA): schreibt je Tag deterministisch (DELETE + INSERT).

Wichtig
-------
- Die Feature-Logik ist in compute_agg() gekapselt und identisch zur Live-Variante (as-of sauber).
- Die Funktion write_agg() schreibt je Stichtag deterministisch in die Zieltabelle.
- Die Backfill-Schleife organisiert nur die Auswahl/Filterung der Input-Daten je Stichtag.

Voraussetzungen
---------------
- In db_connection.py sollte es zwei Builder geben:
    get_engine_prov()  -> Engine für "Provisionen"-DB (Quelle)
    get_engine_fa()    -> Engine für "FA"-DB (Ziel)
  Falls nicht vorhanden: Engines an backfill_update_kennzahlen(...) direkt übergeben.

- Tabellen in Quelle:
    Provisionen.dbo.cd_Datei (DateiID, Datum, DateiName, Pruefziffer, ...)
    Provisionen.dbo.cd_Order
    Provisionen.dbo.cd_Saldo
    Provisionen.dbo.cd_Bestand
    (optional) Provisionen.dbo.cd_Buchung  (falls vorhanden; sonst ohne Buchungen scorieren)

- Zieltabelle in FA-DB:
    dbo.CD_FA_AGG_KENNZAHLEN
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import re
from sqlalchemy import text
from datetime import datetime, timedelta


from db_connection import get_engine
from db_connection import get_engine_source

# -------------------------------------------------------------------
# 1) Engines beziehen (versuche auto, sonst per Parameter übergeben)
# -------------------------------------------------------------------
try:
    from db_connection import get_engine_source, get_engine
except Exception:
    get_engine_source = None
    get_engine   = None


# -------------------------------------------------------------------
# 2) Feature-Logik (identisch zur Live-Variante, as-of sauber)
# -------------------------------------------------------------------
def compute_agg(orders: pd.DataFrame,
                saldo: pd.DataFrame,
                bestand: pd.DataFrame,
                buchungen: pd.DataFrame,
                as_of) -> pd.DataFrame:
    """Berechnet alle Kennzahlen as-of (ohne DB-Schreiben)."""
    as_of_ts = pd.Timestamp(as_of)
    start_30 = as_of_ts - pd.Timedelta(days=30)
    start_90 = as_of_ts - pd.Timedelta(days=90)

    # Kopien + Grundtypen
    orders  = orders.copy()
    saldo   = saldo.copy()
    bestand = bestand.copy()

    # Typisierung / Null-Handling
    orders["Cashflow"]   = pd.to_numeric(orders.get("Cashflow"), errors="coerce").fillna(0.0)
    orders["Orderdatum"] = pd.to_datetime(orders.get("Orderdatum"), errors="coerce")

    if "Datum" in saldo.columns:
        saldo["Datum"] = pd.to_datetime(saldo["Datum"], errors="coerce")
    if "Datum" in bestand.columns:
        bestand["Datum"] = pd.to_datetime(bestand["Datum"], errors="coerce")

    if "Saldosollwert" in saldo.columns:
        saldo["Saldosollwert"] = pd.to_numeric(saldo["Saldosollwert"], errors="coerce").fillna(0.0)
    if "Kurswert" in bestand.columns:
        bestand["Kurswert"] = pd.to_numeric(bestand["Kurswert"], errors="coerce").fillna(0.0)

    # Schlüssel als String
    if "Depotnummer" in orders.columns:
        orders["Depotnummer"] = orders["Depotnummer"].astype(str)
    if "Kontoreferenz" in saldo.columns:
        saldo["Kontoreferenz"] = saldo["Kontoreferenz"].astype(str)
    if "Depotreferenz" in bestand.columns:
        bestand["Depotreferenz"] = bestand["Depotreferenz"].astype(str)

    # Buchungen (optional)
    if buchungen is not None:
        buchungen = buchungen.copy()
        buchungen["Kontoreferenz"] = buchungen["Kontoreferenz"].astype(str)
        buchungen["Valutadatum"]   = pd.to_datetime(buchungen["Valutadatum"], errors="coerce")
        buchungen["Betrag"]        = pd.to_numeric(buchungen["Betrag"], errors="coerce").fillna(0.0)

    # --------- As-of Snapshots / Filter ---------
    orders_asof = orders[orders["Orderdatum"] <= as_of_ts].copy()

    if "Datum" in saldo.columns:
        saldo_view = (saldo[saldo["Datum"] <= as_of_ts]
                      .sort_values("Datum")
                      .drop_duplicates("Kontoreferenz", keep="last"))
    else:
        saldo_view = saldo.copy()

    if "Datum" in bestand.columns:
        bestand_view = bestand[bestand["Datum"] <= as_of_ts].copy()
        idx = (bestand_view.groupby("Depotreferenz")["Datum"].transform("max") == bestand_view["Datum"])
        bestand_snapshot = bestand_view[idx].copy()
    else:
        bestand_snapshot = bestand.copy()

    # Depotmenge as-of (nur was „existiert“)
    alle_depots = (
        pd.Index(orders_asof["Depotnummer"].dropna().unique().astype(str))
         .union(pd.Index(bestand_snapshot["Depotreferenz"].dropna().unique().astype(str)))
         .union(pd.Index(saldo_view["Kontoreferenz"].dropna().unique().astype(str)))
    )
    agg = pd.DataFrame({"Depotnummer": alle_depots})
    agg["Score_Datum"] = as_of_ts

    # --------- Letzter Kauf / Trade ---------
    kauf_orders = orders_asof[orders_asof["Ordertyp"].str.contains("Ankauf", case=False, na=False)]
    last_kauf = kauf_orders.groupby("Depotnummer")["Orderdatum"].max()
    agg["Letzter_Kauf"] = agg["Depotnummer"].map(last_kauf)
    agg["Inaktiv_seit_Kauf"] = (as_of_ts - agg["Letzter_Kauf"]).dt.days
    agg["Last_Buy_Days"]     = agg["Inaktiv_seit_Kauf"]
    last_trade = orders_asof.groupby("Depotnummer")["Orderdatum"].max()
    agg["Last_Trade_Days"] = (as_of_ts - agg["Depotnummer"].map(last_trade)).dt.days

    # --------- Buchungen / Sparplan (saubere Klassifizierung) ---------
    if buchungen is not None and not buchungen.empty:
        b_all = buchungen.copy()
        # Grundreinigung
        for c in ["Buchungtyp", "Bemerkung"]:
            if c in b_all.columns:
                b_all[c] = b_all[c].fillna("").astype(str)
            else:
                b_all[c] = ""

        # Heuristiken für interne Umbuchungen / Depotüberträge ausschließen
        mask_internal = b_all["Bemerkung"].str.contains(
            r"umbuchung|intern|depot[ -]?übertrag|uebertrag|transfer", case=False, na=False
        )
        b_all = b_all[~mask_internal].copy()

        # Klassifizierung: inflow/outflow/ignore
        def classify(t: str) -> str:
            t = (t or "").lower()
            if "einlage" in t or "eingang" in t or "überweisung" in t or "ueberweisung" in t:
                return "in"
            if "entnahme" in t or "ausgang" in t or "auszahlung" in t:
                return "out"
            if "steuer" in t:     # Steuerrückerstattung – optional als 'neutral_in'
                return "tax_in"
            if "zins" in t or "gebühr" in t or "gebuehr" in t or "entgelt" in t or "spesen" in t:
                return "fee_out"
            if "ertrag" in t or "div" in t:
                return "yield_in"  # Dividende/Ertrag – i.d.R. kein frisches externes Geld
            return "ignore"

        b_all["klasse"] = b_all["Buchungtyp"].apply(classify)

        # Nur Zeitfenster bis as_of
        b_all = b_all[pd.to_datetime(b_all["Valutadatum"], errors="coerce") <= as_of_ts].copy()

        # Fenstergrenzen
        b30 = b_all[b_all["Valutadatum"] >= start_30]
        b90 = b_all[b_all["Valutadatum"] >= start_90]

        # Definitionen:
        # - EXTERNE EINZAHLUNG (fresh money): klasse in {'in'}   (optional: + 'tax_in' wenn gewünscht)
        # - Outflows (für Diagnostik): klasse in {'out','fee_out'}
        inflow_classes  = {"in"}            # ggf. {"in","tax_in"} wenn Steuer als investierbar gilt
        outflow_classes = {"out", "fee_out"}

        # Summen je Konto
        inflow_30 = (b30[b30["klasse"].isin(inflow_classes)]
                     .groupby("Kontoreferenz")["Betrag"].sum())
        inflow_90 = (b90[b90["klasse"].isin(inflow_classes)]
                     .groupby("Kontoreferenz")["Betrag"].sum())

        outflow_30 = (b30[b30["klasse"].isin(outflow_classes)]
                      .groupby("Kontoreferenz")["Betrag"].sum())
        outflow_90 = (b90[b90["klasse"].isin(outflow_classes)]
                      .groupby("Kontoreferenz")["Betrag"].sum())

        # In AGG schreiben (rückwärts-kompatible Namen befüllen)
        agg["Einzahlung_30d"] = agg["Depotnummer"].map(inflow_30)
        agg["Einzahlung_90d"] = agg["Depotnummer"].map(inflow_90)

        # Sparplan-Klein (nur echte Einzüge betrachten)
        b180 = b_all[(b_all["Valutadatum"] >= as_of_ts - pd.Timedelta(days=180))
                     & (b_all["klasse"].isin(inflow_classes))].copy()
        b180["Monat"] = pd.to_datetime(b180["Valutadatum"]).dt.to_period("M")
        b180["Tag"]   = pd.to_datetime(b180["Valutadatum"]).dt.day

        #small = b180[(b180["Betrag"] > 0) & (b180["Betrag"] <= 200) & (b180["Tag"] <= 7)]
        # NEUE Regel: jeder Betrag < 1000 gilt als Sparbetrag (Tag egal)
        small = b180[(b180["Betrag"] > 0) & (b180["Betrag"] < 1000)]

        monthly = small.groupby(["Kontoreferenz","Monat"])["Betrag"].sum().reset_index()
        last4 = pd.period_range(end=as_of_ts.to_period("M"), periods=4)
        counts = monthly[monthly["Monat"].isin(last4)].groupby("Kontoreferenz")["Monat"].nunique()
        sparplan_idx = set(counts[counts >= 3].index)
        agg["Sparplan_Klein"] = agg["Depotnummer"].apply(lambda x: 1 if x in sparplan_idx else 0)

        # Effektive Einzahlung: kleine Sparpläne nicht als „frisches Geld“ werten
        # agg["Einzahlung_30d_eff"] = np.where(
        #     (agg["Sparplan_Klein"] == 1) & (agg["Einzahlung_30d"].fillna(0) <= 200),
        #     0.0,
        #     agg["Einzahlung_30d"].fillna(0.0)
        # )
        agg["Einzahlung_30d_eff"] = np.where(
            (agg["Sparplan_Klein"] == 1) & (agg["Einzahlung_30d"].fillna(0) < 1000),
            0.0,
            agg["Einzahlung_30d"].fillna(0.0)
        )

        # NEW: Outflows & Netto für 30/90d
        #      (wir nutzen die bereits gebildeten b30/b90 und die Klassen-Logik)
        outflow_classes = {"out", "fee_out"}  # Entnahme + Gebühren/Zinsen
        Auszahlung_30d = b30[b30["klasse"].isin(outflow_classes)].groupby("Kontoreferenz")["Betrag"].sum()
        Auszahlung_90d = b90[b90["klasse"].isin(outflow_classes)].groupby("Kontoreferenz")["Betrag"].sum()

        agg["Auszahlung_30d"] = agg["Depotnummer"].map(Auszahlung_30d)
        agg["Auszahlung_90d"] = agg["Depotnummer"].map(Auszahlung_90d)
        agg["Nettoflow_30d"]  = agg["Einzahlung_30d"].fillna(0) - agg["Auszahlung_30d"].fillna(0)
        agg["Nettoflow_90d"]  = agg["Einzahlung_90d"].fillna(0) - agg["Auszahlung_90d"].fillna(0)

        # NEW: Sparplan_Strength_6m (Anzahl Monate mit "klein & Monatsanfang")
        # agg["Sparplan_Strength_6m"] = agg["Depotnummer"].map(
        #     b180[(b180["Betrag"]<=200) & (b180["Tag"]<=7)]
        #       .groupby("Kontoreferenz")["Monat"].nunique()
        # ).fillna(0).astype(int)
        agg["Sparplan_Strength_6m"] = agg["Depotnummer"].map(
            b180[(b180["Betrag"] > 0) & (b180["Betrag"] < 1000)]
              .groupby("Kontoreferenz")["Monat"].nunique()
        ).fillna(0).astype(int)


        # NEW: Deposit_Volatility_6m (StdAbw Monats-Einzahlungen 6M)
        dep_6m = (b_all[(b_all["klasse"].isin(inflow_classes)) &
                        (b_all["Valutadatum"] >= as_of_ts - pd.Timedelta(days=180))]
                  .assign(Monat=lambda x: x["Valutadatum"].dt.to_period("M"))
                  .groupby(["Kontoreferenz","Monat"])["Betrag"].sum().reset_index())
        vol_map = dep_6m.groupby("Kontoreferenz")["Betrag"].std()
        agg["Deposit_Volatility_6m"] = agg["Depotnummer"].map(vol_map).fillna(0.0)

        # NEW: Last_Cashflow_Days
        last_cf_map = b_all.groupby("Kontoreferenz")["Valutadatum"].max()
        #agg["Last_Cashflow_Days"] = (as_of_ts - agg["Depotnummer"].map(last_cf_map)).dt.days.fillna(9999).astype(int)
        agg["Last_Cashflow_Days"] = (as_of_ts - agg["Depotnummer"].map(last_cf_map)).dt.days
        agg["Last_Cashflow_Days"] = pd.to_numeric(agg["Last_Cashflow_Days"], errors="coerce")

    else:
        agg["Einzahlung_30d"] = 0.0
        agg["Einzahlung_90d"] = 0.0
        agg["Sparplan_Klein"] = 0
        agg["Einzahlung_30d_eff"] = 0.0
        # ggf. auch: agg["Auszahlung_30d"] = 0.0, ...
        agg["Auszahlung_30d"] = 0.0
        agg["Auszahlung_90d"] = 0.0
        agg["Nettoflow_30d"]  = 0.0
        agg["Nettoflow_90d"]  = 0.0
        agg["Deposit_Volatility_6m"] = 0.0
        agg["Sparplan_Strength_6m"] = 0
        agg["Last_Cashflow_Days"] = np.nan #9999

    # --------- Frequenzen / Volumina ---------
    if not orders_asof.empty:
        orders_asof["Monat"] = orders_asof["Orderdatum"].dt.to_period("M")
    else:
        orders_asof["Monat"] = pd.PeriodIndex([], freq="M")

    agg["Orderfrequenz"]     = agg["Depotnummer"].map(orders_asof.groupby("Depotnummer")["Monat"].nunique())
    freq_30 = orders_asof[orders_asof["Orderdatum"] >= start_30].groupby("Depotnummer").size()
    freq_90 = orders_asof[orders_asof["Orderdatum"] >= start_90].groupby("Depotnummer").size()
    agg["Orderfrequenz_30d"] = agg["Depotnummer"].map(freq_30)
    agg["Orderfrequenz_90d"] = agg["Depotnummer"].map(freq_90)

    orders_asof["Ordervolumen_abs"] = orders_asof["Cashflow"].abs()
    agg["ØOrdervolumen"]     = agg["Depotnummer"].map(orders_asof.groupby("Depotnummer")["Ordervolumen_abs"].mean())
    vol_30 = orders_asof[orders_asof["Orderdatum"] >= start_30].groupby("Depotnummer")["Ordervolumen_abs"].mean()
    vol_90 = orders_asof[orders_asof["Orderdatum"] >= start_90].groupby("Depotnummer")["Ordervolumen_abs"].mean()
    agg["ØOrdervolumen_30d"] = agg["Depotnummer"].map(vol_30)
    agg["ØOrdervolumen_90d"] = agg["Depotnummer"].map(vol_90)

    # --------- Tech / Cash / Aktienquote (Snapshot) ---------
    bestand_snapshot["Wertpapiername"] = bestand_snapshot.get("Wertpapiername", "").fillna("")
    if "Wertpapiertyp" in bestand_snapshot.columns:
        bestand_snapshot["Wertpapiertyp"] = bestand_snapshot["Wertpapiertyp"].fillna("").astype(str)
    else:
        bestand_snapshot["Wertpapiertyp"] = ""

    # bestand_snapshot["TechTreffer"] = bestand_snapshot["Wertpapiername"].str.contains(
    #     "NVIDIA|APPLE|TESLA|TECH|AI", case=False, na=False
    # )
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

    names = bestand_snapshot["Wertpapiername"].fillna("")
    hit_names = names.str.contains(TECH_REGEX, na=False)  # kein UserWarning mehr
    hit_etfs  = names.str.contains(ETF_REGEX,  na=False)
    # TECH ENDE

    bestand_snapshot["TechTreffer"] = hit_names | hit_etfs

    tech_sum = bestand_snapshot.groupby("Depotreferenz")["TechTreffer"].sum()
    agg["TechAffin"] = agg["Depotnummer"].map(tech_sum)

    bestand_sum = bestand_snapshot.groupby("Depotreferenz")["Kurswert"].sum()

    pos_cnt     = bestand_snapshot.groupby("Depotreferenz")["Wertpapiername"].size()
    zero_pos    = (bestand_snapshot.assign(KW0=(bestand_snapshot["Kurswert"]<=0))
                                  .groupby("Depotreferenz")["KW0"].sum())

    s = saldo_view.merge(bestand_sum, left_on="Kontoreferenz", right_on="Depotreferenz", how="left")

    # Verhältnis Cash zu Bestand (unbeschnitten – Diagnose)
    #s["Cash_over_bestand"] = s["Saldosollwert"] / (s["Kurswert"] + 1e-6)
    s["Cash_over_bestand"] = np.where(s["Kurswert"] > 0, s["Saldosollwert"] / s["Kurswert"], np.nan)
    agg["Cash_over_bestand"] = pd.to_numeric(
    agg["Depotnummer"].map(s.set_index("Kontoreferenz")["Cash_over_bestand"]),
    errors="coerce"
).replace([np.inf, -np.inf], np.nan)


    #agg["Cash_over_bestand"] = pd.to_numeric(agg["Depotnummer"].map(s.set_index("Kontoreferenz")["Cash_over_bestand"]),  errors="coerce").replace([np.inf, -np.inf], np.nan)  # und ggf. .clip(0, 100)


    # >>> EMPFOHLEN FÜR REGELN: echter Cash-Anteil am Gesamtdepot (0..1)
    s["Cash_to_total"] = s["Saldosollwert"] / (s["Saldosollwert"] + s["Kurswert"] + 1e-6)

    # In AGG mappen
    agg["Cash_over_bestand"] = agg["Depotnummer"].map(s.set_index("Kontoreferenz")["Cash_over_bestand"])
    agg["Cashquote_true"]    = agg["Depotnummer"].map(s.set_index("Kontoreferenz")["Cash_to_total"])

    # Für Regeln nutzen wir die True-Quote (0..1). Alte Cashquote kannst du behalten, aber nicht mehr für Scoring.
    agg["Cashquote_true"] = pd.to_numeric(agg["Cashquote_true"], errors="coerce").fillna(0.0).clip(0, 1)


    # NEW: Absolute Werte & Positionszahlen
    agg["Cash_abs"]    = agg["Depotnummer"].map(s.set_index("Kontoreferenz")["Saldosollwert"])
    agg["Bestandwert"] = agg["Depotnummer"].map(bestand_sum)
    agg["Depotwert"]   = agg["Cash_abs"].fillna(0) + agg["Bestandwert"].fillna(0)
    agg["Pos_Count"]   = agg["Depotnummer"].map(pos_cnt).fillna(0).astype(int)
    agg["ZeroPos"]     = agg["Depotnummer"].map(zero_pos).fillna(0).astype(int)
    agg["Wertlos_Flag"]= ((agg["Bestandwert"].fillna(0) <= 1) & (agg["ZeroPos"] >= 1)).astype(int)

    # NEW: HHI_Concentration (Herfindahl) – Klumpenrisiko
    w = (bestand_snapshot.groupby(["Depotreferenz","Wertpapierreferenz"], as_index=False)["Kurswert"].sum())
    w["sum_dep"] = w.groupby("Depotreferenz")["Kurswert"].transform("sum")
    w["w"] = np.where(w["sum_dep"]>0, w["Kurswert"]/w["sum_dep"], 0.0)
    hhi = w.groupby("Depotreferenz")["w"].apply(lambda s: float(np.square(s).sum()))
    agg["HHI_Concentration"] = agg["Depotnummer"].map(hhi)

    # NEW: DeepLossShare_50 & WinLoss_Skew (Einstandswerte nötig)
    b_ein = bestand_snapshot.copy()
    if "Einstandswerte" in b_ein.columns:
        b_ein = b_ein[b_ein["Einstandswerte"]>0].copy()
        pnl = b_ein["Kurswert"] - b_ein["Einstandswerte"]
        b_ein["pnl"] = pnl
        tot = b_ein.groupby("Depotreferenz")["Kurswert"].sum()
        deep = b_ein[pnl / b_ein["Einstandswerte"] <= -0.5].groupby("Depotreferenz")["Kurswert"].sum()
        win  = b_ein[b_ein["pnl"]>0].groupby("Depotreferenz")["Kurswert"].sum()
        loss = b_ein[b_ein["pnl"]<=0].groupby("Depotreferenz")["Kurswert"].sum()
        agg["DeepLossShare_50"] = agg["Depotnummer"].map((deep / tot).replace([np.inf,-np.inf], np.nan)).fillna(0.0)
        agg["WinLoss_Skew"]     = agg["Depotnummer"].map(((win - loss) / tot).replace([np.inf,-np.inf], np.nan)).fillna(0.0)
    else:
        agg["DeepLossShare_50"] = 0.0
        agg["WinLoss_Skew"]     = 0.0

    # NEW: TechWeight (Anteil Kurswert in Tech-Titeln)
    #tech_mask = bestand_snapshot["Wertpapiername"].str.contains("NVIDIA|APPLE|TESLA|TECH|AI", case=False, na=False)
    tech_mask = hit_names | hit_etfs
    tech_sum  = bestand_snapshot.loc[tech_mask].groupby("Depotreferenz")["Kurswert"].sum()
    tot_sum   = bestand_snapshot.groupby("Depotreferenz")["Kurswert"].sum()
    tech_w    = (tech_sum / (tot_sum + 1e-6)).replace([np.inf,-np.inf], np.nan)
    agg["TechWeight"] = agg["Depotnummer"].map(tech_w).fillna(0.0)
    # NEW ende


    #s["Cashquote"] = s["Saldosollwert"] / (s["Kurswert"] + 1e-6)
    s["Cashquote"] = np.where(s["Kurswert"] > 0,
                              s["Saldosollwert"] / s["Kurswert"],
                              np.nan)

    agg["Cashquote"] = agg["Depotnummer"].map(s.set_index("Kontoreferenz")["Cashquote"])

    aktien_only = bestand_snapshot[bestand_snapshot["Wertpapiertyp"].str.upper() == "AKTIE"]
    aktien_sum  = aktien_only.groupby("Depotreferenz")["Kurswert"].sum()
    gesamt_sum  = bestand_snapshot.groupby("Depotreferenz")["Kurswert"].sum()
    agg["Aktienquote"] = agg["Depotnummer"].map(aktien_sum / (gesamt_sum + 1e-6))

    # # --------- Typen/Clips ---------

    # FIX: pro-Depot Ableitungen, damit hier garantiert Series entstehen
    # Last_Cashflow_Days: Tage seit letzter (nicht-null) Buchung pro Konto
    if buchungen is not None and not buchungen.empty:
        b_nz = buchungen.copy()
        b_nz["Valutadatum"] = pd.to_datetime(b_nz["Valutadatum"], errors="coerce")
        last_cf = (b_nz[b_nz["Betrag"] != 0]
                   .groupby("Kontoreferenz")["Valutadatum"].max())
        # -> Series je Depotreferenz
        agg["Last_Cashflow_Days"] = (pd.Timestamp(as_of) - agg["Depotnummer"].map(last_cf)).dt.days
    else:
        agg["Last_Cashflow_Days"] = np.nan

    # Activity_Months_6m: in wie vielen der letzten 6 Monate gab es mind. 1 Order?
    six_m = pd.Timestamp(as_of) - pd.DateOffset(months=6)
    o6 = (orders[(orders["Orderdatum"] >= six_m) & (orders["Orderdatum"] <= pd.Timestamp(as_of))]
          .assign(mon=lambda df: df["Orderdatum"].dt.to_period("M")))
    act6 = o6.groupby(["Depotnummer","mon"]).size().reset_index().groupby("Depotnummer")["mon"].nunique()
    agg["Activity_Months_6m"] = agg["Depotnummer"].map(act6)

    # Streak_Inactive_Months: einfache Näherung als Monate seit letztem Trade
    agg["Streak_Inactive_Months"] = np.floor(agg["Last_Trade_Days"].fillna(9999) / 30.0)

    # Sparplan_Strength_6m: wie oft (Monate) kleiner Einzug <=200 am Monatsanfang
    if buchungen is not None and not buchungen.empty:
        b6 = buchungen.copy()
        b6["Valutadatum"] = pd.to_datetime(b6["Valutadatum"], errors="coerce")
        b6 = b6[(b6["Valutadatum"] >= six_m) & (b6["Valutadatum"] <= pd.Timestamp(as_of))]
        b6["Monat"] = b6["Valutadatum"].dt.to_period("M")
        b6["Tag"]   = b6["Valutadatum"].dt.day
        #small = b6[(b6["Betrag"] > 0) & (b6["Betrag"] <= 200) & (b6["Tag"] <= 7)]
        small = b6[(b6["Betrag"] > 0) & (b6["Betrag"] < 1000)]
        sp_strength = small.groupby(["Kontoreferenz","Monat"]).size().reset_index().groupby("Kontoreferenz")["Monat"].nunique()
        agg["Sparplan_Strength_6m"] = agg["Depotnummer"].map(sp_strength)
    else:
        agg["Sparplan_Strength_6m"] = np.nan

    # DIY_FrequentSmall (Beispiel): viele kleine Orders (<200) in 90 Tagen
    o90 = orders[orders["Orderdatum"] >= (pd.Timestamp(as_of) - pd.Timedelta(days=90))].copy()
    o90["abs_cf"] = o90["Cashflow"].abs()
    diy_small = o90[o90["abs_cf"] > 0].groupby("Depotnummer")["abs_cf"].apply(lambda s: (s <= 200).sum())
    agg["DIY_FrequentSmall"] = agg["Depotnummer"].map(diy_small)

    # Pos_Count / ZeroPos / Wertlos_Flag – falls du sie nutzt, ebenfalls pro Depot sicherstellen
    # (entweder hast du sie schon vorher berechnet, sonst hier analog ergänzen)

    agg["Last_Cashflow_Days"] = agg["Last_Cashflow_Days"].fillna(9999).astype("int64")
    # --- ROBUSTER TYPISIERUNGSBLOCK (ersetzt deinen bisherigen) ---

    int_cols = [
        "Orderfrequenz","Orderfrequenz_30d","Orderfrequenz_90d","TechAffin","Sparplan_Klein",
        "Sparplan_Strength_6m","Activity_Months_6m","Pos_Count","ZeroPos",
        "Last_Cashflow_Days","Streak_Inactive_Months","Wertlos_Flag","DIY_FrequentSmall",
        "Inaktiv_seit_Kauf","Last_Buy_Days","Last_Trade_Days"
    ]
    float_cols = [
        "Einzahlung_30d","Einzahlung_30d_eff","Einzahlung_90d",
        "Auszahlung_30d","Auszahlung_90d","Nettoflow_30d","Nettoflow_90d",
        "ØOrdervolumen","ØOrdervolumen_30d","ØOrdervolumen_90d",
        "Aktienquote","Cashquote","Cash_abs","Bestandwert","Depotwert"
    ]

    # 1) fehlende Spalten anlegen – als Series in richtiger Länge
    for c in int_cols:
        if c not in agg.columns or pd.api.types.is_scalar(agg[c]):
            agg[c] = pd.Series(agg[c] if c in agg.columns else 0, index=agg.index)

    for c in float_cols:
        if c not in agg.columns or pd.api.types.is_scalar(agg[c]):
            agg[c] = pd.Series(agg[c] if c in agg.columns else 0.0, index=agg.index)

    # 2) typisieren
    for c in int_cols:
        agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0).astype("int64")
    for c in float_cols:
        agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0.0).astype("float64")

    # Inflow_Strength_30d = Nettoflow_30d relativ zum Depotwert
    agg["Inflow_Strength_30d"] = np.where(agg["Depotwert"]>0,
                                          agg["Nettoflow_30d"]/agg["Depotwert"], 0.0)
    # 3) Plausibilisierung
    agg["Cashquote"] = agg["Cashquote"].clip(0, 1)



    agg = agg.sort_values(["Depotnummer"]).reset_index(drop=True)

    # NEW: Activity_Months_6m (Monate mit >=1 Order in den letzten 6M)
    orders_6m = orders_asof[orders_asof["Orderdatum"] >= (as_of_ts - pd.DateOffset(months=6))].copy()
    orders_6m["Monat"] = orders_6m["Orderdatum"].dt.to_period("M")
    act6 = orders_6m.groupby("Depotnummer")["Monat"].nunique()
    agg["Activity_Months_6m"] = agg["Depotnummer"].map(act6).fillna(0).astype(int)

    # NEW: DIY_FrequentSmall (viel Aktivität, kleine Tickets)
    #      Kriterium: Orderfrequenz_90d >= 6 UND ØOrdervolumen_90d < 2.000
    agg["DIY_FrequentSmall"] = ((agg["Orderfrequenz_90d"].fillna(0) >= 6) &
                                (agg["ØOrdervolumen_90d"].fillna(0) < 2000)).astype(int)

    # NEW: Streak_Inactive_Months (zusammenhängende Monate ohne Order, rückwärts gezählt, max 12)
    o_mon = orders_asof["Orderdatum"].dt.to_period("M")
    last12 = pd.period_range((as_of_ts - pd.DateOffset(months=12)).to_period("M"), as_of_ts.to_period("M"), freq="M")
    last_by_dep = orders_asof.groupby("Depotnummer")["Monat"].apply(lambda s: set(s))
    def streak(dep):
        have = last_by_dep.get(dep, set())
        cnt = 0
        for m in reversed(last12):
            if m in have: break
            cnt += 1
        return cnt
    agg["Streak_Inactive_Months"] = agg["Depotnummer"].apply(streak).astype(int)

    return agg

def write_agg(agg: pd.DataFrame, engine_dest, as_of):
    with engine_dest.begin() as raw_conn:
        # Fast executemany aktivieren
        conn = raw_conn.execution_options(fast_executemany=True)

        conn.execute(
            text("DELETE FROM dbo.CD_FA_AGG_KENNZAHLEN WHERE Score_Datum = :d"),
            {"d": pd.Timestamp(as_of).date()}
        )

        # KEIN method="multi" verwenden
        agg.to_sql(
            "CD_FA_AGG_KENNZAHLEN",
            conn,
            if_exists="append",
            index=False,
            chunksize=1000,      # darf hier relativ groß sein
            method=None          # explizit oder einfach weglassen
        )

# -------------------------------------------------------------------
# 3) Loader aus der Provisionen-DB (Quelle)
# -------------------------------------------------------------------
def pick_file_ids(engine_src, as_of_date):
    """Ermittelt DateiID_U (Orders kumulativ) und DateiID_B (Saldo/Bestand Snapshot) für einen Tag."""
    q_u = text("""
        SELECT TOP 1 DateiID
        FROM Provisionen.dbo.cd_Datei
        WHERE Datum = :d AND DateiName LIKE 'comdirect_U-%'
        ORDER BY Datum DESC, Pruefziffer DESC
    """)
    q_b = text("""
        SELECT TOP 1 DateiID
        FROM Provisionen.dbo.cd_Datei
        WHERE Datum = :d AND DateiName LIKE 'comdirect_B-%'
        ORDER BY Datum DESC, Pruefziffer DESC
    """)
    with engine_src.begin() as conn:
        r_u = conn.execute(q_u, {"d": as_of_date}).scalar()
        r_b = conn.execute(q_b, {"d": as_of_date}).scalar()
    return r_u, r_b


def load_inputs_for_day(engine_src, as_of_date):
    """
    Lädt:
      - orders: kumulativ bis DateiID_U (<=)
      - saldo, bestand: Snapshot = DateiID_B (=)
      - buchungen: optional bis as_of_date (falls Tabelle vorhanden)
    """
    datei_u, datei_b = pick_file_ids(engine_src, as_of_date)
    if datei_u is None or datei_b is None:
        return None, None, None, None  # an diesem Tag nicht vollständig

    with engine_src.begin() as conn:
        orders = pd.read_sql(text("""
            SELECT *
            FROM Provisionen.dbo.cd_Order
            WHERE Datei_ID <= :fid
        """), conn, params={"fid": datei_u}, parse_dates=["Orderdatum", "Valutadatum"])

        saldo = pd.read_sql(text("""
            SELECT *
            FROM Provisionen.dbo.cd_Saldo
            WHERE Datei_ID = :fid
        """), conn, params={"fid": datei_b}, parse_dates=["Datum"])

        bestand = pd.read_sql(text("""
            SELECT *
            FROM Provisionen.dbo.cd_Bestand
            WHERE Datei_ID = :fid
        """), conn, params={"fid": datei_b}, parse_dates=["Datum"])

        # Buchungen sind optional; wenn Tabelle fehlt -> leer lassen
        try:
            buchungen = pd.read_sql(text("""
                SELECT Kontoreferenz, Betrag, Valutadatum
                FROM Provisionen.dbo.cd_Buchung
                WHERE Valutadatum <= :d
            """), conn, params={"d": as_of_date}, parse_dates=["Valutadatum"])
        except Exception:
            buchungen = None

    return orders, saldo, bestand, buchungen


# -------------------------------------------------------------------
# 4) Backfill-Schleife
# -------------------------------------------------------------------
def backfill_update_kennzahlen(engine_src=None,
                               engine_dest=None,
                               months=24,
                               start_date=None,
                               end_date=None,
                               dry_run=False,
                               verbose=True):
    """
    Führt den Backfill über [start_date .. end_date] aus. Wenn nicht gesetzt:
    - end_date = MAX(Datum) aus Provisionen.dbo.cd_Datei (Pruefziffer > 0)
    - start_date = end_date - months (Monate)

    Hinweise:
    - Es werden nur Tage verarbeitet, für die sowohl U- als auch B-Datei existieren.
    - write_agg() ist deterministisch (DELETE + INSERT).
    """
    if engine_src is None and get_engine_source is not None:
        engine_src = get_engine_source()
    if engine_dest is None and get_engine is not None:
        engine_dest = get_engine()
    if engine_src is None or engine_dest is None:
        raise RuntimeError("Engine(s) fehlen. Entweder get_engine_prov/get_engine_fa bereitstellen oder als Parameter übergeben.")

    # Enddatum bestimmen
    with engine_src.begin() as conn:
        max_datum = conn.execute(text("""
            SELECT MAX(Datum) FROM Provisionen.dbo.cd_Datei WHERE Pruefziffer > 0
        """)).scalar()

    if max_datum is None:
        raise RuntimeError("In Provisionen.dbo.cd_Datei wurde kein Datum mit Pruefziffer > 0 gefunden.")

    if end_date is None:
        end_date = pd.Timestamp(max_datum).normalize()
    else:
        end_date = pd.Timestamp(end_date).normalize()

    if start_date is None:
        start_date = (end_date - pd.DateOffset(months=months)).normalize()
    else:
        start_date = pd.Timestamp(start_date).normalize()

    if verbose:
        print(f"Backfill-Zeitraum: {start_date.date()} .. {end_date.date()} (inclusive)")

    d = start_date
    processed = 0
    skipped   = 0

    while d <= end_date:
        as_of = d.date()
        # Prüfen, ob für den Tag beide Datei-IDs existieren
        fid_u, fid_b = pick_file_ids(engine_src, as_of)
        if fid_u is None or fid_b is None:
            if verbose:
                print(f"-- {as_of}: keine vollständigen Dateien (U={fid_u}, B={fid_b}) -> skip")
            d += timedelta(days=1)
            skipped += 1
            continue

        if verbose:
            print(f"-- {as_of}: Dateien U={fid_u}, B={fid_b} – lade Inputs ...", end="", flush=True)

        orders, saldo, bestand, buchungen = load_inputs_for_day(engine_src, as_of)
        if orders is None or saldo is None or bestand is None:
            if verbose:
                print(" fehlend -> skip")
            d += timedelta(days=1)
            skipped += 1
            continue

        if verbose:
            print(" ok. compute_agg ...", end="", flush=True)

        agg = compute_agg(orders=orders, saldo=saldo, bestand=bestand, buchungen=buchungen, as_of=as_of)

        if dry_run:
            if verbose:
                print(f" (dry-run) rows={len(agg)}")
        else:
            if verbose:
                print(f" write ({len(agg)} Zeilen) ...", end="", flush=True)
            write_agg(agg, engine_dest, as_of)
            if verbose:
                print(" done.")

        processed += 1
        d += timedelta(days=1)

    if verbose:
        print(f"Fertig. verarbeitet={processed}, übersprungen={skipped}")


# -------------------------------------------------------------------
# 5) Optionaler CLI-Einstieg
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Beispielaufruf:
    # python backfill_update_kennzahlen.py  (nimmt default: letzte MAX(Datum), -24 Monate)
    # Für individuelle Range: unten Parameter setzen oder per argparse ausbauen.
    try:
        backfill_update_kennzahlen(
            engine_src=None,   # nimmt get_engine_prov(), wenn vorhanden
            engine_dest=None,  # nimmt get_engine_fa(),   wenn vorhanden
            months=24,
            start_date=None,
            end_date=None,
            dry_run=False,
            verbose=True
        )
    except Exception as e:
        print(f"X Backfill-Fehler: {e}")
        sys.exit(1)
