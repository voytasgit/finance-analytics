#!/usr/bin/env python3
"""
Label-Builder für ML-Training: erzeugt label_invest_T{H} (0/1) je (Depotnummer, stichtag).

NEU (angepasst auf eure DB-Infra):
- Nutzt `db_connection.get_engine()` für **AGG** (Ziel-DB) und
  `db_connection.get_engine_source()` für **CD_ORDER/CD_BUCHUNG** (Quell-DB).
- SQL-Pushdown & Fensterung: lädt Order/Buchungen nur im relevanten Datumsfenster
  (min(stichtag)+1 .. max(stichtag)+H) → performant trotz Daten seit 2018.
- Optionales `--date_from/--date_to` überschreibt das Fenster (YYYY-MM-DD).
- Spalten-Projection: lädt nur benötigte Spalten.
- Chunked-Read bei sehr großen Tabellen.

Beispiele:
  CSV (unverändert):
    python label_builder.py --source csv \
      --agg ./data/CD_FA_AGG_KENNZAHLEN.csv \
      --orders ./data/CD_ORDER.csv \
      --buchungen ./data/CD_BUCHUNG.csv \
      --horizon 7 --min_einzahlung 100 \
      --out_parquet ./data/agg_with_labels.parquet

  MSSQL (interne Verbindungen aus db_connection.py):
    python label_builder.py --source mssql \
      --agg_table CD_FA_AGG_KENNZAHLEN --orders_table CD_ORDER --buch_table CD_BUCHUNG \
      --horizon 7 --min_einzahlung 100 \
      --date_from 2023-01-01 --date_to 2024-12-31 \
      --out_table CD_FA_LABELS

Ausgabe:
  - Parquet/CSV-Datei (Features + label_invest_T{H})
  - optional SQL-Tabelle CD_FA_LABELS (Depotnummer, stichtag, label_invest_T{H}, horizon_tage, created_at)

Standard (Fenster automatisch aus AGG):

python label_builder.py --source mssql \
  --agg_table CD_FA_AGG_KENNZAHLEN \
  --orders_table CD_ORDER \
  --buch_table CD_BUCHUNG \
  --horizon 7 --min_einzahlung 100 \
  --out_table CD_FA_LABELS

Mit explizitem Zeitraum (z. B. Backfill/Training):

python label_builder.py --source mssql \
  --agg_table CD_FA_AGG_KENNZAHLEN \
  --orders_table CD_ORDER \
  --buch_table CD_BUCHUNG \
  --horizon 7 --min_einzahlung 100 \
  --date_from 2023-01-01 --date_to 2024-12-31 \
  --out_table CD_FA_LABELS


Technische Details der Anpassungen

Neue Funktion: load_mssql_via_internal(...)

Importiert Engines aus db_connection.py

Liest AGG aus Ziel‑DB (get_engine()), bestimmt Datumsfenster, lädt dann ORDER/BUCHUNG aus Source‑DB (get_engine_source()) nur innerhalb dieses Fensters.

Abfragebeispiele intern erzeugt:

SELECT Depotnummer, Orderdatum[, status] FROM CD_ORDER WHERE Orderdatum BETWEEN <from> AND <to>

SELECT Depotnummer, Buchungdatum, betrag FROM CD_BUCHUNG WHERE Buchungdatum BETWEEN <from> AND <to>

Große Tabellen: pandas.read_sql(..., chunksize=500_000) und anschließendes concat.

Label-Builder für ML-Training: erzeugt label_invest_T{H} (0/1) je (Depotnummer, stichtag).

Label-Builder für ML-Training: erzeugt label_invest_T{H} (0/1) je (Depotnummer, stichtag).

NEU (angepasst):
- Nutzt db_connection.get_engine() für AGG und db_connection.get_engine_source() für ORDER/BUCHUNG.
- Optional: Parameter --start_date und --end_date zum Einschränken des Zeitraums.
- SQL-Pushdown mit Datumseinschränkung verhindert Laden aller Daten seit 2018.

Training/Backfill (MSSQL, 6 Monate):

python label_builder.py --source mssql \
  --agg_table CD_FA_AGG_KENNZAHLEN \
  --orders_table CD_ORDER \
  --buch_table CD_BUCHUNG \
  --horizon 7 --min_einzahlung 100 \
  --start_date 2025-01-01 --end_date 2025-06-30 \
  --out_table CD_FA_LABELS


CSV‑Testlauf (kleiner Zeitraum):

python label_builder.py --source csv \
  --agg ./data/CD_FA_AGG_KENNZAHLEN.csv \
  --orders ./data/CD_ORDER.csv \
  --buchungen ./data/CD_BUCHUNG.csv \
  --horizon 7 --min_einzahlung 100 \
  --start_date 2025-04-01 --end_date 2025-04-30 \
  --out_parquet ./data/agg_with_labels.parquet


Beispiel: zwei Läufe hintereinander
# 1) T=7
python label_builder.py --source mssql \
  --agg_table CD_FA_AGG_KENNZAHLEN \
  --orders_table cd_Order --buch_table cd_Buchung \
  --start_date 2025-01-01 --end_date 2025-03-31 \
  --horizon 7 --out_table CD_FA_LABELS

# 2) T=14 (füllt NUR label_invest_T14; T7 bleibt unberührt)
python label_builder.py --source mssql \
  --agg_table CD_FA_AGG_KENNZAHLEN \
  --orders_table cd_Order --buch_table cd_Buchung \
  --start_date 2025-01-01 --end_date 2025-03-31 \
  --horizon 14 --out_table CD_FA_LABELS


"""
from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional

# ----------------------------------
# Utilities
# ----------------------------------
def _ensure_datetime(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns and not np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c])
    return df

def _find_future_event_within(left: pd.DataFrame, right: pd.DataFrame, horizon_days: int) -> pd.Series:
    left = left.sort_values(["Depotnummer", "stichtag"]).reset_index(drop=True)
    right = right.sort_values(["Depotnummer", "event_date"]).reset_index(drop=True)

    result = np.zeros(len(left), dtype=bool)
    r_groups = right.groupby("Depotnummer")
    l_idx = left.groupby("Depotnummer").indices

    for depot, idxs in l_idx.items():
        l_dates = left.loc[idxs, "stichtag"].values.astype("datetime64[ns]")
        try:
            r_sub = r_groups.get_group(depot)
        except KeyError:
            continue
        r_dates = r_sub["event_date"].values.astype("datetime64[ns]")
        if r_dates.size == 0:
            continue
        start_dates = l_dates + np.timedelta64(1, 'D')
        end_dates = l_dates + np.timedelta64(horizon_days, 'D')
        pos = np.searchsorted(r_dates, start_dates, side='left')
        ok = (pos < r_dates.size)
        valid = ok & (r_dates[np.clip(pos, 0, r_dates.size-1)] <= end_dates)
        result[idxs] = valid
    return pd.Series(result, index=left.index)

# ----------------------------------
# Loader
# ----------------------------------
def load_mssql_via_internal(
    agg_table: str,
    orders_table: str,
    buch_table: str,
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    horizon: int,
    stichtag_col: str = "Score_Datum",
    order_date_col: str = "Orderdatum",
    buch_date_col: str = "Buchungdatum",
    depot_col: str = "Depotnummer",
    konto_col: str = "Kontoreferenz",
    order_status_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        from db_connection import get_engine, get_engine_source
    except ImportError:
        import sys, pathlib
        parent = pathlib.Path(__file__).resolve().parent.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        from db_connection import get_engine, get_engine_source
    eng_agg = get_engine()
    eng_src = get_engine_source()

    where_agg = ""
    if start_date and end_date:
        where_agg = f"WHERE {stichtag_col} >= CONVERT(date, '{start_date}') AND {stichtag_col} <= CONVERT(date, '{end_date}')"
    q_agg = f"SELECT * FROM {agg_table} {where_agg}"
    agg = pd.read_sql(q_agg, eng_agg)
    _ensure_datetime(agg, [stichtag_col])

    if len(agg) == 0:
        return agg, agg.head(0), agg.head(0)

    min_s = agg[stichtag_col].min() + pd.Timedelta(days=1)
    max_s = agg[stichtag_col].max() + pd.Timedelta(days=horizon)
    df_str = pd.to_datetime(min_s).date().isoformat()
    dt_str = pd.to_datetime(max_s).date().isoformat()

    proj_order = f"{depot_col}, {order_date_col}"
    if order_status_col:
        proj_order += f", {order_status_col}"
    q_order = f"SELECT {proj_order} FROM {orders_table} WHERE {order_date_col} BETWEEN CONVERT(date, '{df_str}') AND CONVERT(date, '{dt_str}')"

    q_buch = (
        f"SELECT {konto_col}, {buch_date_col}, betrag FROM {buch_table} "
        f"WHERE {buch_date_col} BETWEEN CONVERT(date, '{df_str}') AND CONVERT(date, '{dt_str}')"
    )

    orders = pd.concat(pd.read_sql(q_order, eng_src, chunksize=500_000), ignore_index=True)
    buch = pd.concat(pd.read_sql(q_buch, eng_src, chunksize=500_000), ignore_index=True)
    return agg, orders, buch

# ----------------------------------
# Labeling
# ----------------------------------
def build_labels(
    agg: pd.DataFrame,
    orders: pd.DataFrame,
    buch: pd.DataFrame,
    *,
    horizon: int = 7,
    min_einzahlung: float = 100.0,
    order_status_col: Optional[str] = None,
    order_exec_values: Tuple[str, ...] = ("EXECUTED", "FILLED", "AUSGEFÜHRT"),
    buch_typ_col: Optional[str] = None,
    buch_credit_values: Tuple[str, ...] = ("Einzahlung", "Gutschrift"),
) -> pd.DataFrame:
    agg = agg.rename(columns={"Score_Datum": "stichtag"})
    orders = orders.rename(columns={"Orderdatum": "Orderdatum"})
    buch = buch.rename(columns={"Buchungdatum": "Buchungdatum", "betrag": "betrag"})

    _ensure_datetime(agg, ["stichtag"])
    _ensure_datetime(orders, ["Orderdatum"])
    _ensure_datetime(buch, ["Buchungdatum"])

    if order_status_col and order_status_col in orders.columns:
        orders = orders[orders[order_status_col].astype(str).str.upper().isin([v.upper() for v in order_exec_values])]

    if buch_typ_col and buch_typ_col in buch.columns:
        buch_credit = buch[buch[buch_typ_col].astype(str).isin(buch_credit_values)].copy()
    else:
        buch_credit = buch[buch["betrag"].astype(float) > 0].copy()

    einzahlungen = (
        buch_credit
        .groupby(["Kontoreferenz", pd.Grouper(key="Buchungdatum", freq="D")])["betrag"].sum()
        .reset_index()
        .rename(columns={"Buchungdatum": "event_date", "betrag": "einzahlung_sum"})
    )
    einzahlungen = einzahlungen[einzahlungen["einzahlung_sum"] >= float(min_einzahlung)]

    order_events = (
        orders
        .groupby(["Depotnummer", pd.Grouper(key="Orderdatum", freq="D")])
        .size()
        .reset_index(name="order_count")
        .rename(columns={"Orderdatum": "event_date"})
    )
    # Buchungen auf Depot-Schlüssel bringen (falls Konto!=Depot, hier Mapping ergänzen)
    if "Kontoreferenz" in einzahlungen.columns and "Depotnummer" not in einzahlungen.columns:
        einzahlungen = einzahlungen.rename(columns={"Kontoreferenz": "Depotnummer"})
    events = pd.concat([
        order_events[["Depotnummer", "event_date"]],
        einzahlungen[["Depotnummer", "event_date"]]
    ], ignore_index=True).drop_duplicates()

    left = agg[["Depotnummer", "stichtag"]].copy()
    has_future_event = _find_future_event_within(left, events, horizon)

    label_col = f"label_invest_T{horizon}"
    agg[label_col] = has_future_event.astype(int)
    agg["horizon_tage"] = horizon
    return agg

# ----------------------------------
# Persistenz (UPSERT für mehrere Horizonte)
# ----------------------------------

def save_to_sql_upsert(df: pd.DataFrame, out_table: str, horizon: int):
    """
    Schreibt Labels in CD_FA_LABELS, ohne andere Horizonte zu überschreiben.
    Wichtig: KEIN REPLACE der Zieltabelle, nur MERGE/UPSERT.
    Erwartete Zieltabelle:
      CD_FA_LABELS(Depotnummer, stichtag,
                   label_invest_T7 INT NULL, label_invest_T14 INT NULL,
                   created_at DATETIME2, PK(Depotnummer, stichtag))
    """
    from db_connection import get_engine
    eng = get_engine()
    label_col = f"label_invest_T{horizon}"

    if label_col not in df.columns:
        raise ValueError(f"Erwartete Spalte {label_col} nicht im DataFrame vorhanden.")

    # Minimales Staging mit passender Labelspalte
    stg_name = f"_stg_{out_table}_{horizon}"
    stg_df = df[["Depotnummer", "stichtag", label_col]].copy()

    with eng.begin() as conn:
        # 0) Basistabelle sicherstellen (ohne Labelspalten)
        ddl = f"""
        IF OBJECT_ID('{out_table}', 'U') IS NULL
        BEGIN
            CREATE TABLE {out_table} (
                Depotnummer  VARCHAR(64) NOT NULL,
                stichtag     DATE        NOT NULL,
                created_at   DATETIME2   NOT NULL DEFAULT SYSUTCDATETIME(),
                CONSTRAINT PK_{out_table} PRIMARY KEY (Depotnummer, stichtag)
            );
        END
        """
        conn.exec_driver_sql(ddl)

        # 1) Zielspalte anlegen, falls fehlt
        alter_sql = f"""
        IF COL_LENGTH('{out_table}', '{label_col}') IS NULL
            ALTER TABLE {out_table} ADD {label_col} INT NULL;
        """
        conn.exec_driver_sql(alter_sql)

        # 2) Staging schreiben (ersetzbar)
        stg_df.to_sql(stg_name, conn, if_exists="replace", index=False)

        # 3) MERGE Upsert
        merge_sql = f"""
        MERGE {out_table} AS tgt
        USING {stg_name} AS src
          ON tgt.Depotnummer = src.Depotnummer
         AND tgt.stichtag    = src.stichtag
        WHEN MATCHED THEN
          UPDATE SET tgt.{label_col} = src.{label_col},
                     tgt.created_at  = SYSUTCDATETIME()
        WHEN NOT MATCHED THEN
          INSERT (Depotnummer, stichtag, {label_col}, created_at)
          VALUES (src.Depotnummer, src.stichtag, src.{label_col}, SYSUTCDATETIME());
        """
        conn.exec_driver_sql(merge_sql)

        # 4) Staging entfernen
        conn.exec_driver_sql(f"DROP TABLE {stg_name};")

# ----------------------------------
# CLI
# ----------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build labels for ML invest readiness")
    p.add_argument("--source", choices=["mssql"], required=True)
    p.add_argument("--agg_table", default="CD_FA_AGG_KENNZAHLEN")
    p.add_argument("--orders_table", default="CD_ORDER")
    p.add_argument("--buch_table", default="CD_BUCHUNG")
    p.add_argument("--start_date", help="YYYY-MM-DD – optional Startdatum für AGG/Events")
    p.add_argument("--end_date", help="YYYY-MM-DD – optional Enddatum für AGG/Events")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--min_einzahlung", type=float, default=100.0)
    p.add_argument("--out_table", help="SQL-Tabelle für Labels (z. B. CD_FA_LABELS)")
    return p.parse_args()

def main():
    args = parse_args()
    agg, orders, buch = load_mssql_via_internal(
        args.agg_table, args.orders_table, args.buch_table,
        start_date=args.start_date, end_date=args.end_date, horizon=args.horizon
    )
    labeled = build_labels(agg, orders, buch, horizon=args.horizon, min_einzahlung=args.min_einzahlung)
    if args.out_table:
        save_to_sql_upsert(labeled, args.out_table, args.horizon)
    label_col = f"label_invest_T{args.horizon}"
    print(f"Fertig. Datensätze: {len(labeled)}, Positiv-Rate: {labeled[label_col].mean():.4f}")

if __name__ == "__main__":
    main()
