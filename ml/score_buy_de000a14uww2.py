# -*- coding: utf-8 -*-
# score_buy_de000a14uww2.py
from __future__ import annotations
import io, os, json, joblib, sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sqlalchemy import text
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from db_connection import get_engine
# am Kopf:
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD oder 'latest'", default="latest")
    return ap.parse_args()

# lade AGG für Datum:
def load_agg_for_date(engine, date_str: str):
    if date_str.lower()=="latest":
        qd = text(f"SELECT MAX(Score_Datum) AS d FROM {AGG_TABLE}")
        d = pd.read_sql(qd, engine)["d"].iloc[0]
    else:
        d = pd.to_datetime(date_str).date()
    q = text(f"SELECT * FROM {AGG_TABLE} WHERE Score_Datum=:d AND Pre_First_Target=1")
    df = pd.read_sql(q, engine, params={"d": d})
    df["Score_Datum"] = pd.to_datetime(df["Score_Datum"]).dt.date
    return d, df

MODEL_NAME = os.getenv("FA_MODEL_NAME", "buy-DE000A14UWW2")
REGISTRY_TABLE = os.getenv("FA_MODEL_REGISTRY_TABLE", "dbo.CD_FA_MODEL_REGISTRY")
AGG_TABLE  = "dbo.CD_FA_AGG_BUY_DE000A14UWW2"
OUT_TABLE  = "dbo.CD_FA_BUY_DE000A14UWW2_ML"

def get_engine():
    from db_connection import get_engine
    return get_engine()

# -- Registry (DB-Blob)
def fetch_active_bundle(engine, horizon: int) -> Tuple[str, dict]:
    # 1) aktiv bevorzugen
    q_active = text(f"""
        SELECT TOP 1 model_id, version, artifact
        FROM {REGISTRY_TABLE}
        WHERE model_name=:n AND horizon_days=:h AND is_active=1
        ORDER BY created_at DESC, model_id DESC
    """)
    df = pd.read_sql(q_active, engine, params={"n": MODEL_NAME, "h": horizon})
    if df.empty:
        # 2) Fallback: neueste Version (nicht-aktiv)
        q_latest = text(f"""
            SELECT TOP 1 model_id, version, artifact
            FROM {REGISTRY_TABLE}
            WHERE model_name=:n AND horizon_days=:h
            ORDER BY created_at DESC, model_id DESC
        """)
        df = pd.read_sql(q_latest, engine, params={"n": MODEL_NAME, "h": horizon})
        if df.empty:
            raise RuntimeError(f"Kein Registry-Eintrag für {MODEL_NAME}, T={horizon}.")
    version = str(df["version"].iloc[0])
    blob = df["artifact"].iloc[0]
    if blob is None:
        raise RuntimeError(f"Artifact-Blob fehlt (T={horizon}, version={version}).")
    bundle = joblib.load(io.BytesIO(blob))  # enthält: model, calibrator (Isotonic), features, ...
    for k in ("model","calibrator","features"):
        if k not in bundle:
            raise RuntimeError(f"Bundle unvollständig: '{k}' fehlt (T={horizon}, version={version}).")
    return version, bundle

def load_latest_agg(engine) -> Tuple[pd.Timestamp, pd.DataFrame]:
    d = pd.read_sql(text(f"SELECT MAX(Score_Datum) d FROM {AGG_TABLE}"), engine)["d"].iloc[0]
    if pd.isna(d):
        raise RuntimeError("Kein Score_Datum in AGG gefunden.")
    q = text(f"""
        SELECT * FROM {AGG_TABLE}
        WHERE Score_Datum=:d AND Pre_First_Target=1
    """)
    df = pd.read_sql(q, engine, params={"d": d})
    df["Score_Datum"] = pd.to_datetime(df["Score_Datum"]).dt.date
    return pd.to_datetime(d), df

def ensure_feature_matrix(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    X = df.copy()
    miss = [c for c in features if c not in X.columns]
    for c in miss: X[c] = 0.0
    X = X[features].astype(float).fillna(0.0)
    return X.values

def score_with_bundle(df_agg: pd.DataFrame, bundle: dict) -> np.ndarray:
    feats = bundle["features"]
    X = ensure_feature_matrix(df_agg, feats)
    p_raw = bundle["model"].predict_proba(X)[:, 1]
    iso = bundle["calibrator"]
    return iso.transform(p_raw)

def write_scores(engine, rows: List[Tuple], score_date):
    with engine.begin() as sa_conn:
        sa_conn.exec_driver_sql(
            f"DELETE FROM {OUT_TABLE} WHERE Score_Datum = ?",
            (pd.to_datetime(score_date).date(),)
        )
        raw = sa_conn.connection
        cur = raw.cursor()
        cur.fast_executemany = True
        cur.executemany(f"""
            INSERT INTO {OUT_TABLE}(
                Score_Datum, Depotnummer, p_buy_7d, p_buy_14d, p_buy_30d, Model_Version, Created_At
            ) VALUES (?, ?, ?, ?, ?, ?, SYSUTCDATETIME())
        """, rows)

def main():
    args = parse_args()
    eng = get_engine()
    score_date, agg = load_agg_for_date(eng, args.date)
    #score_date, agg = load_latest_agg(eng)

    versions, preds = {}, {}
    for T in (7, 14, 30):
        ver, bundle = fetch_active_bundle(eng, T)
        versions[T] = ver
        preds[T] = score_with_bundle(agg, bundle)

    # NEU: sanft clippen (nur nach oben)
    p7  = preds[7]
    p14 = np.maximum(preds[14], p7)
    p30 = np.maximum(preds[30], p14)
    preds[7], preds[14], preds[30] = p7, p14, p30

    # Optional: Monotonie sanft erzwingen (auskommentiert, bei Bedarf aktivieren)
    # p7, p14, p30 = preds[7], np.maximum(preds[14], preds[7]), None
    # p30 = np.maximum(preds[30], p14)
    # preds[14], preds[30] = p14, p30

    version_str = ";".join([f"T{T}:{versions[T]}" for T in (7,14,30)])

    rows = []
    depots = agg["Depotnummer"].astype(str).values
    for i in range(len(agg)):
        rows.append((
            score_date.date(), depots[i],
            float(preds[7][i]),
            float(preds[14][i]),
            float(preds[30][i]),
            version_str
        ))
    write_scores(eng, rows, score_date)
    print(f"[SCORED] {len(rows)} Depots @ {score_date.date()} -> {OUT_TABLE}")
    print(f"Versions (aktiv): {version_str}")

if __name__ == "__main__":
    main()
