# eval_buy_de000a14uww2.py  (Registry=CD_FA_MODEL_REGISTRY, VARBINARY-Blob)
from __future__ import annotations
import argparse, io, os, json, sys
import numpy as np, pandas as pd, joblib
from typing import Dict, Tuple, List
from sqlalchemy import text
from sklearn.metrics import average_precision_score
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from db_connection import get_engine

MODEL_NAME = os.getenv("FA_MODEL_NAME", "buy-DE000A14UWW2")
REGISTRY_TABLE = os.getenv("FA_MODEL_REGISTRY_TABLE", "dbo.CD_FA_MODEL_REGISTRY")
AGG_TABLE      = "dbo.CD_FA_AGG_BUY_DE000A14UWW2"
LABELS_TABLE   = "dbo.CD_FA_LABELS_DE000A14UWW2"
EVAL_TABLE     = os.getenv("FA_MODEL_EVAL_TABLE", "dbo.CD_FA_MODEL_EVAL")

def get_engine():
    from db_connection import get_engine
    return get_engine()

# -------- Registry: aktives oder spezifisches Bundle laden --------
def fetch_bundle(engine, horizon: int, version: str | None) -> Tuple[str, dict]:
    if version:
        q = text(f"""
            SELECT TOP 1 model_id, version, artifact
            FROM {REGISTRY_TABLE}
            WHERE model_name=:n AND horizon_days=:h AND version=:v
            ORDER BY created_at DESC, model_id DESC
        """)
        df = pd.read_sql(q, engine, params={"n": MODEL_NAME, "h": horizon, "v": version})
    else:
        # 1) aktiv bevorzugen
        q1 = text(f"""
            SELECT TOP 1 model_id, version, artifact
            FROM {REGISTRY_TABLE}
            WHERE model_name=:n AND horizon_days=:h AND is_active=1
            ORDER BY created_at DESC, model_id DESC
        """)
        df = pd.read_sql(q1, engine, params={"n": MODEL_NAME, "h": horizon})
        if df.empty:
            # 2) Fallback: neueste
            q2 = text(f"""
                SELECT TOP 1 model_id, version, artifact
                FROM {REGISTRY_TABLE}
                WHERE model_name=:n AND horizon_days=:h
                ORDER BY created_at DESC, model_id DESC
            """)
            df = pd.read_sql(q2, engine, params={"n": MODEL_NAME, "h": horizon})
    if df.empty:
        raise RuntimeError(f"Kein Registry-Eintrag für {MODEL_NAME}, T={horizon}, version={version}.")
    ver = str(df["version"].iloc[0])
    blob = df["artifact"].iloc[0]
    if blob is None:
        raise RuntimeError(f"Artifact-Blob fehlt (T={horizon}, version={ver}).")
    bundle = joblib.load(io.BytesIO(blob))  # erwartet: {'model','calibrator','features',...}
    for k in ("model", "calibrator", "features"):
        if k not in bundle:
            raise RuntimeError(f"Bundle unvollständig: '{k}' fehlt (T={horizon}, version={ver}).")
    return ver, bundle

# -------- Daten laden (mit Pushdown + Pre_First_Target) --------
def load_frame(engine, start: str, end: str, horizon: int) -> pd.DataFrame:
    label_col = f"label_T{horizon}"
    q = text(f"""
        SELECT a.*, l.{label_col}
        FROM {AGG_TABLE} a
        JOIN {LABELS_TABLE} l
          ON l.Depotnummer = a.Depotnummer AND l.Stichtag = a.Score_Datum
        WHERE a.Pre_First_Target=1
          AND a.Score_Datum BETWEEN :s AND :e
          AND l.{label_col} IS NOT NULL
    """)
    df = pd.read_sql(q, engine, params={"s": start, "e": end})
    df = df.rename(columns={"Score_Datum": "stichtag"})
    df["stichtag"] = pd.to_datetime(df["stichtag"])
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    return df

def ensure_feature_matrix(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    X = df.copy()
    miss = [c for c in features if c not in X.columns]
    for c in miss: X[c] = 0.0
    return X[features].astype(float).fillna(0.0).values

def score_with_bundle(bundle: dict, X_df: pd.DataFrame) -> np.ndarray:
    feats = bundle["features"]
    X = ensure_feature_matrix(X_df, feats)
    p_raw = bundle["model"].predict_proba(X)[:, 1]
    iso = bundle["calibrator"]
    return iso.transform(p_raw)

# -------- Metrics --------
def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    n = max(1, int(np.ceil(len(y_score)*k)))
    idx = np.argsort(-y_score)[:n]
    return float(np.mean(y_true[idx]))

def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    base = float(np.mean(y_true)) or 1e-9
    return precision_at_k(y_true, y_score, k) / base

# -------- Eval-Tabelle sicherstellen --------
def ensure_eval_table(engine):
    ddl = f"""
    IF OBJECT_ID('{EVAL_TABLE}','U') IS NULL
    BEGIN
      CREATE TABLE {EVAL_TABLE}(
        id BIGINT IDENTITY(1,1) PRIMARY KEY,
        model_name   VARCHAR(128) NOT NULL,
        isin         VARCHAR(32)  NOT NULL,
        horizon_days INT          NOT NULL,
        version      VARCHAR(128) NOT NULL,
        start_date   DATE         NOT NULL,
        end_date     DATE         NOT NULL,
        n_samples    INT          NOT NULL,
        pos_rate     FLOAT        NOT NULL,
        ap_test      FLOAT        NOT NULL,
        p_at_1pct    FLOAT        NULL,
        p_at_3pct    FLOAT        NULL,
        p_at_5pct    FLOAT        NULL,
        lift_1pct    FLOAT        NULL,
        lift_3pct    FLOAT        NULL,
        lift_5pct    FLOAT        NULL,
        created_at   DATETIME2    NOT NULL DEFAULT SYSUTCDATETIME()
      );
      CREATE INDEX IX_MODEL_EVAL_Q ON {EVAL_TABLE}(model_name, horizon_days, version, created_at);
    END
    """
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)

# -------- Single-Horizon Eval --------
def eval_one(engine, horizon: int, start: str, end: str, version: str | None):
    ver, bundle = fetch_bundle(engine, horizon, version)
    df = load_frame(engine, start, end, horizon)
    y = df[f"label_T{horizon}"].values
    p = score_with_bundle(bundle, df)

    ap = float(average_precision_score(y, p))
    p1, p3, p5 = (precision_at_k(y, p, k) for k in (0.01, 0.03, 0.05))
    l1, l3, l5 = (lift_at_k(y, p, k) for k in (0.01, 0.03, 0.05))

    print(f"[EVAL] T={horizon} {start}..{end}  n={len(df)}  pos={int(y.sum())} "
          f"pos_rate={y.mean():.6f}  AP={ap:.6f}  "
          f"P@1/3/5={p1:.6f}/{p3:.6f}/{p5:.6f}  Lift@1/3/5={l1:.2f}/{l3:.2f}/{l5:.2f}")

    ensure_eval_table(engine)
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {EVAL_TABLE}
              (model_name, isin, horizon_days, version, start_date, end_date,
               n_samples, pos_rate, ap_test, p_at_1pct, p_at_3pct, p_at_5pct,
               lift_1pct, lift_3pct, lift_5pct)
            VALUES (:m,:i,:h,:v,:s,:e,:n,:pr,:ap,:p1,:p3,:p5,:l1,:l3,:l5)
        """), {
            "m": MODEL_NAME, "i": "DE000A14UWW2", "h": horizon, "v": ver,
            "s": start, "e": end, "n": int(len(df)), "pr": float(y.mean()),
            "ap": ap, "p1": p1, "p3": p3, "p5": p5, "l1": l1, "l3": l3, "l5": l5
        })

# -------- CLI --------
def parse_args():
    ap = argparse.ArgumentParser("Eval A14UWW2 First-Buy")
    ap.add_argument("--horizons", default="7,14,30")
    ap.add_argument("--start", default="2025-07-01")
    ap.add_argument("--end", default="2025-08-20")
    ap.add_argument("--version", default=None, help="optional: feste Modellversion (sonst aktive)")
    return ap.parse_args()

def main():
    args = parse_args()
    eng = get_engine()
    for h in [int(x) for x in args.horizons.split(",") if x]:
        eval_one(eng, h, args.start, args.end, args.version)

if __name__ == "__main__":
    main()
