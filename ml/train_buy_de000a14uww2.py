"""
python train_buy_de000a14uww2.py ^
  --horizons 7,14,30 ^
  --train_start 2024-01-01 --train_end 2025-06-30 ^
  --test_start  2025-07-01 --test_end  2025-08-20


1) Mini-Sweep (robust, kalibriert)
:: Beispiel: 3×3 Raster – Depth × LR (mit mehr Trees)
python train_buy_de000a14uww2.py --horizons 7,14,30 ^
  --train_start 2024-01-01 --train_end 2025-06-30 ^
  --test_start  2025-07-01 --test_end  2025-08-20 ^
  --max_depth 5 --learning_rate 0.05 --n_estimators 1200

python train_buy_de000a14uww2.py --horizons 7,14,30 ^
  --train_start 2024-01-01 --train_end 2025-06-30 ^
  --test_start  2025-07-01 --test_end  2025-08-20 ^
  --max_depth 6 --learning_rate 0.03 --n_estimators 1600

python train_buy_de000a14uww2.py --horizons 7,14,30 ^
  --train_start 2024-01-01 --train_end 2025-06-30 ^
  --test_start  2025-07-01 --test_end  2025-08-20 ^
  --max_depth 4 --learning_rate 0.06 --n_estimators 1000


Tipps:

Bei ultra-seltenen Labels eher: kleinere learning_rate, mehr n_estimators, max_depth 4–6, colsample_bytree/subsample 0.8–0.9.

scale_pos_weight setzen wir bereits automatisch.

Kurzfassung: Deine drei Läufe zeigen, dass P@1 % / P@5 % in allen Varianten identisch sind (also gleich viele Treffer im Top-K). Die Unterschiede liegen nur in AP(test) (Qualität im „Long Tail“). Für Kampagnen (Top-1–5 %) ist das egal; fürs generelle Ranking ist der höchste AP leicht zu bevorzugen. Einen „ohne Parameter“-Run brauchst du nicht extra neu zu starten.

Empfehlung (pro Horizont den besten AP nehmen)

T=7 → bester AP = 0.00183 → Version T7-v7-iso90d (max_depth=6, lr=0.03, n_estimators=1600)

T=14 → bester AP = 0.00663 → Version T14-v7-iso90d (max_depth=4, lr=0.06, n_estimators=1000)

T=30 → bester AP = 0.00215 → Version T30-v5-iso90d (max_depth=4, lr=0.06, n_estimators=1000)

Da P@k gleich ist, kannst du aus Betriebsgründen auch überall die leichtere Variante (depth 4, lr 0.06, 1000 Trees) nehmen. Ich würde aber die oben genannten „AP-Gewinner“ aktivieren.


"""
from __future__ import annotations
import argparse, json, os, hashlib, time, io, pathlib, sys
from dataclasses import dataclass
from typing import List, Tuple, Dict
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from db_connection import get_engine

# =========================
# Artefakt-Verzeichnis
# =========================
ARTIFACT_DIR = os.environ.get("FA_ART_DIR", r"C:\apano_Dienste\ML")
pathlib.Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# Konstanten/Defaults
# =========================
MODEL_NAME = os.environ.get("FA_MODEL_NAME", "buy-DE000A14UWW2")  # passt zu eurer Registry
AGG_TABLE   = "dbo.CD_FA_AGG_BUY_DE000A14UWW2"
LABELS_TABLE= "dbo.CD_FA_LABELS_DE000A14UWW2"
REGISTRY_TABLE = os.environ.get("FA_MODEL_REGISTRY_TABLE", "dbo.CD_FA_MODEL_REGISTRY")  # eure Tabelle

def get_engine():
    from db_connection import get_engine
    return get_engine()

# =========================
# Laden & Feature-Handling
# =========================
def load_period(engine, start: str, end: str, horizon: int) -> pd.DataFrame:
    label_col = f"label_T{horizon}"
    q = text(f"""
        SELECT a.*, l.{label_col}
        FROM {AGG_TABLE} a
        JOIN {LABELS_TABLE} l
          ON l.Depotnummer = a.Depotnummer
         AND l.Stichtag    = a.Score_Datum
        WHERE a.Pre_First_Target = 1
          AND a.Score_Datum BETWEEN :s AND :e
          AND l.{label_col} IS NOT NULL
    """)
    df = pd.read_sql(q, engine, params={"s": start, "e": end})
    if "Score_Datum" in df.columns:
        df = df.rename(columns={"Score_Datum":"stichtag"})
    return df

NON_FEATURE = {"Depotnummer","stichtag","First_Target_Buy"}
def coerce_numeric(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    # 1) bool/nullable-bool -> int
    import pandas.api.types as ptypes
    for c in d.columns:
        if c in NON_FEATURE or c == label_col:
            continue
        if ptypes.is_bool_dtype(d[c]):
            d[c] = d[c].astype("uint8")

    # 2) object -> numeric (de/US Delimiter robust)
    obj_cols = [c for c in d.columns if d[c].dtype == object and c not in NON_FEATURE and c != label_col]
    for c in obj_cols:
        s = (
            d[c].astype(str)
                .str.replace(r"\s+", "", regex=True)  # Whitespaces
                .str.replace(".", "", regex=False)     # Tausenderpunkt
                .str.replace(",", ".", regex=False)    # Komma -> Punkt
        )
        d[c] = pd.to_numeric(s, errors="coerce")

    # 3) Nur NUMERISCHE Spalten bereinigen/fillna (vermeidet FutureWarning)
    num_cols = d.select_dtypes(include=[np.number]).columns
    d[num_cols] = d[num_cols].replace([np.inf, -np.inf], np.nan)
    d[num_cols] = d[num_cols].fillna(0.0)

    # 4) Objekttypen ggf. präzisieren (keine stillen Downcasts)
    d = d.infer_objects(copy=False)

    feat_cols = [c for c in d.select_dtypes(include=[np.number]).columns
                 if c not in NON_FEATURE | {label_col}]
    return d, feat_cols

# =========================
# Metriken
# =========================
def precision_at_k(y_true, y_score, k: float) -> float:
    n = max(1, int(np.ceil(len(y_score)*k)))
    idx = np.argsort(-y_score)[:n]
    return float(np.mean(y_true[idx]))
def lift_at_k(y_true, y_score, k: float) -> float:
    base = float(np.mean(y_true)) or 1e-9
    return precision_at_k(y_true,y_score,k)/base

# =========================
# Registry: eure Tabelle
# =========================
# def ensure_registry_table(engine, table: str = REGISTRY_TABLE):
#     ddl = f"""
#     IF OBJECT_ID('{table}','U') IS NULL
#     BEGIN
#       CREATE TABLE {table}(
#         model_id INT IDENTITY(1,1) PRIMARY KEY,
#         model_name VARCHAR(100) NULL,
#         version    VARCHAR(50)  NULL,
#         isin       VARCHAR(12)  NULL,
#         horizon_days INT NOT NULL,
#         created_at DATETIME2(7) NULL CONSTRAINT DF_{table}_created_at DEFAULT SYSUTCDATETIME(),
#         created_by VARCHAR(100) NULL,
#         sha256     CHAR(64) NULL,
#         metrics_json NVARCHAR(MAX) NULL,
#         artifact   VARBINARY(MAX) NOT NULL,
#         is_active  BIT NOT NULL CONSTRAINT DF_{table}_is_active DEFAULT(0)
#       );
#     END
#     """
#     with engine.begin() as c:
#         c.exec_driver_sql(ddl)

def _next_version(engine, model_name: str, horizon: int, calib_days: int) -> str:
    # Nimmt laufende Nummer je (model,horizon). Version-Format: T{H}-v{N}-iso{calib_days}d
    sql = text(f"""
        SELECT COUNT(*) AS cnt
        FROM {REGISTRY_TABLE}
        WHERE model_name=:m AND horizon_days=:h AND version LIKE :likepat
    """)
    likepat = f"T{horizon}-v%-iso{calib_days}d"
    row = pd.read_sql(sql, engine, params={"m": model_name, "h": horizon, "likepat": likepat})
    n = int(row["cnt"].iloc[0]) + 1
    return f"T{horizon}-v{n}-iso{calib_days}d"

def _blob_from_bundle(bundle: dict) -> bytes:
    # Serialisiert mit joblib in Bytes (kompatibel, komprimiert)
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    return buf.getvalue()

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def upsert_cd_model_registry(engine,
                             model_name: str,
                             version: str,
                             horizon: int,
                             metrics: dict,
                             bundle_bytes: bytes,
                             created_by: str | None = None,
                             activate: bool = True,
                             isin: str = "DE000A14UWW2") -> int:
    metrics_json = json.dumps(metrics, ensure_ascii=False)
    sha = _sha256(bundle_bytes)
    with engine.begin() as c:
        # optional: ältere deaktivieren (selbes model+horizon)
        if activate:
            c.exec_driver_sql(
                f"UPDATE {REGISTRY_TABLE} SET is_active=0 WHERE model_name=? AND horizon_days=?",
                (model_name, horizon)
            )
        # insert
        c.exec_driver_sql(f"""
            INSERT INTO {REGISTRY_TABLE}
            (model_name, version, isin, horizon_days, created_by, sha256, metrics_json, artifact, is_active)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (model_name, version, isin, horizon, created_by, sha, metrics_json, bundle_bytes, 1 if activate else 0))
        new_id = pd.read_sql(f"SELECT MAX(model_id) AS mid FROM {REGISTRY_TABLE}", c)["mid"].iloc[0]
    return int(new_id)

# =========================
# Training + Kalibrierung
# =========================
@dataclass
class TrainCfg:
    horizon: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 800
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 2.0
    random_state: int = 42
    calib_days: int = 90
    out_dir: str = ARTIFACT_DIR
    registry_table: str = REGISTRY_TABLE
    created_by: str = os.environ.get("USERNAME") or os.environ.get("USER") or "ml-trainer"
    activate: bool = True

def train_one(cfg: TrainCfg) -> Dict[str,str]:
    eng = get_engine()
    # ensure_registry_table(eng, cfg.registry_table)

    # --- Daten
    df_tr = load_period(eng, cfg.train_start, cfg.train_end, cfg.horizon)
    df_te = load_period(eng, cfg.test_start,  cfg.test_end,  cfg.horizon)
    label_col = f"label_T{cfg.horizon}"
    if df_tr[label_col].sum() < 1:
        raise ValueError("Keine positiven Beispiele im Training.")

    df_tr[label_col] = pd.to_numeric(df_tr[label_col], errors="coerce").fillna(0).astype(int)
    df_te[label_col] = pd.to_numeric(df_te[label_col], errors="coerce").fillna(0).astype(int)
    df_tr["stichtag"] = pd.to_datetime(df_tr["stichtag"])
    df_te["stichtag"] = pd.to_datetime(df_te["stichtag"])
    df_tr, feat_cols = coerce_numeric(df_tr, label_col)
    df_te, _         = coerce_numeric(df_te,  label_col)

    # --- Calib-Schnitt
    cut = pd.to_datetime(cfg.train_end) - pd.Timedelta(days=cfg.calib_days)
    tr_sub = df_tr[df_tr["stichtag"] <= cut]
    cal_sub= df_tr[df_tr["stichtag"] >  cut]
    if len(tr_sub)==0 or len(cal_sub)==0:
        q80 = df_tr["stichtag"].quantile(0.8)
        tr_sub = df_tr[df_tr["stichtag"] <= q80]
        cal_sub= df_tr[df_tr["stichtag"] >  q80]

    X_tr = tr_sub[feat_cols].values; y_tr = tr_sub[label_col].values
    X_cal= cal_sub[feat_cols].values; y_cal= cal_sub[label_col].values
    X_te = df_te[feat_cols].values;  y_te = df_te[label_col].values

    # class imbalance
    pos = max(1, y_tr.sum()); neg = max(1, len(y_tr)-pos)
    spw = neg/pos

    # --- Modell
    model = XGBClassifier(
        n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth, subsample=cfg.subsample, colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda, random_state=cfg.random_state, n_jobs=-1,
        scale_pos_weight=spw, eval_metric="logloss",
    )
    model.fit(X_tr, y_tr)

    # --- Isotonic
    p_cal_raw = model.predict_proba(X_cal)[:,1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_raw, y_cal)

    # --- Test Scores + Metriken
    p_te = iso.transform(model.predict_proba(X_te)[:,1])
    ap_test = float(average_precision_score(y_te, p_te))
    p1 = float(precision_at_k(y_te, p_te, 0.01))
    p5 = float(precision_at_k(y_te, p_te, 0.05))
    l1 = float(lift_at_k(y_te, p_te, 0.01))
    l5 = float(lift_at_k(y_te, p_te, 0.05))

    # --- Artefakte auf Disk (Dateinamen eindeutig)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"{MODEL_NAME}_T{cfg.horizon}_a14uww2_{ts}"
    paths = {
        "model":      os.path.join(cfg.out_dir, f"{base}_model.pkl"),
        "calibrator": os.path.join(cfg.out_dir, f"{base}_calibrator.pkl"),
        "features":   os.path.join(cfg.out_dir, f"{base}_features.json"),
        "metrics":    os.path.join(cfg.out_dir, f"{base}_metrics.json"),
    }
    joblib.dump({"model": model, "features": feat_cols}, paths["model"])
    joblib.dump({"iso": iso}, paths["calibrator"])
    with open(paths["features"], "w", encoding="utf-8") as f:
        json.dump({"features": feat_cols}, f, indent=2)
    metrics_dict = {
        "task": f"first_buy_DE000A14UWW2_T{cfg.horizon}",
        "n_train": int(len(df_tr)), "n_test": int(len(df_te)),
        "pos_rate_test": float(y_te.mean()),
        "AP_test": float(ap_test),
        "P@1%": p1, "P@5%": p5,
        "Lift@1%": l1, "Lift@5%": l5,
        "calibration": {"method":"isotonic", "window_days": cfg.calib_days}
    }
    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    # --- DB-Bundle (VARBINARY) + Registry-Eintrag
    version = _next_version(eng, MODEL_NAME, cfg.horizon, cfg.calib_days)
    bundle = {
        "model_name": MODEL_NAME,
        "version": version,
        "horizon": cfg.horizon,
        "features": feat_cols,
        "model": model,
        "calibrator": iso,
        "train_range": (cfg.train_start, cfg.train_end),
        "test_range": (cfg.test_start, cfg.test_end),
        "created_at": ts
    }
    blob = _blob_from_bundle(bundle)
    new_id = upsert_cd_model_registry(
        engine=eng,
        model_name=MODEL_NAME,
        version=version,
        horizon=cfg.horizon,
        metrics=metrics_dict,
        bundle_bytes=blob,
        created_by=cfg.created_by,
        activate=cfg.activate,
        isin="DE000A14UWW2"
    )

    print(f"[OK] T={cfg.horizon}  AP(test)={ap_test:.5f}  "
          f"P@1%={p1:.5f}  P@5%={p5:.5f}  Lift@1%={l1:.2f}  Lift@5%={l5:.2f}")
    print(f"→ Files @ {cfg.out_dir}  |  Registry row model_id={new_id}, version={version}")
    return {"version": version, **paths}

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser("Train A14UWW2 First-Buy (XGB + Isotonic)")
    ap.add_argument("--horizons", default="7,14,30")
    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--test_start",  required=True)
    ap.add_argument("--test_end",    required=True)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--n_estimators", type=int, default=800)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--reg_lambda", type=float, default=2.0)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--calib_days", type=int, default=90)
    ap.add_argument("--out_dir", default=ARTIFACT_DIR)
    ap.add_argument("--registry_table", default=REGISTRY_TABLE)
    ap.add_argument("--created_by", default=os.environ.get("USERNAME") or os.environ.get("USER") or "ml-trainer")
    ap.add_argument("--no_activate", action="store_true", help="nicht automatisch als active markieren")
    return ap.parse_args()

def main():
    args = parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    for h in horizons:
        cfg = TrainCfg(
            horizon=h,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start,   test_end=args.test_end,
            max_depth=args.max_depth, learning_rate=args.learning_rate, n_estimators=args.n_estimators,
            subsample=args.subsample, colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda, random_state=args.random_state, calib_days=args.calib_days,
            out_dir=args.out_dir, registry_table=args.registry_table,
            created_by=args.created_by, activate=(not args.no_activate)
        )
        train_one(cfg)

if __name__ == "__main__":
    main()
