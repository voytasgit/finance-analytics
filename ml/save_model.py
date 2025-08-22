# save_model.py
import joblib, json, hashlib, io
from sqlalchemy import text
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from db_connection import get_engine

def blob_of(obj) -> bytes:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def register_model(bundle, name, version, horizon, isin=None, metrics=None, activate=True):
    eng = get_engine()
    blob = blob_of(bundle)
    h = sha256(blob)
    with eng.begin() as c:
        if activate:
            c.execute(text("""
              UPDATE dbo.CD_FA_MODEL_REGISTRY
              SET is_active=0
              WHERE model_name=:n AND horizon_days=:h
            """), {"n": name, "h": horizon})
        c.execute(text("""
          INSERT INTO dbo.CD_FA_MODEL_REGISTRY
            (model_name,version,isin,horizon_days,sha256,metrics_json,artifact,is_active)
          VALUES (:n,:v,:i,:h,:s,:m,:a,:act)
        """), {
          "n": name, "v": version, "i": isin, "h": horizon, "s": h,
          "m": json.dumps(metrics or {}), "a": blob, "act": 1 if activate else 0
        })

def load_active_model(name, horizon):
    from sqlalchemy import text
    from db_connection import get_engine
    import joblib, io
    eng = get_engine()
    # row = eng.execute(text("""
    #   SELECT TOP 1 artifact FROM dbo.CD_FA_MODEL_REGISTRY
    #   WHERE model_name=:n AND horizon_days=:h AND is_active=1
    #   ORDER BY model_id DESC
    # """), {"n": name, "h": horizon}).fetchone()
    with eng.connect() as conn:
        row = conn.execute(
            text("""
              SELECT TOP 1 artifact 
              FROM dbo.CD_FA_MODEL_REGISTRY
              WHERE model_name=:n AND horizon_days=:h AND is_active=1
              ORDER BY model_id DESC
            """),
            {"n": name, "h": horizon}
        ).fetchone()
    if not row: raise RuntimeError("No active model registered.")
    return joblib.load(io.BytesIO(row[0]))
