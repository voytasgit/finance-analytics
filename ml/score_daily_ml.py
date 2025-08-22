#!/usr/bin/env python3
"""
Daily-Scorer für ML-Investitionsbereitschaft
- Lädt heutige (oder per --run_date übergebene) AGG-Daten aus CD_FA_AGG_KENNZAHLEN
- Lädt Modell-Bundle (model_invest.pkl)
- Scored p_invest, baut Score 0..100 und Rang innerhalb des Laufs
- Schreibt/Upsert nach CD_FA_INVEST_BEREITSCHAFT_ML

Beispiel:
python score_daily_ml.py --run_date 2025-08-19 --model_path model_invest.pkl

Daily-Scorer für ML-Investitionsbereitschaft
- Lädt heutige (oder per --run_date übergebene) AGG-Daten aus CD_FA_AGG_KENNZAHLEN
- Lädt Modell-Bundle (model_invest.pkl)
- Scored p_invest, baut Score 0..100 und Rang innerhalb des Laufs
- Schreibt/Upsert nach CD_FA_INVEST_BEREITSCHAFT_ML

Beispiel:
python score_daily_ml.py --run_date 2025-08-19 --model_path model_invest.pkl

Daily-Scorer ist drin (Canvas links: score_daily_ml.py).

So nutzt du ihn
cd C:\apano_Dienste\finance-analytics\ml

REM Heute scor(en)
python score_daily_ml.py --model_path model_invest.pkl

REM Oder explizites Datum (z. B. Backfill für 2025-08-19)
python score_daily_ml.py --run_date 2025-08-19 --model_path model_invest.pkl

Was passiert

zieht AGG für Score_Datum = run_date aus CD_FA_AGG_KENNZAHLEN

lädt model_invest.pkl → nimmt genau die Featureliste, fügt fehlende Features als 0 hinzu (mit Warnung)

schreibt/aktualisiert in CD_FA_INVEST_BEREITSCHAFT_ML via MERGE Upsert

Felder: run_date, Depotnummer, p_invest, score(0..100), rank_in_run, model_version, featureset_version, created_at

legt Tabelle + Index automatisch an, falls nicht vorhanden

Tipp

Häng den Aufruf an deine bestehende Pipeline nach der AGG-Befüllung:

run_pipeline.py → nach CD_FA_AGG_KENNZAHLEN Schritt score_daily_ml.py ausführen

Wenn du getrennte Modelle für T7/T14 hast, speichere sie z. B. als model_invest_T7.pkl und model_invest_T14.pkl und rufe den Scorer 2× auf (mit je eigener pred_table oder zusätzlicher Spalte).


Daily-Scorer für ML-Investitionsbereitschaft
- Lädt heutige (oder per --run_date übergebene) AGG-Daten aus CD_FA_AGG_KENNZAHLEN
- Lädt Modell-Bundle (model_invest.pkl)
- Scored p_invest, baut Score 0..100 und Rang innerhalb des Laufs
- Schreibt/Upsert nach CD_FA_INVEST_BEREITSCHAFT_ML

Beispiel:
python score_daily_ml.py --run_date 2025-08-19 --model_path model_invest.pkl
"""
from __future__ import annotations
import argparse
from datetime import date, datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression  # <--- HIER

# ------------------------------------------------------------
# DB Helpers
# ------------------------------------------------------------

class CalibratedModel:
    def __init__(self, model, iso):
        self.model = model
        self.iso = iso
    def predict_proba(self, X):
        p = self.model.predict_proba(X)[:, 1]
        p_cal = self.iso.transform(p)
        return np.vstack([1 - p_cal, p_cal]).T
    @property
    def feature_importances_(self):
        return getattr(self.model, 'feature_importances_', None)

def get_engine_safe():
    try:
        from db_connection import get_engine
    except ImportError:
        import sys, pathlib
        parent = pathlib.Path(__file__).resolve().parent.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        from db_connection import get_engine
    return get_engine()

# ------------------------------------------------------------
# Load today's AGG
# ------------------------------------------------------------

def load_agg_for_date(eng, agg_table: str, run_date: str) -> pd.DataFrame:
    q = f"""
        SELECT *
        FROM {agg_table}
        WHERE Score_Datum = CONVERT(date, '{run_date}')
    """
    df = pd.read_sql(q, eng)
    if 'Score_Datum' in df.columns:
        df = df.rename(columns={'Score_Datum': 'stichtag'})
    return df

# ------------------------------------------------------------
# Feature selection consistent with training
# ------------------------------------------------------------

IDENT_COLS = {"Depotnummer", "stichtag", "Score_Datum", "created_at"}

def select_feature_columns(df: pd.DataFrame, features_from_model: list[str]) -> list[str]:
    # Verwende exakt die Featureliste aus dem Modellbundle, falls vorhanden
    cols = [c for c in features_from_model if c in df.columns]
    missing = [c for c in features_from_model if c not in df.columns]
    if missing:
        print("[WARN] folgende Features fehlen in AGG und werden mit 0 ersetzt:", missing)
        for m in missing:
            df[m] = 0.0
        cols = features_from_model
    return cols

# ------------------------------------------------------------
# Upsert writer
# ------------------------------------------------------------

def ensure_pred_table(eng, table: str):
    ddl = f"""
    IF OBJECT_ID('{table}', 'U') IS NULL
    BEGIN
        CREATE TABLE {table} (
            run_date        DATE        NOT NULL,
            Depotnummer     VARCHAR(64) NOT NULL,
            p_invest        FLOAT       NOT NULL,
            score           INT         NOT NULL,
            rank_in_run     INT         NOT NULL,
            model_version   VARCHAR(32) NOT NULL,
            featureset_version VARCHAR(32) NULL,
            created_at      DATETIME2   NOT NULL DEFAULT SYSUTCDATETIME(),
            PRIMARY KEY (run_date, Depotnummer)
        );
        CREATE INDEX IX_{table}_rank ON {table}(run_date, rank_in_run);
    END
    """
    with eng.begin() as conn:
        conn.exec_driver_sql(ddl)


def upsert_predictions(eng, table: str, df: pd.DataFrame):
    stg = f"_stg_{table}"
    with eng.begin() as conn:
        df.to_sql(stg, conn, if_exists="replace", index=False)
        merge = f"""
        MERGE {table} AS tgt
        USING {stg} AS src
          ON tgt.run_date = src.run_date AND tgt.Depotnummer = src.Depotnummer
        WHEN MATCHED THEN UPDATE SET
            tgt.p_invest = src.p_invest,
            tgt.score = src.score,
            tgt.rank_in_run = src.rank_in_run,
            tgt.model_version = src.model_version,
            tgt.featureset_version = src.featureset_version,
            tgt.created_at = SYSUTCDATETIME()
        WHEN NOT MATCHED THEN INSERT
            (run_date, Depotnummer, p_invest, score, rank_in_run, model_version, featureset_version, created_at)
            VALUES (src.run_date, src.Depotnummer, src.p_invest, src.score, src.rank_in_run, src.model_version, src.featureset_version, SYSUTCDATETIME());
        DROP TABLE {stg};
        """
        conn.exec_driver_sql(merge)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Score daily ML invest readiness")
    p.add_argument("--run_date", help="YYYY-MM-DD. Default = heute", default=None)
    p.add_argument("--agg_table", default="CD_FA_AGG_KENNZAHLEN")
    p.add_argument("--pred_table", default="CD_FA_INVEST_BEREITSCHAFT_ML")
    p.add_argument("--model_path", default="model_invest.pkl")
    p.add_argument("--featureset_version", default="fs1.0")
    return p.parse_args()


def main():
    args = parse_args()
    run_date = args.run_date or date.today().isoformat()

    # Load model bundle
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    model_features = bundle.get("features", [])
    model_version = bundle.get("version", "v1.0")

    eng = get_engine_safe()
    ensure_pred_table(eng, args.pred_table)

    # Load today's AGG
    agg = load_agg_for_date(eng, args.agg_table, run_date)
    if agg.empty:
        print(f"[INFO] Keine AGG-Daten für {run_date} gefunden. Abbruch.")
        return

    # Compose feature matrix
    feat_cols = select_feature_columns(agg, model_features)
    X = agg[feat_cols].fillna(0.0).values

    # Predict
    p = model.predict_proba(X)[:, 1]
    score = np.rint(p * 100).astype(int)
    # rank: 1 = best
    rank = (-p).argsort().argsort() + 1

    out = pd.DataFrame({
        "run_date": run_date,
        "Depotnummer": agg["Depotnummer"].astype(str),
        "p_invest": p,
        "score": score,
        "rank_in_run": rank,
        "model_version": model_version,
        "featureset_version": args.featureset_version,
    })

    upsert_predictions(eng, args.pred_table, out)

    print(f"Fertig: {len(out)} Depots gescored für {run_date}. Top1% p_invest~{np.quantile(p, 0.99):.3f}")


if __name__ == "__main__":
    main()
