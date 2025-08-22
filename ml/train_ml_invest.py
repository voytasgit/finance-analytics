#!/usr/bin/env python3
"""
Trainiert ein ML-Modell zur Vorhersage der Investitionsbereitschaft (T-Horizont) auf Basis
von CD_FA_AGG_KENNZAHLEN + CD_FA_LABELS.

- Liest Daten direkt aus MSSQL über db_connection.get_engine()
- Zeitbasierte Splits: Train/Val auf Historie, optional separater Test-Zeitraum
- Entfernt Identifikatoren (Depotnummer, Datum) aus Features
- Modell: XGBoost (tabellarische Daten), mit Klassenungleichgewicht (scale_pos_weight)
- Kennzahlen: PR-AUC, Precision@k, Lift@k für k∈{1%,5%,10%}
- Persistiert: model_invest.pkl (inkl. Featureliste & Version), metrics.json, importances.csv

Beispiel:
python train_ml_invest.py \
  --horizon 7 \
  --train_start 2022-01-01 --train_end 2024-12-31 \
  --test_start 2025-01-01 --test_end 2025-03-31


Es trainiert ein XGBoost‑Modell direkt aus SQL, macht einen zeitbasierten Split, entfernt Depotnummer/Datum aus den Features und speichert Modell + Metriken.

So startest du’s (Beispielempfehlung)
cd C:\apano_Dienste\finance-analytics\ml

python train_ml_invest.py ^
  --horizon 7 ^
  --train_start 2022-01-01 --train_end 2024-12-31 ^
  --test_start 2025-01-01 --test_end 2025-03-31

Was es genau macht

Laden: joint CD_FA_AGG_KENNZAHLEN (per Depotnummer + Score_Datum) mit CD_FA_LABELS (per Depotnummer + stichtag) und zieht die Label‑Spalte label_invest_T{h} für den gewünschten Horizont.

Features: nimmt automatisch nur numerische Spalten, droppt Depotnummer, Score_Datum/stichtag, created_at, Label.

Ungleichgewicht: setzt scale_pos_weight = Neg/Pos.

Metriken: PR‑AUC, Precision@k & Lift@k (1/5/10%).

Artefakte:

model_invest.pkl (Modell + Featureliste + Config)

metrics.json (Kennzahlen)

importances.csv (Feature-Importance Topliste)
##########################

python train_ml_invest.py ^
  --horizon 7 ^
  --train_start 2022-01-01 --train_end 2024-12-31 ^
  --test_start 2025-01-01 --test_end 2025-03-31


"""
from __future__ import annotations
from sklearn.isotonic import IsotonicRegression
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

# ------------------------------------------------------------
# DB Loading
# ------------------------------------------------------------

def load_sql_interval(engine, agg_table: str, labels_table: str, start: str, end: str, horizon: int) -> pd.DataFrame:
    label_col = f"label_invest_T{horizon}"
    # Selektiere alle AGG-Spalten + Label, pushdown via Datum
    q = f"""
        SELECT a.*, l.{label_col}
        FROM {agg_table} a
        INNER JOIN {labels_table} l
          ON a.Depotnummer = l.Depotnummer
         AND a.Score_Datum = l.stichtag
        WHERE a.Score_Datum >= CONVERT(date, '{start}')
          AND a.Score_Datum <= CONVERT(date, '{end}')
          AND l.{label_col} IS NOT NULL
    """
    df = pd.read_sql(q, engine)
    # Normalisiere Datumsfeld auf 'stichtag'
    if 'Score_Datum' in df.columns:
        df = df.rename(columns={'Score_Datum': 'stichtag'})
    return df

# ------------------------------------------------------------
# Feature Handling
# ------------------------------------------------------------

NON_FEATURE_COLS = {"Depotnummer", "stichtag", "Score_Datum", "created_at"}

def coerce_numeric_frame(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Versucht object/bool-Spalten (außer IDs/Datum/Label) in numerisch zu konvertieren.
    - Entfernt Tausenderpunkte, ersetzt Komma->Punkt
    - to_numeric(errors='coerce')
    - bool -> uint8
    Returns: (df_converted, numeric_cols, failed_cols)
    """
    df = df.copy()
    candidates = []
    for c in df.columns:
        if c in NON_FEATURE_COLS or c == label_col:
            continue
        dt = df[c].dtype
        if np.issubdtype(dt, np.number):
            continue
        if dt == bool or str(dt) == "boolean":
            # map bool to 0/1
            df[c] = df[c].astype("uint8")
            continue
        if dt == object:
            candidates.append(c)

    failed = []
    for c in candidates:
        s = df[c].astype(str).str.strip()
        # häufige Formatierungen: 1.234,56  |  1 234,56  |  1,234.56
        s = (
            s.str.replace(r"\s", "", regex=True)      # Leerzeichen weg
             .str.replace(".", "", regex=False)       # Tausenderpunkt raus
             .str.replace(",", ".", regex=False)      # Komma -> Punkt
        )
        coerced = pd.to_numeric(s, errors="coerce")
        # Wenn >70% nicht-NaN -> übernehmen
        ok_ratio = coerced.notna().mean()
        if ok_ratio >= 0.7:
            df[c] = coerced
        else:
            failed.append(c)

    # bool-Spalten in numerisch
    for c in df.select_dtypes(include=["bool"]).columns:
        if c not in NON_FEATURE_COLS and c != label_col:
            df[c] = df[c].astype("uint8")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df, numeric_cols, failed



def select_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = NON_FEATURE_COLS.union({label_col})
    feat_cols = [c for c in num_cols if c not in drop_cols]
    return feat_cols


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    assert 0 < k <= 1
    n = max(1, int(np.ceil(len(y_score) * k)))
    idx = np.argsort(-y_score)[:n]
    return float(y_true[idx].mean())


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    base = float(y_true.mean()) or 1e-9
    return precision_at_k(y_true, y_score, k) / base

# ------------------------------------------------------------
# Train / Eval
# ------------------------------------------------------------

@dataclass
class TrainConfig:
    horizon: int
    agg_table: str = "CD_FA_AGG_KENNZAHLEN"
    labels_table: str = "CD_FA_LABELS"
    train_start: str = "2022-01-01"
    train_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-03-31"
    model_out: str = "model_invest.pkl"
    metrics_out: str = "metrics.json"
    importances_out: str = "importances.csv"
    cal_window_days: int = 90  # letzte N Tage der Trainingsperiode für Isotonic-Kalibrierung

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

def train_and_eval(cfg: TrainConfig):
    # DB Engine
    try:
        from db_connection import get_engine
    except ImportError:
        import sys, pathlib
        parent = pathlib.Path(__file__).resolve().parent.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        from db_connection import get_engine

    eng = get_engine()

    # Load
    train_df = load_sql_interval(eng, cfg.agg_table, cfg.labels_table, cfg.train_start, cfg.train_end, cfg.horizon)
    test_df = load_sql_interval(eng, cfg.agg_table, cfg.labels_table, cfg.test_start, cfg.test_end, cfg.horizon)
    
    label_col = f"label_invest_T{cfg.horizon}"



    # 1. Label-Spalte typisieren (Sicherheit!)
    train_df[label_col] = pd.to_numeric(train_df[label_col], errors="coerce")
    test_df[label_col]  = pd.to_numeric(test_df[label_col], errors="coerce")

    # 2. Label-Verteilung prüfen
    print("Label-Verteilung (Train):", train_df[label_col].value_counts(dropna=False))
    print("Label-Verteilung (Test):", test_df[label_col].value_counts(dropna=False))
    print("Label-Typ (Train):", train_df[label_col].dtype)

    # 3. NaNs entfernen
    train_df = train_df[train_df[label_col].notna()]
    test_df  = test_df[test_df[label_col].notna()]

    # 4. Sicherstellen, dass es mindestens ein positives Label gibt
    if train_df[label_col].sum() < 1:
        raise ValueError("Trainings-Label enthält keine positiven Beispiele (keine 1).")

    # Direkt danach:
    train_df = robust_feature_typing(train_df)
    test_df  = robust_feature_typing(test_df)


    # Robust: Objektspalten numerisch machen (Komma/Format fixen)
    train_df, train_numeric_cols, train_failed = coerce_numeric_frame(train_df, label_col)
    test_df,  test_numeric_cols,  test_failed  = coerce_numeric_frame(test_df,  label_col)

    if train_failed:
        print("[WARN] Nicht numerisch konvertierbare Spalten (Train) wurden ignoriert:", train_failed[:10], "..." if len(train_failed) > 10 else "")
    if test_failed:
        print("[WARN] Nicht numerisch konvertierbare Spalten (Test) wurden ignoriert:", test_failed[:10], "..." if len(test_failed) > 10 else "")

    # Kalibrierfenster = letzte N Tage der Trainingsperiode
    train_df['stichtag'] = pd.to_datetime(train_df['stichtag'])
    cal_cut = pd.to_datetime(cfg.train_end) - pd.Timedelta(days=cfg.cal_window_days)
    train_sub = train_df[train_df['stichtag'] <= cal_cut].copy()
    cal_sub   = train_df[train_df['stichtag'] >  cal_cut].copy()
    if len(cal_sub) == 0 or len(train_sub) == 0:
        # Fallback: 80/20 per Zeitschnitt
        perc80 = train_df['stichtag'].quantile(0.8)
        train_sub = train_df[train_df['stichtag'] <= perc80].copy()
        cal_sub   = train_df[train_df['stichtag'] >  perc80].copy()

    # Feature columns
    feat_cols = select_feature_columns(train_df, label_col)
    if not feat_cols:
        raise RuntimeError("Keine Feature-Spalten gefunden. Prüfe AGG-Tabelle und Datentypen.")

    print("[INFO] Anzahl Feature-Spalten:", len(feat_cols))
    print("[INFO] Beispiele Features:", feat_cols[:15])
    print("[INFO] Positivrate (Train/Test):", train_df[label_col].mean(), test_df[label_col].mean())

    X_train = train_df[feat_cols].values
    y_train = train_df[label_col].values.astype(int)
    X_test = test_df[feat_cols].values
    y_test = test_df[label_col].values.astype(int)

    # Class imbalance handling
    pos = max(1, y_train.sum())
    neg = max(1, len(y_train) - pos)
    scale_pos_weight = neg / pos

    # Class imbalance
    X_tr = train_sub[feat_cols].fillna(0.0).values
    y_tr = train_sub[label_col].values.astype(int)
    X_cal = cal_sub[feat_cols].fillna(0.0).values
    y_cal = cal_sub[label_col].values.astype(int)

    pos = max(1, y_tr.sum())
    neg = max(1, len(y_tr) - pos)
    scale_pos_weight = neg / pos

    base = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    base.fit(X_tr, y_tr)

    # Isotonic auf separatem Kalibrierfenster
    p_cal_raw = base.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip').fit(p_cal_raw, y_cal)

    # Wrapper mit predict_proba(), kompatibel zum Daily-Scorer
    model = CalibratedModel(base, iso)

    # Predictions
    # p_train = model.predict_proba(X_train)[:, 1]
    # p_test = model.predict_proba(X_test)[:, 1]

    p_train_raw = base.predict_proba(train_df[feat_cols].fillna(0.0).values)[:, 1]
    p_train = iso.transform(p_train_raw)
    X_test = test_df[feat_cols].fillna(0.0).values
    y_test = test_df[label_col].values.astype(int)
    p_test_raw = base.predict_proba(X_test)[:, 1]
    p_test = iso.transform(p_test_raw)


    # Metrics
    ap_train = average_precision_score(y_train, p_train)
    ap_test = average_precision_score(y_test, p_test)
    k_list = [0.01, 0.05, 0.10]
    metrics = {
        "horizon": cfg.horizon,
        "train_range": [cfg.train_start, cfg.train_end],
        "test_range": [cfg.test_start, cfg.test_end],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_test": float(y_test.mean()),
        "ap_train": float(ap_train),
        "ap_test": float(ap_test),
        "precision_at_k_test": {str(int(k*100))+"%": float(precision_at_k(y_test, p_test, k)) for k in k_list},
        "lift_at_k_test": {str(int(k*100))+"%": float(lift_at_k(y_test, p_test, k)) for k in k_list},
        "scale_pos_weight": float(scale_pos_weight),
        "n_features": len(feat_cols),
        "calibrated": True,
        "cal_method": "isotonic",
        "cal_window_days": cfg.cal_window_days,
    }

    # Save artifacts
    bundle = {
        "model": model,
        "features": feat_cols,
        "label_col": label_col,
        "version": f"v1.1-cal-iso{cfg.cal_window_days}d",
        "train_cfg": cfg.__dict__,
    }
    joblib.dump(bundle, cfg.model_out)

    # Importances
    try:
        importances = pd.DataFrame({
            "feature": feat_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importances.to_csv(cfg.importances_out, index=False)
    except Exception:
        importances = pd.DataFrame()

    with open(cfg.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Kurzer Report
    print("=== Training fertig ===")
    print(f"Train: {cfg.train_start}..{cfg.train_end}  |  Test: {cfg.test_start}..{cfg.test_end}")
    print(f"AP(train)={ap_train:.4f}  AP(test)={ap_test:.4f}")
    for k in k_list:
        print(f"P@{int(k*100)}%={metrics['precision_at_k_test'][str(int(k*100))+'%']:.4f}  "
              f"Lift@{int(k*100)}%={metrics['lift_at_k_test'][str(int(k*100))+'%']:.2f}")
    if not importances.empty:
        top5 = importances.head(5)
        print("Top-5 Features:")
        for r in top5.itertuples(index=False):
            print(f"  {r.feature:<30s} {r.importance:.4f}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ML model for invest readiness")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--train_start", required=True)
    p.add_argument("--train_end", required=True)
    p.add_argument("--test_start", required=True)
    p.add_argument("--test_end", required=True)
    p.add_argument("--agg_table", default="CD_FA_AGG_KENNZAHLEN")
    p.add_argument("--labels_table", default="CD_FA_LABELS")
    p.add_argument("--model_out", default="model_invest.pkl")
    p.add_argument("--metrics_out", default="metrics.json")
    p.add_argument("--importances_out", default="importances.csv")
    p.add_argument("--cal_window_days", type=int, default=90)
    return p.parse_args()

def robust_feature_typing(df, today=None):
    """
    Typisiert alle relevanten Features aus AGG für ML-Training robust als numerisch.
    Wandelt Datumsspalten in sinnvolle numerische Features (z.B. Tage seit letztem Kauf).
    """
    # Heutiges Datum für Zeitdifferenzen
    if today is None:
        today = pd.Timestamp.today()

    # Zahlenspalten explizit konvertieren
    num_cols = [
        "Orderfrequenz", "ØOrdervolumen", "Aktienquote", "Cashquote", "TechAffin",
        "Einzahlung_30d", "Inaktiv_seit_Kauf", "Orderfrequenz_30d", "Orderfrequenz_90d",
        "ØOrdervolumen_30d", "ØOrdervolumen_90d", "Einzahlung_90d", "Last_Buy_Days", "Last_Trade_Days",
        "Sparplan_Klein", "Einzahlung_30d_eff", "Auszahlung_30d", "Auszahlung_90d", "Nettoflow_30d",
        "Nettoflow_90d", "Inflow_Strength_30d", "Deposit_Volatility_6m", "Sparplan_Strength_6m",
        "Activity_Months_6m", "Last_Cashflow_Days", "Streak_Inactive_Months", "HHI_Concentration",
        "DeepLossShare_50", "WinLoss_Skew", "TechWeight", "DIY_FrequentSmall", "Cash_abs", "Bestandwert",
        "Depotwert", "Pos_Count", "ZeroPos", "Wertlos_Flag", "Cash_over_bestand", "Cashquote_true"
    ]
    for col in num_cols:
        if col in df.columns:
            # .astype(str) hilft bei Importfehlern mit Komma/Punkt
            # Dann to_numeric (coerce für Fehler)
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    # Bit-Spalten als int umwandeln (falls als bool oder object importiert)
    bit_cols = ["Sparplan_Klein", "DIY_FrequentSmall", "Wertlos_Flag"]
    for col in bit_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Datumsspalte 'Letzter_Kauf' in 'Last_Buy_Days' oder 'days_since_last_buy' umwandeln
    if "Letzter_Kauf" in df.columns:
        df["Letzter_Kauf"] = pd.to_datetime(df["Letzter_Kauf"], errors="coerce")
        df["days_since_last_buy"] = (today - df["Letzter_Kauf"]).dt.days
        df["days_since_last_buy"] = pd.to_numeric(df["days_since_last_buy"], errors="coerce")
        # Evtl. 'Last_Buy_Days' überschreiben, falls sinnvoll
        df.drop("Letzter_Kauf", axis=1, inplace=True)

    # Fallback: NaNs durch sinnvolle Defaults ersetzen (optional, für ML oft relevant)
    df.fillna(0, inplace=True)

    return df

def main():
    args = parse_args()
    cfg = TrainConfig(
        horizon=args.horizon,
        agg_table=args.agg_table,
        labels_table=args.labels_table,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        importances_out=args.importances_out,
        cal_window_days=args.cal_window_days,
    )
    train_and_eval(cfg)


if __name__ == "__main__":
    main()
