First-Buy ML Pipeline

Goal: End-to-end pipeline to predict the probability of a first investment for a fund (T = 7/14/30 days) before the first relevant purchase (“Pre-First-Target”).
Daily batch: Aggregation → Labels → Training (ad hoc) → Scoring → Monitoring.

Technology / Prerequisites

Python 3.11, pandas, numpy, scikit-learn, xgboost, sqlalchemy, pyodbc

MS SQL Server, ODBC Driver 18

Database access via db_connection.get_engine() (target/FA) and get_engine_source() (source/commissions)

Artifact path (models):

ARTIFACT_DIR = os.environ.get("FA_ART_DIR", r"C:\_Dienste\ML")


(The directory is created automatically.)

Core Tables (SQL, already existing)

Provisionen.dbo.cd_Order / cd_Saldo / cd_Bestand / cd_Buchung / cd_Datei

dbo.CD_FA_AGG_BUY_ – Daily aggregates (PK: Score_Date, Depot_Number)

dbo.CD_FA_LABELS_ – Labels (first_buy_date, label_T7/T14/T30)

dbo.CD_FA_BUY__ML – Daily scores (p_buy_7d/14d/30d, version)

dbo.FA_MODEL_REGISTRY – Model registry (version, paths, metrics, active flag)

dbo.FA_MODEL_EVAL – Historical offline evaluation (AP, P@k, lift)

Data Logic (key rules)

Cancellations: Any transaction whose transaction number has a remark LIKE '%storn%' is completely discarded.

Effective Buys: OrderType='Purchase', not canceled, ticket ≥ 1,000 EUR.

Pre-First-Target: No non-canceled target purchase (ISIN= , ticket ≥ 1,000) until cutoff date.

Features (excerpt):

Momentum (Buys_7d/30d, Last_Buy_Days)

Tickets (Avg_Ticket_90d)

Liquidity (Cash, Cash_Ratio, Deposit_30d_eff)

Turnover

Diversity (Pos_Count, NewRefs_90d)

Cancellations (CancelRate_30d)

Scripts & CLI
1) Aggregation (daily / backfill)

File: agg_buy_.py (or equivalent module code)

Logic: Select U/B file per day, load inputs, compute compute_agg(), deterministically DELETE + INSERT.

Examples:

python agg_buy_.py --mode daily
python agg_buy_.py --mode backfill --months 24
python agg_buy_.py --mode backfill --start_date 2024-01-01 --end_date 2025-08-20

2) Label Builder (First-Buy, windows T=7/14/30)

File: build_labels_buy_.py

SQL pushdown: Orders filtered by date + cancellation CTE (performant).

Idempotent: DELETE + bulk insert via temp staging.

Example:

python build_labels_buy_.py --start_date 2023-09-01 --end_date 2025-08-20

3) Training & Registry (XGBoost + Isotonic)

File: train_buy_.py

Splits: Train / Calib (last 90 days of train) / Test (out-of-time)

Metrics: PR-AUC, Precision@{1%,3%,5%}, Lift@{1%,3%,5%}

Persistence: Artifacts (model.pkl, calibrator.pkl, features.json, metrics.json) in ARTIFACT_DIR; entry in FA_MODEL_REGISTRY (no drop)

Example:

python train_buy_.py --horizons 7,14,30 \
  --train_start 2024-01-01 --train_end 2025-06-30 \
  --test_start  2025-07-01 --test_end  2025-08-20 \
  --max_depth 6 --learning_rate 0.05 --n_estimators 1200

4) Scoring (daily)

File: score_buy_.py

Loads latest registry entries per horizon, retrieves artifacts, scores daily aggregates (only Pre_First_Target=1), writes to CD_FA_BUY__ML.

Monotonicity clipping: p14 = max(p14, p7), p30 = max(p30, p14).

Example:

python score_buy_.py

5) Evaluation (offline, optional – for governance & regression tests)

File: eval_buy_.py

Evaluates registered models on historical intervals, writes to FA_MODEL_EVAL.

Example:

python eval_buy_.py --horizons 7,14,30 --start 2025-07-01 --end 2025-08-20

6) Daily QC (automatable)

File: qc_buy_daily_.py

Checks: number of AGG/Labels/Scores per date, positive rates T7/T14/T30, share of NULLs, simple alert thresholds (e.g., pos_30d deviation >50% vs 14-day median; NULL share >98%).

Example:

python qc_buy_daily_.py --window_days 30

7) Pipeline Runner (daily)

File: run_buy_pipeline_.py

Recommended order:

Aggregation (daily)

Labels (last day only)

Scoring (daily)

QC (daily)

Exit codes & time limits per step; suitable for Task Scheduler / CRON.

KPIs & Interpretation

AP (PR-AUC): Area under the Precision-Recall curve; robust to strong class imbalance.

P@1/3/5 %: Precision in top k% by score (directly useful for campaigns).

Lift@1/3/5 %: Ratio of P@k to baseline positive rate (e.g., Lift@1% =16 → 16× better than random).

Drift signals (tuning):

Declining AP(test) over days/weeks

Decreasing P@1% / Lift@1%

Increasing NULL share in labels/scores

Strong fluctuations in daily positive rate vs 14-day median

Action: Update AGG/Labels, retrain model (same pipeline), register new version.

Performance Notes

SQL pushdown (date + cancellation CTE) reduces IO massively.

Bulk insert: temp table + fast_executemany=True, chunksize=5–10k, INSERT ... WITH (TABLOCK).

Avoid method="multi" in to_sql (slow with pyodbc).

Keep indexes and PKs on target table (clustered PK on (Score_Date, Depot_Number)).

Operation / Recovery

Daily run recommended. If a day fails: backfill (aggregation/labels) and score for last days – pipeline is deterministic (DELETE + INSERT per day).

Monotonicity clipping ensures interpretable p(7/14/30).

Compliance / Data Hygiene

No sensitive features (gender, profession, exact address).

Logging of feature importances, metrics, registry versions.

No performance guarantees for campaigns; scores = probabilities, not advice.

Quickstart (typical first run)
# 1) Backfill 24M aggregation
python agg_buy_.py --mode backfill --months 24

# 2) Labels for the same period
python build_labels_buy_.py --start_date 2023-09-01 --end_date 2025-08-20

# 3) Training & registry
python train_buy_.py --horizons 7,14,30 \
  --train_start 2024-01-01 --train_end 2025-06-30 \
  --test_start  2025-07-01 --test_end  2025-08-20

# 4) Daily scoring
python score_buy_.py

# 5) QC
python qc_buy_daily_.py --window_days 30

Troubleshooting

Slow inserts: fast_executemany=True, temp staging, chunksize=10000, WITH (TABLOCK).

“Syntax near '.'” in DDL in Python: run DDL once manually; otherwise, run DDL separately, not in a batch with GO.

Pandas FutureWarning (fillna downcasting): instead of .fillna(...), use infer_objects(copy=False) or explicitly type columns (already implemented in scripts).
