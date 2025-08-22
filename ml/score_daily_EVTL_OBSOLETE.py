
# score_daily.py
import pandas as pd, joblib
from datetime import date
from sqlalchemy import create_engine

bundle = joblib.load("model_invest.pkl")
model, feature_cols, version = bundle["model"], bundle["features"], bundle["version"]

# 1) Heutige AGG-Kennzahlen laden
agg = pd.read_parquet("agg_today.parquet")  # oder SELECT ... FROM CD_FA_AGG_KENNZAHLEN WHERE asof_date = today

X = agg[feature_cols].fillna(0.0).values
p = model.predict_proba(X)[:,1]

out = agg[["Depotnummer"]].copy()
out["run_date"] = date.today()
out["p_invest"] = p
out["score"] = (p*100).round().astype(int)
out["rank_in_run"] = out["p_invest"].rank(method="first", ascending=False).astype(int)
out["model_version"] = version
out["featureset_version"] = "fs1.0"

# 2) In SQL schreiben
# engine = create_engine("mssql+pyodbc://...")
# out.to_sql("CD_FA_INVEST_BEREITSCHAFT_ML", engine, if_exists="append", index=False)
