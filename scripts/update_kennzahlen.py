# -*- coding: utf-8 -*-
# taeglich_update_kennzahlen.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from datetime import datetime
from db_connection import get_engine, get_engine_source
# Pfad ggf. anpassen:
from scripts.backfill_update_kennzahlen import compute_agg, write_agg, backfill_update_kennzahlen


# Engines
engine_src = get_engine_source()
engine_dest = get_engine()

# Letztes verfügbares Datum in der Quelle holen
with engine_src.begin() as conn:
    max_datum = conn.execute(text("""
        SELECT MAX(Datum) 
        FROM Provisionen.dbo.cd_Datei 
        WHERE Pruefziffer > 0
    """)).scalar()

if max_datum is None:
    raise RuntimeError("Keine Daten in Provisionen.dbo.cd_Datei gefunden.")

print(f"! Letztes verfügbares Datum in der Quelle: {max_datum}")

# Nur diesen Tag verarbeiten
backfill_update_kennzahlen(
    engine_src=engine_src,
    engine_dest=engine_dest,
    months=0,
    start_date=max_datum,
    end_date=max_datum,
    dry_run=False,
    verbose=True
)

