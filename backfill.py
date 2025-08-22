# backfill.py
from datetime import date
from scripts.backfill_update_kennzahlen import backfill_update_kennzahlen
# Optional: eigene Engines übergeben (sonst werden get_engine_prov/fa aus db_connection genutzt)
from db_connection import get_engine, get_engine_source

def main():
    # Beispiele:
    # 1) Letzte 24 Monate bis max(Datum) in der Quelle:
    backfill_update_kennzahlen(
        engine_src=get_engine_source(),  # Quelle (Provisionen)
        engine_dest=get_engine(),        # Ziel (FA)
        #months=0,
        months=24, # 24
        #start_date='2025-08-07', #None,
        start_date = None,
        #end_date=date.today().isoformat(),     # bis heute
        end_date = None,
        dry_run=False,
        verbose=True
    )

    # 2) Fixer Zeitraum:
    # backfill_update_kennzahlen(
    #     months=0,
    #     start_date="2023-08-01",
    #     end_date="2025-08-10",
    #     dry_run=False,
    #     verbose=True
    # )

if __name__ == "__main__":
    main()
