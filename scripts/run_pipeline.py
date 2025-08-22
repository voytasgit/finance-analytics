import subprocess, sys, traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"

# Timeout pro Schritt (Sekunden) – anpassen falls nötig
STEP_TIMEOUT = 1800  # 30 Minuten

STEPS = [
    SCRIPTS / "update_kennzahlen.py",
    SCRIPTS / "scoring.py",
    SCRIPTS / "invest_bereitschaft.py",
    SCRIPTS / "alerts.py",
    SCRIPTS / "segmentierung.py",
    [SCRIPTS / "fill_leads.py", "--date", "today"],
    SCRIPTS / "qc_monitor.py",
    SCRIPTS / "agg_validation_starter.py",
    # NEU: Score-ML aus Unterverzeichnis "ml"
    [BASE / "ml" / "score_daily_ml.py", "--model_path", "model_invest.pkl"],
]

def run(step):
    if isinstance(step, (list, tuple)):
        parts = [str(p) for p in step]
    else:
        parts = [str(step)]
    cmd = [sys.executable] + parts
    pretty = " ".join(parts)
    print(f"[RUN] {pretty}", flush=True)
    try:
        # Setze das Arbeitsverzeichnis für ML-Skripte korrekt!
        cwd = str(BASE)
        # Falls Skript im "ml" liegt, setze das CWD auf "ml"
        if parts[0].endswith("score_daily_ml.py"):
            cwd = str(BASE / "ml")
        r = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            timeout=STEP_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {pretty} nach {STEP_TIMEOUT}s", flush=True)
        sys.exit(124)
    except Exception:
        print(f"[ERROR] Unerwarteter Fehler in {pretty}:\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
    if r.returncode != 0:
        print(f"[FAIL] ExitCode {r.returncode} in {pretty}", flush=True)
        sys.exit(r.returncode)

if __name__ == "__main__":
    for s in STEPS:
        run(s)
    print("[OK] Pipeline erfolgreich.", flush=True)