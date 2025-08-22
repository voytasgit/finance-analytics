@echo off
REM =====================================================
REM Label-Builder Launcher (Windows)
REM Pfad: C:\apano_Dienste\finance-analytics\ml\run_labels.bat
REM Erwartung: db_connection.py liegt eine Ebene höher (..\finance-analytics)
REM Optional: .env liegt ebenfalls eine Ebene höher
REM Aufruf: run_labels.bat [YYYY-MM-DD] [YYYY-MM-DD] [HORIZON]
REM Beispiel: run_labels.bat 2025-01-01 2025-01-31 7
REM =====================================================

REM 1) In den ML-Ordner wechseln
cd /d "C:\apano_Dienste\finance-analytics\ml"

REM 2) Parameter einlesen + Defaults setzen
set "START_DATE=%~1"
if "%START_DATE%"=="" set "START_DATE=2025-01-01"

set "END_DATE=%~2"
if "%END_DATE%"=="" set "END_DATE=2025-01-31"

set "HORIZON=%~3"
if "%HORIZON%"=="" set "HORIZON=7"

REM 3) (Optional) Virtuelle Umgebung aktivieren, wenn vorhanden
if exist "..\venv\Scripts\activate.bat" (
  call "..\venv\Scripts\activate.bat"
)

REM 4) Sicherstellen, dass Python den Parent-Ordner kennt (für db_connection.py)
set "PYTHONPATH=%PYTHONPATH%;.."

echo Starte Label-Build fuer %START_DATE% bis %END_DATE% (H=%HORIZON%) ...
python "label_builder.py" ^
  --source mssql ^
  --agg_table CD_FA_AGG_KENNZAHLEN ^
  --orders_table cd_Order ^
  --buch_table cd_Buchung ^
  --start_date %START_DATE% ^
  --end_date %END_DATE% ^
  --horizon %HORIZON% ^
  --out_table CD_FA_LABELS

echo.
echo Fertig. Zum Schliessen Taste druecken...
pause
