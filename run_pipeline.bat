@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "BASE=C:\apano_Dienste\finance-analytics"
set "VENV=%BASE%\venv"
set "LOGDIR=%BASE%\logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

set "LOGFILE=%LOGDIR%\task_debug.log"
(
  echo [!date! !time!] Task gestartet...
  whoami
  echo Aktuelles Verzeichnis (vor pushd) 
) >> "%LOGFILE%" 2>&1

pushd "%BASE%" >> "%LOGFILE%" 2>&1

REM Zeitstempel YYYYMMDD_HHMM
for /f "tokens=1-3 delims=." %%a in ("%date%") do set _D=%%c%%b%%a
for /f "tokens=1-2 delims=: " %%h in ("%time%") do set _T=%%h%%i
set "TS=%_D%_%_T%"
set "RUNLOG=%LOGDIR%\run_%TS%.log"

set "PYEXE=%VENV%\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

(
  echo [!date! !time!] BASE=%BASE%
  echo LOGDIR=%LOGDIR%
  echo Verwende Python "!PYEXE!"
) >> "%LOGFILE%" 2>&1

if exist "%VENV%\Scripts\activate.bat" call "%VENV%\Scripts\activate.bat" >> "%LOGFILE%" 2>&1
"%PYEXE%" -V >> "%LOGFILE%" 2>&1

echo [!date! !time!] Starte Pipeline... >> "%LOGFILE%" 2>&1
"%PYEXE%" scripts\run_pipeline.py >> "%RUNLOG%" 2>&1
set "RC=%ERRORLEVEL%"

echo [!date! !time!] Pipeline ExitCode=!RC! >> "%LOGFILE%" 2>&1

if exist "%LOGDIR%\*.log" forfiles /p "%LOGDIR%" /m *.log /d -14 /c "cmd /c del @path" >nul 2>&1

popd
endlocal & exit /b %RC%
