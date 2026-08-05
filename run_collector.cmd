@echo off
REM Daily data collection: Python strategy first, then TradingView sync.
REM
REM tv_sync runs SECOND and its failure is non-fatal. TradingView needs the
REM desktop app running with CDP, which cannot be guaranteed at 18:00 ? but an
REM absent TradingView must never block the Python collection, and it is
REM recorded as an explicit TV_UNAVAILABLE row rather than silently skipped.
cd /d "%~dp0"

REM Rotate the log before appending. Two scheduled runs a day append forever;
REM without this the file grows unbounded and nothing ever truncates it.
if exist "%~dp0collector.log" for %%A in ("%~dp0collector.log") do (
  if %%~zA GTR 2097152 (
    if exist "%~dp0collector.log.1" del "%~dp0collector.log.1"
    move /y "%~dp0collector.log" "%~dp0collector.log.1" >nul
  )
)

"%~dp0venv\Scripts\python.exe" "%~dp0collector.py" --once-per-day >> "%~dp0collector.log" 2>&1
set COLLECTOR_RC=%ERRORLEVEL%

"%~dp0venv\Scripts\python.exe" "%~dp0tv_sync.py" >> "%~dp0collector.log" 2>&1
set TVSYNC_RC=%ERRORLEVEL%

"%~dp0venv\Scripts\python.exe" "%~dp0status.py" >> "%~dp0collector.log" 2>&1
set GATE_RC=%ERRORLEVEL%

echo [%DATE% %TIME%] collector=%COLLECTOR_RC% tv_sync=%TVSYNC_RC% gate=%GATE_RC% >> "%~dp0collector.log"

REM Only the collector's failure is a real failure.
exit /b %COLLECTOR_RC%
