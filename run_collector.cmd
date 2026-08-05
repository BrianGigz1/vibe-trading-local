@echo off
REM Daily data collection: Python strategy first, then TradingView sync.
REM
REM tv_sync runs SECOND and its failure is non-fatal. TradingView needs the
REM desktop app running with CDP, which cannot be guaranteed at 18:00 ? but an
REM absent TradingView must never block the Python collection, and it is
REM recorded as an explicit TV_UNAVAILABLE row rather than silently skipped.
cd /d "%~dp0"

"%~dp0venv\Scripts\python.exe" "%~dp0collector.py" --once-per-day >> "%~dp0collector.log" 2>&1
set COLLECTOR_RC=%ERRORLEVEL%

"%~dp0venv\Scripts\python.exe" "%~dp0tv_sync.py" >> "%~dp0collector.log" 2>&1
set TVSYNC_RC=%ERRORLEVEL%

echo [%DATE% %TIME%] collector=%COLLECTOR_RC% tv_sync=%TVSYNC_RC% >> "%~dp0collector.log"

REM Only the collector's failure is a real failure.
exit /b %COLLECTOR_RC%
