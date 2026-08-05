@echo off
REM Daily paper-trading data collection.
REM Wrapper exists because schtasks /TR mangles command lines containing
REM spaces; pointing the task at a single .cmd avoids the quoting problem.
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" "%~dp0collector.py" >> "%~dp0collector.log" 2>&1
exit /b %ERRORLEVEL%
