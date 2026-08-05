@echo off
REM Long-running dashboard server. Started hidden at logon.
REM
REM Unlike the collector this never exits, so the scheduled task must carry no
REM ExecutionTimeLimit ? otherwise Task Scheduler kills it after the default
REM three days and the page quietly stops responding.
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" "%~dp0dashboard.py" >> "%~dp0dashboard.log" 2>&1
