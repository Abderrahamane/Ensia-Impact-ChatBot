@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"

set "PY_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%PY_EXE%" (
  echo [ERROR] Python executable not found: %PY_EXE%
  exit /b 1
)

set "LOG_DIR=%REPO_ROOT%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

start "ENSIA_WEB" /min cmd /c "cd /d "%REPO_ROOT%" && "%PY_EXE%" web\app.py >> "%LOG_DIR%\web_stdout.log" 2>> "%LOG_DIR%\web_stderr.log""
start "ENSIA_BOT" /min cmd /c "cd /d "%REPO_ROOT%" && "%PY_EXE%" bot\telegram_bot.py >> "%LOG_DIR%\bot_stdout.log" 2>> "%LOG_DIR%\bot_stderr.log""

exit /b 0

