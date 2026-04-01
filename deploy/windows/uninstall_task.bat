@echo off
setlocal

set "TASK_NAME=ENSIA_IMPACT_Stack_OnBoot"

schtasks /Query /TN "%TASK_NAME%" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Task not found: %TASK_NAME%
  exit /b 0
)

schtasks /Delete /TN "%TASK_NAME%" /F
if errorlevel 1 (
  echo [ERROR] Failed to delete task: %TASK_NAME%
  echo [HINT] Run this script from an Administrator CMD.
  exit /b 1
)

echo [OK] Task deleted: %TASK_NAME%
exit /b 0

