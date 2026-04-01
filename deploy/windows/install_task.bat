@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "TASK_NAME=ENSIA_IMPACT_Stack_OnBoot"
set "XML_TEMPLATE=%SCRIPT_DIR%ensia_stack_on_boot.xml"
set "XML_RESOLVED=%TEMP%\ensia_stack_on_boot_resolved.xml"
set "START_BAT=%SCRIPT_DIR%start_ensia_stack.bat"

if not exist "%XML_TEMPLATE%" (
  echo [ERROR] Missing XML template: %XML_TEMPLATE%
  exit /b 1
)
if not exist "%START_BAT%" (
  echo [ERROR] Missing start script: %START_BAT%
  exit /b 1
)

powershell -NoProfile -Command "$p=(Get-Content -Raw '%XML_TEMPLATE%'); $p=$p -replace '__START_BAT_PATH__','%START_BAT:\=\\%'; Set-Content -Path '%XML_RESOLVED%' -Value $p -Encoding Unicode"

schtasks /Create /TN "%TASK_NAME%" /XML "%XML_RESOLVED%" /F
if errorlevel 1 (
  echo [ERROR] Failed to create scheduled task. Run this script from an Administrator CMD.
  exit /b 1
)

echo [OK] Task created: %TASK_NAME%
echo To test now: schtasks /Run /TN "%TASK_NAME%"
exit /b 0


