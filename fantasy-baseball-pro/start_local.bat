@echo off
setlocal

cd /d "%~dp0"

if not exist ".\venv\Scripts\python.exe" (
  echo [ERROR] venv python not found at .\venv\Scripts\python.exe
  echo Make sure your virtual environment exists.
  pause
  exit /b 1
)

echo Starting Fantasy Baseball Scout backend...
start "Fantasy Baseball API" cmd /k ".\venv\Scripts\python.exe .\main.py"

timeout /t 3 /nobreak >nul

echo Opening app in browser...
start "" ".\index.html"

echo Done. You can close this window.
endlocal
