@echo off
echo ============================================
echo  CitySight — Smart City Vision Analytics
echo ============================================
echo.

REM Check for .env
if not exist .env (
    echo [!] No .env found — copying .env.example
    copy .env.example .env >nul
    echo [*] Edit .env with your settings, then re-run
    pause
    exit /b
)

REM Setup venv if needed
if not exist .venv\ (
    echo [*] Creating virtual environment...
    python -m venv .venv
    echo [*] Created .venv
)

REM Activate
call .venv\Scripts\activate.bat

REM Install backend
echo [*] Installing backend dependencies...
pip install -q -r backend\requirements.txt

REM Install frontend
echo [*] Installing frontend dependencies...
pip install -q -r frontend\requirements.txt

echo.
echo ============================================
echo  Starting CitySight...
echo ============================================
echo.
echo  Backend API:  http://localhost:8000
echo  API Docs:     http://localhost:8000/docs
echo  Frontend:     http://localhost:8501 (run separately)
echo.
echo  [1] Starting backend server...
start "CitySight Backend" .venv\Scripts\python.exe -m backend.main

timeout /t 5 /nobreak >nul

echo  [2] Starting Streamlit dashboard...
start "CitySight Dashboard" .venv\Scripts\streamlit.exe run frontend\app.py

echo.
echo ============================================
echo  Both servers started!
echo  Open http://localhost:8501 in your browser
echo ============================================
pause
