@echo off
chcp 65001 >nul
echo.
echo ========================================
echo    Data Cleaning Dashboard
echo ========================================
echo.
echo Starting application...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo.
    echo Please run setup.bat first to set up the environment.
    echo.
    echo Double-click setup.bat and follow the instructions.
    echo.
    pause
    exit /b 1
)

REM Check if requirements are installed
if not exist "venv\Lib\site-packages\streamlit" (
    echo WARNING: Dependencies not fully installed.
    echo.
    echo Please run setup.bat first to install requirements.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment and run app
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting Data Cleaning Dashboard...
echo.
echo The application will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C in this window to stop the application.
echo.

timeout /t 3 /nobreak >nul

streamlit run app.py

REM Keep window open after app closes
echo.
echo Application closed.
echo.
pause