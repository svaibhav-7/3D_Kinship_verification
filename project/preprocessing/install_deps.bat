@echo off
REM Installation script for KinFace-II preprocessing dependencies

echo ========================================
echo Installing Preprocessing Dependencies
echo ========================================

REM Activate virtual environment
call %~dp0..\env\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing required packages...
cd %~dp0
python -m pip install -r requirements.txt

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run: run_preprocessing.bat
echo    This will process all KinFace-II images to 256x256
echo.
pause
