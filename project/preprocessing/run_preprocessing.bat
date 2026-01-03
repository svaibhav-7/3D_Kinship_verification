@echo off
REM Script to run KinFace-II preprocessing

echo ========================================
echo KinFace-II Preprocessing for EG3D
echo ========================================

REM Activate virtual environment
call %~dp0..\env\Scripts\activate.bat

REM Run preprocessing script
echo.
echo Starting preprocessing...
echo This may take several minutes...
echo.
cd %~dp0preprocessing
python preprocess_kinface.py

echo.
echo ========================================
echo Preprocessing Complete!
echo ========================================
echo.
echo Processed images saved to: KinFaceW-II-Processed/
echo.
pause
