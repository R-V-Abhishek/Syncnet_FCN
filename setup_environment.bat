@echo off
REM Setup script for SyncNet FCN project

echo ================================================================================
echo SyncNet FCN - Environment Setup
echo ================================================================================

echo.
echo Creating virtual environment...
python -m venv venv

echo.
echo Activating virtual environment...
call .\venv\Scripts\activate.bat

echo.
echo Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing other dependencies...
pip install opencv-python scipy python_speech_features numpy

echo.
echo ================================================================================
echo Setup complete!
echo ================================================================================
echo.
echo To activate the environment in the future, run:
echo   .\venv\Scripts\activate.bat
echo.
echo Then you can run your scripts:
echo   python test_sync_detection.py --video your_video.mp4
echo.
pause
