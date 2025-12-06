@echo off
REM Quick setup script for remote system
echo Setting up SyncNet FCN environment...

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate

REM Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install other dependencies
pip install -r requirements.txt

echo.
echo ============================================
echo   Setup Complete!
echo   Now run: train_rtx_a5000.bat
echo ============================================
pause
