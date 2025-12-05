@echo off
REM Training script for RTX A5000 (24GB VRAM)
REM Run this after running setup_remote.bat

echo ============================================
echo   SyncNet FCN Training - RTX A5000
echo ============================================

REM Activate virtual environment
call venv\Scripts\activate

REM Set CUDA device
set CUDA_VISIBLE_DEVICES=0

REM Training with optimized defaults (already set in script)
REM Dataset path: E:\newtestdataset (contains s2, s3, s5... folders)
python train_syncnet_fcn_classification.py --data_dir "E:\newtestdataset"

echo ============================================
echo   Training Complete!
echo   Best model saved to: checkpoints_classification\best.pth
echo ============================================
pause
