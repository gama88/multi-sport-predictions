@echo off
title Download Datasets
echo.
echo  ============================================
echo   📥 Downloading Kaggle Datasets
echo  ============================================
echo.
echo  Downloading all sports datasets...
python scripts/download_datasets.py --sport all
echo.
echo  Copying to project folder...
python scripts/copy_datasets.py
echo.
echo  ✅ Done!
pause
