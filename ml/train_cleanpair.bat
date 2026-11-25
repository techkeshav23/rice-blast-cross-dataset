@echo off
REM ================================================================
REM  CLEAN single-leaf cross-dataset experiment.
REM  Source: anshulm  (kaggle_rice) - single-leaf, green background
REM  Target: riceleafs (nizorogbezuode) - single-leaf, clean background
REM  Strong lesion signal in both -> fair test for MixStyle DG.
REM
REM  MobileNetV3-Small, 5 seeds, 20 epochs, batch 64, GPU + RAM cache.
REM  Just double-click, or run in CMD. (~20-25 min)
REM ================================================================
cd /d "C:\Users\KESHAV UPADHYAY\Desktop\Major project\ml"

set RICE_SOURCE_DOMAIN=kaggle_rice
set RICE_TARGET_DOMAIN=riceleafs
set RICE_NUM_WORKERS=0
set RICE_CACHE=1

echo ============================================================
echo  CLEAN PAIRING: anshulm -> riceleafs (single-leaf, both)
echo  MobileNetV3, 5 seeds, 20 epochs, batch 64
echo ============================================================
echo.

".venv\Scripts\python.exe" prepare_data.py
".venv\Scripts\python.exe" run_seeds.py --model mobilenet_v3_small --seeds 42 1 2 3 4 --epochs 20 --batch 64

echo.
echo ============================================================
echo  DONE! Result:
echo    outputs\seed_summary_mobilenet_v3_small_kaggle_rice-riceleafs.json
echo  Reopen Claude -> it reads this + does significance analysis.
echo ============================================================
pause
