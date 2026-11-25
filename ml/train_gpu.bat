@echo off
REM ================================================================
REM  GPU training (RTX 4050). Run AFTER applying fix_gpu_tdr.bat
REM  (as admin) and restarting, so the GPU watchdog won't crash.
REM
REM  Moderate config first to confirm stability; scale up later.
REM  Just double-click, or run in CMD.
REM ================================================================
cd /d "C:\Users\KESHAV UPADHYAY\Desktop\Major project\ml"

set RICE_SOURCE_DOMAIN=paddy_doctor
set RICE_TARGET_DOMAIN=philippines
set RICE_NUM_WORKERS=0
set RICE_CACHE=1

echo ============================================================
echo  GPU TRAINING - MobileNetV3, 3 seeds, 25 epochs, batch 64
echo  RAM image cache ON (fast) + RTX 4050 + TDR fix = quick + stable.
echo ============================================================
echo.

".venv\Scripts\python.exe" prepare_data.py
".venv\Scripts\python.exe" run_seeds.py --model mobilenet_v3_small --seeds 42 1 2 --epochs 25 --batch 64

echo.
echo ============================================================
echo  DONE! Results in outputs\seed_summary_mobilenet_v3_small.json
echo  Reopen Claude and it will read the results + finish the report.
echo ============================================================
pause
