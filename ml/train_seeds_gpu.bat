@echo off
REM ================================================================
REM  FINAL multi-seed run (8 seeds) for statistical significance.
REM  MobileNetV3-Small, Paddy Doctor -> Philippines, GPU + RAM cache.
REM
REM  RESUME: seeds already trained (42,1,2) are skipped automatically,
REM  so only the 5 new seeds actually train (~35 min).
REM
REM  BEFORE running (recommended for stability):
REM    1) Right-click fix_gpu_tdr.bat -> Run as administrator
REM    2) Restart the laptop
REM  (MobileNetV3 @ batch 64 already ran stable for 3 seeds, so this
REM   config is proven; the TDR fix is extra safety.)
REM
REM  Just double-click this file, or run it in CMD.
REM ================================================================
cd /d "C:\Users\KESHAV UPADHYAY\Desktop\Major project\ml"

set RICE_SOURCE_DOMAIN=paddy_doctor
set RICE_TARGET_DOMAIN=philippines
set RICE_NUM_WORKERS=0
set RICE_CACHE=1

echo ============================================================
echo  8-SEED RUN (resumes; ~35 min for the 5 new seeds)
echo  MobileNetV3-Small, 25 epochs, batch 64, GPU + cache
echo ============================================================
echo.

".venv\Scripts\python.exe" run_seeds.py --model mobilenet_v3_small --seeds 42 1 2 3 4 5 6 7 --epochs 25 --batch 64

echo.
echo ============================================================
echo  DONE! Final 8-seed result:
echo    outputs\seed_summary_mobilenet_v3_small.json
echo  Reopen Claude -> it reads this and does the significance
echo  analysis + writes the paper/report.
echo ============================================================
pause
