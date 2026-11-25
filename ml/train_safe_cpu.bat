@echo off
REM ================================================================
REM  SAFE training on CPU only (GPU disabled) -> NO driver crash.
REM  Slower than GPU but 100%% stable. Best for this laptop.
REM  Just double-click this file, or run it in CMD.
REM ================================================================
cd /d "C:\Users\KESHAV UPADHYAY\Desktop\Major project\ml"

set RICE_SOURCE_DOMAIN=paddy_doctor
set RICE_TARGET_DOMAIN=philippines
set RICE_NUM_WORKERS=0
REM Empty CUDA_VISIBLE_DEVICES hides the GPU -> torch uses CPU (no TDR crash)
set CUDA_VISIBLE_DEVICES=

echo ============================================================
echo  SAFE CPU TRAINING (no GPU) - MobileNetV3, 3 seeds, 20 epochs
echo  This will take ~30-45 min. Laptop stays cool and stable.
echo ============================================================
echo.

".venv\Scripts\python.exe" prepare_data.py
".venv\Scripts\python.exe" run_seeds.py --model mobilenet_v3_small --seeds 42 1 2 --epochs 20 --batch 64

echo.
echo ============================================================
echo  DONE! Results saved in:
echo    outputs\seed_summary_mobilenet_v3_small.json
echo    outputs\metrics_baseline_mobilenet_v3_small_s*.json
echo    outputs\metrics_mixstyle_mobilenet_v3_small_s*.json
echo  Reopen Claude later and it will read these and finish the report.
echo ============================================================
pause
