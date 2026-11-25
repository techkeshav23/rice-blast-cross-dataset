@echo off
REM ================================================================
REM  FULL DG COMPARISON (for 100%% complete research):
REM  baseline vs MixStyle vs RandConv, on BOTH real pairings.
REM  Resumes: baseline+MixStyle already done -> only RandConv trains.
REM  MobileNetV3-Small, GPU + RAM cache.  (~35-40 min)
REM ================================================================
cd /d "C:\Users\KESHAV UPADHYAY\Desktop\Major project\ml"
set RICE_NUM_WORKERS=0
set RICE_CACHE=1

echo ############ PAIRING 1: Paddy Doctor -> Philippines (canopy) ############
set RICE_SOURCE_DOMAIN=paddy_doctor
set RICE_TARGET_DOMAIN=philippines
".venv\Scripts\python.exe" prepare_data.py
".venv\Scripts\python.exe" run_seeds.py --model mobilenet_v3_small --methods none mixstyle randconv --seeds 42 1 2 3 4 5 6 7 --epochs 25 --batch 64

echo.
echo ############ PAIRING 2: anshulm -> riceleafs (single-leaf) ############
set RICE_SOURCE_DOMAIN=kaggle_rice
set RICE_TARGET_DOMAIN=riceleafs
".venv\Scripts\python.exe" prepare_data.py
".venv\Scripts\python.exe" run_seeds.py --model mobilenet_v3_small --methods none mixstyle randconv --seeds 42 1 2 3 4 --epochs 20 --batch 64

echo.
echo ============================================================
echo  DONE! Results:
echo    outputs\seed_summary_mobilenet_v3_small_paddy_doctor-philippines.json
echo    outputs\seed_summary_mobilenet_v3_small_kaggle_rice-riceleafs.json
echo  Reopen Claude -> full multi-method significance analysis + paper.
echo ============================================================
pause
