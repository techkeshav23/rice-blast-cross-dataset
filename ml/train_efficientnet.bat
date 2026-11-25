@echo off
REM ================================================================
REM  BULLETPROOF breadth: 2nd backbone (EfficientNet-B0).
REM  Trains baseline + MixStyle on BOTH pairings (3 seeds) so we can
REM  show the findings (DG fails, AdaBN helps) generalise across
REM  backbones, not just MobileNetV3.  GPU + RAM cache.  (~60-80 min)
REM ================================================================
cd /d "C:\Users\KESHAV UPADHYAY\Desktop\Major project\ml"
set RICE_NUM_WORKERS=0
set RICE_CACHE=1

echo ############ EfficientNet-B0 | Paddy Doctor -> Philippines ############
set RICE_SOURCE_DOMAIN=paddy_doctor
set RICE_TARGET_DOMAIN=philippines
".venv\Scripts\python.exe" prepare_data.py
".venv\Scripts\python.exe" run_seeds.py --model efficientnet_b0 --methods none mixstyle --seeds 42 1 2 --epochs 20 --batch 64

echo.
echo ############ EfficientNet-B0 | anshulm -> riceleafs ############
set RICE_SOURCE_DOMAIN=kaggle_rice
set RICE_TARGET_DOMAIN=riceleafs
".venv\Scripts\python.exe" prepare_data.py
".venv\Scripts\python.exe" run_seeds.py --model efficientnet_b0 --methods none mixstyle --seeds 42 1 2 --epochs 20 --batch 64

echo.
echo ============================================================
echo  DONE! EfficientNet-B0 results saved. Reopen Claude -> it
echo  AdaBN-evals EfficientNet + finalises the bulletproof study.
echo ============================================================
pause
