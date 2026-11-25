@echo off
REM ================================================================
REM  GPU STABILITY FIX for ML/compute on Windows.
REM  Increases the GPU "TDR" watchdog timeout from 2s -> 20s so the
REM  display driver is NOT reset (black-screen crash) during heavy
REM  CUDA training. This is the standard fix for GPU compute crashes.
REM
REM  HOW TO RUN:  Right-click this file -> "Run as administrator"
REM  Then RESTART the laptop for it to take effect.
REM ================================================================

net session >nul 2>&1
if %errorLevel% neq 0 (
  echo.
  echo  [!] Not running as admin.
  echo  [!] RIGHT-CLICK this file and choose "Run as administrator".
  echo.
  pause
  exit /b 1
)

echo Setting GPU TDR timeout to 20 seconds...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay    /t REG_DWORD /d 20 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDdiDelay /t REG_DWORD /d 20 /f
echo.
echo  DONE. Now RESTART the laptop.
echo  After restart, GPU training will be stable (no driver-watchdog crash).
echo.
pause
