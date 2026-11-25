# Raise the 4 NEW-pairing cells from 4 -> 8 seeds (baseline + MixStyle), epochs 20,
# batch 64 (matches the existing new-cell recipe). run_seeds.py resumes (skips seeds
# already trained). Guards the app/backend metrics.json from run_seeds' push side-effect.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$env:RICE_NUM_WORKERS = "2"
$env:RICE_CACHE = "1"
$env:PYTHONUNBUFFERED = "1"

# --- backup the metrics that run_seeds.py may clobber (MNet + paddy_doctor source) ---
$bkOut = "outputs\metrics.json.bak_8seed"
$bkBe  = "..\backend\models\metrics.json.bak_8seed"
if (Test-Path "outputs\metrics.json") { Copy-Item "outputs\metrics.json" $bkOut -Force }
if (Test-Path "..\backend\models\metrics.json") { Copy-Item "..\backend\models\metrics.json" $bkBe -Force }

$cells = @(
  @{src="paddy_doctor"; tgt="riceleafs";   model="mobilenet_v3_small"},
  @{src="paddy_doctor"; tgt="riceleafs";   model="efficientnet_b0"},
  @{src="kaggle_rice";  tgt="philippines"; model="mobilenet_v3_small"},
  @{src="kaggle_rice";  tgt="philippines"; model="efficientnet_b0"}
)

foreach ($c in $cells) {
  $env:RICE_SOURCE_DOMAIN = $c.src
  $env:RICE_TARGET_DOMAIN = $c.tgt
  Write-Host "`n########## $($c.model) | $($c.src)->$($c.tgt) | seeds 42 1 2 3 4 5 6 7 ##########"
  & $py run_seeds.py --methods none mixstyle --seeds 42 1 2 3 4 5 6 7 `
        --epochs 20 --batch 64 --model $c.model
}

# --- restore the app/backend metrics (keep the official canopy MNet numbers) ---
if (Test-Path $bkOut) { Copy-Item $bkOut "outputs\metrics.json" -Force }
if (Test-Path $bkBe)  { Copy-Item $bkBe  "..\backend\models\metrics.json" -Force }
Write-Host "`nAll 4 new cells at 8 seeds. Restored app/backend metrics.json."
