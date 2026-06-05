# Local TTS Studio - one-command installer for Windows (PowerShell)
# Usage:  ./install.ps1
# Creates a local virtual environment, installs dependencies, checks ffmpeg, and launches the app.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  Local TTS Studio - setup" -ForegroundColor Green
Write-Host "  =========================" -ForegroundColor Green
Write-Host ""

# --- 1. Find a suitable Python (3.10+) -------------------------------------
function Get-Python {
    $candidates = @(
        @{ Exe = "py";      PyArgs = @("-3") },
        @{ Exe = "python";  PyArgs = @() },
        @{ Exe = "python3"; PyArgs = @() }
    )
    foreach ($c in $candidates) {
        if (Get-Command $c.Exe -ErrorAction SilentlyContinue) {
            try {
                $ver = & $c.Exe @($c.PyArgs) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
                if ($LASTEXITCODE -eq 0 -and $ver) {
                    $maj, $min = $ver.Trim().Split(".")
                    if ([int]$maj -eq 3 -and [int]$min -ge 10) {
                        return $c
                    }
                }
            } catch { }
        }
    }
    return $null
}

$py = Get-Python
if ($null -eq $py) {
    Write-Host "  ERROR: Python 3.10+ was not found." -ForegroundColor Red
    Write-Host "  Install it from https://www.python.org/downloads/ and re-run this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "  Using Python: $($py.Exe) $($py.PyArgs -join ' ')" -ForegroundColor Cyan

# --- 2. Create / reuse virtual environment ---------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "  Creating virtual environment (.venv)..." -ForegroundColor Cyan
    & $py.Exe @($py.PyArgs) -m venv .venv
}
$venvPy = Join-Path (Resolve-Path ".venv") "Scripts\python.exe"

# --- 3. Install dependencies -----------------------------------------------
Write-Host "  Upgrading pip..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip --quiet

Write-Host "  Installing dependencies (this can take a few minutes)..." -ForegroundColor Cyan
& $venvPy -m pip install -r requirements.txt

# --- 4. Check ffmpeg (needed for MP3/M4A) ----------------------------------
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($null -eq $ffmpeg) {
    Write-Host ""
    Write-Host "  ffmpeg was not found (needed for MP3/M4A export)." -ForegroundColor Yellow
    Write-Host "  Attempting to install it with winget..." -ForegroundColor Yellow
    try {
        winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    } catch {
        Write-Host "  Could not auto-install ffmpeg. Install it manually:" -ForegroundColor Yellow
        Write-Host "    winget install Gyan.FFmpeg" -ForegroundColor Yellow
        Write-Host "  WAV output works without ffmpeg." -ForegroundColor Yellow
    }
}

# --- 5. Launch --------------------------------------------------------------
Write-Host ""
Write-Host "  Setup complete. Starting Local TTS Studio..." -ForegroundColor Green
Write-Host "  The first run downloads model weights (~10 GB) - please be patient." -ForegroundColor Cyan
Write-Host "  Open http://localhost:8000 in your browser." -ForegroundColor Green
Write-Host ""
& $venvPy run_local.py
