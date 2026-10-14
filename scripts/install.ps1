# Lewdex installer (run through Install.cmd). Installs pixi if needed, the Python environment
# (~5 GB, PyTorch with CUDA), the models (~4 GB unless they came with the archive), builds
# Lewdex.exe and creates Desktop / Start menu shortcuts.
# Safe to run again: it only does what is still missing, so it also repairs and updates.
param([switch]$NoShortcuts)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Magenta }
function Warn($text) { Write-Host "    ! $text" -ForegroundColor Yellow }

Write-Host "Lewdex - auto-tagger for Eagle" -ForegroundColor Magenta
Write-Host "Folder: $root"

if ($root -match '[^\x20-\x7E]' -or $root -match ' ') {
    Warn "The folder path has spaces or non-Latin letters. Some GPU libraries break on such"
    Warn "paths; if anything fails, move the folder to e.g. C:\Lewdex and run Install.cmd again."
}

Step "Checking the GPU"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) | Select-Object -First 1
    Write-Host "    $gpu"
    $driver = [double](($gpu -split ',')[1].Trim() -replace '^(\d+\.\d+).*', '$1')
    if ($driver -lt 580) {
        Warn "NVIDIA driver $driver is too old for CUDA 13; update it (R580 or newer) from nvidia.com."
    }
} else {
    Warn "No NVIDIA GPU / driver found. Lewdex will run on the CPU, which is VERY slow."
}

Step "pixi (environment manager)"
$pixi = (Get-Command pixi -ErrorAction SilentlyContinue).Source
foreach ($c in @("$env:LOCALAPPDATA\pixi\bin\pixi.exe", "$env:USERPROFILE\.pixi\bin\pixi.exe")) {
    if (-not $pixi -and (Test-Path $c)) { $pixi = $c }
}
if (-not $pixi) {
    Write-Host "    installing pixi from https://pixi.sh ..."
    Invoke-RestMethod -UseBasicParsing https://pixi.sh/install.ps1 | Invoke-Expression
    foreach ($c in @("$env:LOCALAPPDATA\pixi\bin\pixi.exe", "$env:USERPROFILE\.pixi\bin\pixi.exe")) {
        if (-not $pixi -and (Test-Path $c)) { $pixi = $c }
    }
    if (-not $pixi) { throw "pixi installation failed" }
}
Write-Host "    $pixi"

Step "Python environment (PyTorch + CUDA, ~5 GB download the first time)"
& $pixi install
if ($LASTEXITCODE) { throw "pixi install failed ($LASTEXITCODE)" }

Step "Models and self-check (downloads ~4 GB of models the first time)"
& $pixi run lewdex doctor
if ($LASTEXITCODE) { Warn "Some checks failed (see above). If it is only 'Eagle', start Eagle later." }

Step "Lewdex.exe"
$exe = Join-Path $root 'Lewdex.exe'
$ico = Join-Path $root 'src\lewdex\data\lewdex.ico'
$csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path $csc)) { $csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe' }
& $csc /nologo /target:winexe /optimize "/out:$exe" "/win32icon:$ico" `
    /reference:System.Windows.Forms.dll (Join-Path $root 'scripts\Lewdex.cs')
if ($LASTEXITCODE) { throw "building Lewdex.exe failed ($LASTEXITCODE)" }
Write-Host "    $exe"

if ($NoShortcuts) { Step "Done (shortcuts skipped)"; exit 0 }
Step "Shortcuts"
$shell = New-Object -ComObject WScript.Shell
$places = @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))
foreach ($dir in $places) {
    $lnk = $shell.CreateShortcut((Join-Path $dir 'Lewdex.lnk'))
    $lnk.TargetPath = $exe
    $lnk.WorkingDirectory = $root
    $lnk.IconLocation = "$ico,0"
    $lnk.Description = 'Lewdex - auto-tagger for Eagle'
    $lnk.Save()
    Write-Host "    $dir\Lewdex.lnk"
}

Step "Done"
Write-Host "    Start Eagle, then open Lewdex from the Desktop or the Start menu and press Start."
Write-Host "    The first run tags the whole library (hours for a large one); later runs only new files."
