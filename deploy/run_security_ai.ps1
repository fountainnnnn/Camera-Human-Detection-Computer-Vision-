$ErrorActionPreference = "Stop"

param(
    [int]$RestartDelaySeconds = 5,
    [switch]$RunOnce
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$MainScript = Join-Path $ProjectRoot "src\main.py"
$LogDir = Join-Path $ProjectRoot "logs"
$RuntimeLog = Join-Path $LogDir "runtime.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path $PythonExe)) {
    throw "Python virtual environment not found at $PythonExe"
}

function Write-RunnerLog {
    param(
        [string]$Message
    )

    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    Add-Content -Path $RuntimeLog -Value "[$timestamp] runner $Message"
}

Push-Location $ProjectRoot
try {
    while ($true) {
        Write-RunnerLog "starting runtime"
        & $PythonExe $MainScript *>> $RuntimeLog
        $exitCode = $LASTEXITCODE

        if ($RunOnce -or $exitCode -eq 0) {
            Write-RunnerLog "runtime exited with code $exitCode; runner stopping"
            exit $exitCode
        }

        Write-RunnerLog "runtime exited with code $exitCode; restarting in $RestartDelaySeconds second(s)"
        Start-Sleep -Seconds $RestartDelaySeconds
    }
} finally {
    Pop-Location
}
