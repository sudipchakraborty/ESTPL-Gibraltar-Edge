$ErrorActionPreference = "Stop"

$repositoryDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $repositoryDirectory ".venv\Scripts\python.exe"
$applicationScript = Join-Path $repositoryDirectory "Projects\Jutemill\main.py"

if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw "Virtual-environment Python was not found: $pythonExecutable"
}

Write-Host "[LOCAL TEST] Jutemill will start its embedded AlertServer."
Write-Host "[LOCAL TEST] Connect Postman to http://127.0.0.1:5000"
Write-Host "[LOCAL TEST] Listen for event: alert_received"

& $pythonExecutable $applicationScript
