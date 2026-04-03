param(
    [ValidateSet("local", "dev", "pr", "qa", "prod")]
    [string]$Env = "local",

    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env.$Env"

if (-not (Test-Path $envFile)) {
    throw "Environment file not found: $envFile"
}

Write-Host "== upnews-pipeline timing =="
Write-Host "Env file: $envFile"

$buildTime = $null
if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "[1/2] Building image..."
    $buildTime = Measure-Command {
        docker compose --env-file $envFile build pipeline
    }
    Write-Host ("Build finished in {0:N1}s" -f $buildTime.TotalSeconds)
}

Write-Host ""
Write-Host "[2/2] Running pipeline container..."
$runTime = Measure-Command {
    docker compose --env-file $envFile run --rm pipeline
}
Write-Host ("Run finished in {0:N1}s" -f $runTime.TotalSeconds)

if ($buildTime) {
    $totalSeconds = $buildTime.TotalSeconds + $runTime.TotalSeconds
    Write-Host ("Total timed duration: {0:N1}s" -f $totalSeconds)
}
