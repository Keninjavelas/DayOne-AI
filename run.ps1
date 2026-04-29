param(
    [switch]$SkipIngest,
    [switch]$SkipInstall,
    [switch]$NoStreamlit,
    [switch]$UseSeparateTerminals
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$envPath = Join-Path $projectRoot '.env'
if (-not (Test-Path $envPath)) {
    throw 'CRITICAL: .env file not found. Cannot start system.'
}

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Step,
        [switch]$AllowFailure
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        if ($AllowFailure) {
            Write-Warning "$Step failed with exit code $LASTEXITCODE. Continuing startup."
            return
        }

        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

function Stop-DayOneJobs {
    $jobs = Get-Job -Name 'DayOne-*' -ErrorAction SilentlyContinue
    if (-not $jobs) {
        return
    }

    foreach ($job in $jobs) {
        try {
            Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
        }
        catch {
            Write-Warning "Unable to stop job $($job.Name): $($_.Exception.Message)"
        }
    }

    foreach ($job in $jobs) {
        try {
            Remove-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
        }
        catch {
            Write-Warning "Unable to remove job $($job.Name): $($_.Exception.Message)"
        }
    }
}

function Invoke-IngestWithRetry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [int]$MaxAttempts = 3
    )

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-Host ("Running initial ingest (attempt {0}/{1})..." -f $attempt, $MaxAttempts) -ForegroundColor Cyan
        & $PythonExe ingest.py

        if ($LASTEXITCODE -eq 0) {
            return $true
        }

        Write-Warning ("Initial ingest attempt {0} failed with exit code {1}." -f $attempt, $LASTEXITCODE)

        if ($attempt -lt $MaxAttempts) {
            Write-Host 'Attempting lock recovery (stopping existing DayOne background jobs) before retry...' -ForegroundColor DarkYellow
            Stop-DayOneJobs
            Start-Sleep -Seconds ([Math]::Min(3, $attempt))
        }
    }

    return $false
}

function Test-DatabaseConnection {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe
    )

    & $PythonExe -c "import os; from dotenv import load_dotenv; from sqlalchemy import create_engine, text; load_dotenv(); url=os.getenv('DATABASE_URL'); assert url, 'DATABASE_URL is required'; engine=create_engine(url, pool_pre_ping=True); conn=engine.connect(); conn.execute(text('SELECT 1')); conn.close(); print('DB OK')"
    return ($LASTEXITCODE -eq 0)
}

function Ensure-DatabaseReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    Write-Host 'Checking database connectivity...' -ForegroundColor Cyan
    if (Test-DatabaseConnection -PythonExe $PythonExe) {
        return
    }

    $composeFile = Join-Path $ProjectRoot 'infra\docker-compose.yml'
    if (Test-Path $composeFile) {
        $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
        if ($dockerCmd) {
            Write-Host 'Database is unavailable. Attempting to start local postgres via docker compose...' -ForegroundColor DarkYellow
            & $dockerCmd.Source compose -f $composeFile up -d postgres
            if ($LASTEXITCODE -ne 0) {
                throw 'CRITICAL: Failed to start postgres container via docker compose.'
            }

            for ($i = 1; $i -le 10; $i++) {
                if (Test-DatabaseConnection -PythonExe $PythonExe) {
                    return
                }
                Start-Sleep -Seconds 2
            }
        }
    }

    throw 'CRITICAL: Database is not reachable. Ensure postgres is running and DATABASE_URL points to the correct host/port.'
}

function Start-BackgroundProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,
        [string]$WorkingDirectory = $PWD.Path,
        [hashtable]$Env = @{}
    )

    $runtimeLogDir = Join-Path $projectRoot 'logs\runtime'
    New-Item -ItemType Directory -Path $runtimeLogDir -Force | Out-Null

    $stdoutPath = Join-Path $runtimeLogDir ("{0}.out.log" -f $Name)
    $stderrPath = Join-Path $runtimeLogDir ("{0}.err.log" -f $Name)

    $jobName = "DayOne-$Name"
    $existing = Get-Job -Name $jobName -ErrorAction SilentlyContinue
    if ($existing) {
        $existing | Stop-Job -ErrorAction SilentlyContinue
        $existing | Remove-Job -ErrorAction SilentlyContinue
    }

    $job = Start-Job -Name $jobName -ScriptBlock {
        param(
            [string]$wd,
            [string]$exe,
            [string[]]$args,
            [string]$stdout,
            [string]$stderr,
            [hashtable]$envMap
        )

        Set-Location $wd
        foreach ($entry in $envMap.GetEnumerator()) {
            Set-Item -Path ("Env:{0}" -f $entry.Key) -Value ([string]$entry.Value)
        }

        & $exe @args 1>> $stdout 2>> $stderr
    } -ArgumentList $WorkingDirectory, $FilePath, $ArgumentList, $stdoutPath, $stderrPath, $Env

    Write-Host ("Started {0} (Job {1}). Logs: {2}" -f $Name, $job.Id, $stdoutPath) -ForegroundColor DarkCyan
}

$venvDir = Join-Path $projectRoot '.venv'
$pythonExe = Join-Path $venvDir 'Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        throw 'Python is not installed or not on PATH. Install Python 3.10+ and re-run .\run.ps1.'
    }

    Write-Host 'Creating virtual environment (.venv)...' -ForegroundColor Cyan
    & $pythonCmd.Source -m venv $venvDir
}

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe after virtual environment creation."
}

# Some Windows shells leak global Python vars that cause '<prefix>' startup warnings.
if (Test-Path Env:PYTHONHOME) { Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue }
if (Test-Path Env:PYTHONPATH) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }

# Force the expected local Postgres endpoint for this shell session.
$env:DATABASE_URL = "postgresql+psycopg://dayone:dayone@localhost:55432/dayone"

if (-not $SkipInstall) {
    Write-Host 'Installing Python dependencies...' -ForegroundColor Cyan
    Invoke-ExternalCommand -Step 'pip upgrade' -Command { & $pythonExe -m pip install --upgrade pip }
    Invoke-ExternalCommand -Step 'pip install requirements' -Command { & $pythonExe -m pip install -r (Join-Path $projectRoot 'requirements.txt') }

    $npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    }
    if (-not $npmCmd) {
        throw 'npm is not installed or not on PATH. Install Node.js 18+ and re-run .\run.ps1.'
    }

    Write-Host 'Installing Node.js dependencies...' -ForegroundColor Cyan
    Invoke-ExternalCommand -Step 'npm install' -Command { & $npmCmd.Source install }
}

Ensure-DatabaseReady -PythonExe $pythonExe -ProjectRoot $projectRoot

if (-not $SkipIngest) {
    Write-Host "Using DATABASE_URL=$env:DATABASE_URL"
    if (-not (Invoke-IngestWithRetry -PythonExe $pythonExe -MaxAttempts 3)) {
        throw 'CRITICAL: Ingestion failed. Aborting startup.'
    }

    $verificationJson = & $pythonExe -c "import json; from backend.services.auth_db import require_engine; from sqlalchemy import text; e=require_engine();
with e.connect() as c:
 count=int(c.execute(text('SELECT COUNT(*) FROM embeddings')).scalar() or 0)
 row=c.execute(text('SELECT e.tenant_id::text AS tenant_id FROM embeddings e GROUP BY e.tenant_id ORDER BY COUNT(*) DESC LIMIT 1')).mappings().first()
 tenant_id=(row['tenant_id'] if row else '')
print(json.dumps({'embedding_count': count, 'tenant_id': tenant_id}))"
    if ($LASTEXITCODE -ne 0) {
        throw 'CRITICAL: Failed to verify embeddings after ingestion.'
    }

    $verification = $verificationJson | ConvertFrom-Json
    $embeddingCount = [int]$verification.embedding_count
    $tenantId = [string]$verification.tenant_id

    if ($tenantId) {
        Write-Host ("[INFO] Tenant: {0}" -f $tenantId) -ForegroundColor DarkCyan
    }

    if ($embeddingCount -gt 0) {
        Write-Host ("[OK] Embeddings loaded: {0}" -f $embeddingCount) -ForegroundColor Green
        Write-Host '[OK] System ready' -ForegroundColor Green
    }
    else {
        Write-Host '[ERROR] No embeddings found' -ForegroundColor Red
        Write-Host '[ERROR] System is NOT ready' -ForegroundColor Red
        throw 'CRITICAL: No embeddings found after ingestion. Aborting startup.'
    }
}

if ($UseSeparateTerminals) {
    Write-Host 'Starting auto-ingest watcher in a new terminal...' -ForegroundColor Cyan
    $watcherCommand = "Set-Location '$projectRoot'; & '$pythonExe' auto_ingest.py"
    Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $watcherCommand | Out-Null

    Write-Host 'Starting FastAPI backend in a new terminal...' -ForegroundColor Cyan
    $apiCommand = "Set-Location '$projectRoot'; & '$pythonExe' -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
    Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $apiCommand | Out-Null

    Write-Host 'Starting Next.js frontend in a new terminal...' -ForegroundColor Cyan
    $frontendCommand = "Set-Location '$projectRoot'; `$env:NEXT_PUBLIC_API_BASE_URL='http://127.0.0.1:8000'; npm.cmd run dev -- -p 3000"
    Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $frontendCommand | Out-Null
}
else {
    Write-Host 'Starting auto-ingest watcher in background (same terminal session)...' -ForegroundColor Cyan
    Start-BackgroundProcess -Name 'watcher' -FilePath $pythonExe -ArgumentList @('auto_ingest.py') -WorkingDirectory $projectRoot

    Write-Host 'Starting FastAPI backend in background (same terminal session)...' -ForegroundColor Cyan
    Start-BackgroundProcess -Name 'api' -FilePath $pythonExe -ArgumentList @('-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000', '--reload') -WorkingDirectory $projectRoot

    $npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    }
    if (-not $npmCmd) {
        throw 'npm is not installed or not on PATH. Install Node.js 18+ and re-run .\run.ps1.'
    }

    Write-Host 'Starting Next.js frontend in background (same terminal session)...' -ForegroundColor Cyan
    Start-BackgroundProcess `
        -Name 'frontend' `
        -FilePath $npmCmd.Source `
        -ArgumentList @('run', 'dev', '--', '-p', '3000') `
        -WorkingDirectory $projectRoot `
        -Env @{ NEXT_PUBLIC_API_BASE_URL = 'http://127.0.0.1:8000' }

    Write-Host "Background logs: $projectRoot\logs\runtime" -ForegroundColor Green
}

if ($NoStreamlit) {
    Write-Host 'Startup complete.' -ForegroundColor Green
    Write-Host 'Frontend: http://localhost:3000' -ForegroundColor Green
    Write-Host 'Backend : http://127.0.0.1:8000' -ForegroundColor Green
    if ($UseSeparateTerminals) {
        Write-Host 'Watcher : running in separate terminal' -ForegroundColor Green
    }
    else {
        Write-Host 'Watcher : running in background process (same terminal session)' -ForegroundColor Green
        Write-Host "Logs    : $projectRoot\logs\runtime" -ForegroundColor Green
        Write-Host "Stop all: Get-Job DayOne-* | Stop-Job ; Get-Job DayOne-* | Remove-Job" -ForegroundColor Green
    }
    return
}

Write-Host 'Starting Streamlit app (current terminal)...' -ForegroundColor Green
Write-Host 'Streamlit: http://localhost:8501' -ForegroundColor Green
& $pythonExe -m streamlit run app.py
