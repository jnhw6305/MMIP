$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    python -m venv .venv
}

& $Python -c 'import streamlit' 2>$null
if ($LASTEXITCODE -ne 0) {
    & $Python -m pip install -r requirements-lock.txt
}

& $Python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
