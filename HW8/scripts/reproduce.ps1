$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) { python -m venv .venv }
& '.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
& '.venv\Scripts\python.exe' -m src.mmip_rag.cli all
& '.venv\Scripts\python.exe' -m src.mmip_rag.ragas_eval
& '.venv\Scripts\python.exe' -m src.mmip_rag.reproduce
& '.venv\Scripts\python.exe' -m unittest discover -s tests -v
