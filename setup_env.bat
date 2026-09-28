@echo off
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"
where uv >nul 2>&1
if errorlevel 1 (
    echo uv not found on PATH. Install from https://astral.sh/uv and re-run.
    exit /b 1
)
uv venv .venv
call .venv\Scripts\activate.bat
uv pip install -r requirements.txt
echo Environment ready at %ROOT%\.venv
