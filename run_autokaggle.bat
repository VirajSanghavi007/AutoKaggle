@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "VENV=%ROOT%\.venv"
set "LOGDIR=%ROOT%\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set TODAY=%%c%%a%%b
set "RUNLOG=%LOGDIR%\run_%TODAY%.log"

echo ===== AutoKaggle run started %date% %time% ===== >> "%RUNLOG%"

where kaggle >nul 2>&1
if errorlevel 1 (
    echo [FAIL] kaggle CLI not found on PATH. >> "%RUNLOG%"
    exit /b 1
)
if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo [FAIL] kaggle.json not found at %USERPROFILE%\.kaggle\kaggle.json - run `kaggle config` or place credentials. >> "%RUNLOG%"
    exit /b 1
)
where git >nul 2>&1
if errorlevel 1 (
    echo [FAIL] git not found on PATH. >> "%RUNLOG%"
    exit /b 1
)
if not exist "%ROOT%\kaggledataset.txt" (
    echo [FAIL] kaggledataset.txt not found at %ROOT%\kaggledataset.txt >> "%RUNLOG%"
    exit /b 1
)

if exist "%VENV%\Scripts\activate.bat" (
    call "%VENV%\Scripts\activate.bat"
)

cd /d "%ROOT%\src"

echo [INFO] Running prepare_job.py (download -> EDA) >> "%RUNLOG%"
python prepare_job.py >> "%RUNLOG%" 2>&1
set "PREPARE_EXIT=%ERRORLEVEL%"

if "%PREPARE_EXIT%"=="2" (
    echo [INFO] kaggledataset.txt was empty - nothing to do today. >> "%RUNLOG%"
    exit /b 0
)
if not "%PREPARE_EXIT%"=="0" (
    echo [FAIL] prepare_job.py exited with code %PREPARE_EXIT% - aborting. >> "%RUNLOG%"
    exit /b 1
)

echo [INFO] Running run_pipeline.py (benchmark -> notebook -> publish) >> "%RUNLOG%"
echo [INFO] This step requires the Claude Co-Work Routine to have already written pipeline_config.json for today's job. >> "%RUNLOG%"
python run_pipeline.py >> "%RUNLOG%" 2>&1
set "PIPELINE_EXIT=%ERRORLEVEL%"

if "%PIPELINE_EXIT%"=="2" (
    echo [INFO] pipeline_config.json not ready yet - waiting on Claude Co-Work Routine, will retry next run. >> "%RUNLOG%"
    exit /b 0
)
if not "%PIPELINE_EXIT%"=="0" (
    echo [FAIL] run_pipeline.py exited with code %PIPELINE_EXIT% - see log above. >> "%RUNLOG%"
    exit /b 1
)

echo [INFO] run_pipeline.py succeeded - pushing notebook to Kaggle >> "%RUNLOG%"
python kaggle_push.py >> "%RUNLOG%" 2>&1
if errorlevel 1 (
    echo [FAIL] kaggle push failed - see log above. >> "%RUNLOG%"
    exit /b 1
)

echo [INFO] Kaggle push succeeded >> "%RUNLOG%"
echo ===== AutoKaggle run finished %date% %time% ===== >> "%RUNLOG%"
exit /b 0
