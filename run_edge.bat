@echo off
setlocal

set "APP_ROOT=%~dp0"
set "PADDLE_PDX_CACHE_HOME=%APP_ROOT%runtime\paddlex"
set "YOLO_CONFIG_DIR=%APP_ROOT%runtime\ultralytics"

if not exist "%APP_ROOT%runtime" mkdir "%APP_ROOT%runtime"
if not exist "%PADDLE_PDX_CACHE_HOME%" mkdir "%PADDLE_PDX_CACHE_HOME%"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
if not exist "%APP_ROOT%runtime\Saved_Image_Gibraltar" mkdir "%APP_ROOT%runtime\Saved_Image_Gibraltar"

if not exist "%APP_ROOT%.venv\Scripts\python.exe" (
    echo ERROR: Python environment is missing: %APP_ROOT%.venv
    echo Recreate it with Python 3.11 and install requirements.txt.
    exit /b 1
)

cd /d "%APP_ROOT%"
"%APP_ROOT%.venv\Scripts\python.exe" "%APP_ROOT%Projects\main.py" --ocr-device gpu:0 %*
exit /b %ERRORLEVEL%
