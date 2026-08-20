@echo off
setlocal

set "APP_ROOT=%~dp0"
set "USERPROFILE=%APP_ROOT%.runtime"
set "HOME=%APP_ROOT%.runtime"
set "PADDLE_PDX_CACHE_HOME=%APP_ROOT%.runtime\paddlex"
set "YOLO_CONFIG_DIR=%APP_ROOT%.runtime\ultralytics"

if not exist "%USERPROFILE%" mkdir "%USERPROFILE%"
if not exist "%APP_ROOT%runtime\Saved_Image_Gibraltar" mkdir "%APP_ROOT%runtime\Saved_Image_Gibraltar"

if not exist "%APP_ROOT%.venv\Scripts\python.exe" (
    echo ERROR: Python environment is missing: %APP_ROOT%.venv
    echo Recreate it with Python 3.11 and install requirements.txt.
    exit /b 1
)

cd /d "%APP_ROOT%"
call "%APP_ROOT%.venv\Scripts\activate.bat"
python "%APP_ROOT%Projects\main.py" --ocr-device cpu %*
exit /b %ERRORLEVEL%
