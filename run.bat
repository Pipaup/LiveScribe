@echo off
title LiveScribe

cd /d "%~dp0"

:: Try to find python from conda env, fall back to system python
set "PYTHON_CMD=python"

:: Check if a conda environment is configured
if not "%CONDA_PREFIX%"=="" (
    set "PYTHON_CMD=%CONDA_PREFIX%\python.exe"
)

echo Starting LiveScribe...
echo Press Ctrl+C to stop
echo.

"%PYTHON_CMD%" main.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] LiveScribe exited with error code %errorlevel%
    pause
)
