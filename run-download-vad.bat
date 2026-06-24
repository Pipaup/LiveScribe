@echo off
title LiveScribe - Download VAD Model

cd /d "%~dp0"

:: Try to find python, prefer conda env if active
set "PYTHON_CMD=python"
if not "%CONDA_PREFIX%"=="" (
    set "PYTHON_CMD=%CONDA_PREFIX%\python.exe"
)

"%PYTHON_CMD%" main.py --download-vad
pause
