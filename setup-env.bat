@echo off
title LiveScribe - Environment Setup

echo ============================================
echo  LiveScribe Environment Setup
echo ============================================
echo.

where conda >nul 2>&1
if errorlevel 1 (
    echo [INFO] conda not found in PATH, using pip directly
    goto :pip_install
)

:: Try to find or create conda env
set "CONDA_ENV_NAME=qwen3-asr"
conda info --envs | findstr /C:"%CONDA_ENV_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Found existing conda env '%CONDA_ENV_NAME%'
    call conda activate %CONDA_ENV_NAME%
    goto :install_deps
)

echo [1/3] Creating conda environment: %CONDA_ENV_NAME%
call conda create -n %CONDA_ENV_NAME% python=3.11 -y
if errorlevel 1 (
    echo [WARN] Failed to create conda env, falling back to pip
    goto :pip_install
)

call conda activate %CONDA_ENV_NAME%

:install_deps
echo.
echo [2/3] Installing PyTorch with CUDA support...
pip install torch>=2.6.0 torchaudio>=2.6.0

echo.
echo [3/3] Installing project dependencies...
cd /d "%~dp0"
pip install -r requirements.txt
pip install -e .

echo.
echo ============================================
echo  Setup Complete!
echo ============================================
echo.
echo Next steps:
echo   1. Download models to ./models/ directory
echo      (Qwen3-ASR: https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
echo   2. Copy config.example.yaml to config.yaml and edit
echo   3. Run: double-click run.bat
echo.
pause
goto :eof

:pip_install
echo.
echo Installing with pip...
cd /d "%~dp0"
pip install -r requirements.txt
echo.
echo Setup complete! See README.md for next steps.
pause
