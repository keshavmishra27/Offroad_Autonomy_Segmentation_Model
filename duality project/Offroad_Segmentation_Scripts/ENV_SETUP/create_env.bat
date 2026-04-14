@echo off

:: Check if conda is available in the system
where conda >nul 2>nul
IF ERRORLEVEL 1 (
    if exist "C:\Users\hp\miniconda3\Scripts\conda.exe" (
        set "CONDA_PATH="C:\Users\hp\miniconda3\Scripts\conda.exe""
    ) else (
        echo "Conda is not found in your system. Please install Miniconda or Anaconda first."
        pause
        exit /b 1
    )
) else (
    set "CONDA_PATH=conda"
)

:: Create the environment
echo Creating the Conda environment 'EDU' with Python 3.10...
%CONDA_PATH% create --name EDU python=3.10 -y

echo Environment 'EDU' created successfully.
pause
