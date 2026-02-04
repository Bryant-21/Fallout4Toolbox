@echo off
setlocal

REM Activate the conda environment "fo4"
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" fo4 || goto :eof

REM Switch to the script's directory and run the app
pushd "%~dp0"
python FalloutToolbox.py
popd