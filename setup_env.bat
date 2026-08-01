@echo off
REM Thin wrapper -- all logic lives in setup_env.py so this and setup_env.sh
REM can't drift. Usage: setup_env.bat [cpu^|cu121]   (default: cu121)
python "%~dp0setup_env.py" %*
