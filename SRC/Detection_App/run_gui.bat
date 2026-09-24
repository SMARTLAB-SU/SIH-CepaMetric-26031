@echo off
title Intel RealSense D455f Object Measurement System
echo ======================================================================
echo Starting Intel RealSense D455f Live Object Measurement System...
echo ======================================================================

if exist "%~dp0..\..\.venv\Scripts\python.exe" (
    "%~dp0..\..\.venv\Scripts\python.exe" "%~dp0main.py" %*
) else (
    python "%~dp0main.py" %*
)
