@echo off
title Intel RealSense D455f Camera Test
echo ======================================================================
echo Starting Intel RealSense D455f Camera & Depth Verification Test...
echo ======================================================================

if exist "%~dp0..\..\.venv\Scripts\python.exe" (
    "%~dp0..\..\.venv\Scripts\python.exe" "%~dp0camera_test.py" %*
) else (
    python "%~dp0camera_test.py" %*
)
