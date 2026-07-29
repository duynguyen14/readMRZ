@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: read_image.bat path\to\image.jpg
  exit /b 1
)
dist\readmrz\readmrz.exe "%~1" --pretty
