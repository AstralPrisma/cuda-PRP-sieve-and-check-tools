@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-all-windows.ps1" %*
exit /b %errorlevel%
