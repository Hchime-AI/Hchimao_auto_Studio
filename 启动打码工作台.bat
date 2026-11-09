@echo off
chcp 65001 >nul
title 打码工作台
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start.ps1" %*
