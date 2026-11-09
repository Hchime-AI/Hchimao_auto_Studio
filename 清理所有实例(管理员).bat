@echo off
chcp 65001 >nul
title 清理所有实例
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\cleanup_admin.ps1"
