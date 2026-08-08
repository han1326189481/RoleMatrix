@echo off
chcp 65001 >nul
title RoleMatrix 微信接入启动器
cd /d D:\RoleMatrix

echo ============================================
echo  RoleMatrix 微信接入 - 一键启动
echo ============================================

REM 1. 启动 RoleMatrix 核心服务（识图/记忆/表情包/大脑）
echo [1/4] 启动 RoleMatrix 核心服务 (127.0.0.1:8765) ...
start "RoleMatrix-Server" cmd /k ".venv\Scripts\python.exe -m uvicorn rolematrix.bridge.server:create_app --factory --host 127.0.0.1 --port 8765"
timeout /t 12 /nobreak >nul

REM 2. 启动 OpenClaw Gateway（加载微信/Telegram/桥接插件）
echo [2/4] 启动 OpenClaw Gateway ...
start "OpenClaw-Gateway" cmd /k "tools\node-v22.23.2-win-x64\node.exe tools\node-v22.23.2-win-x64\node_modules\openclaw\openclaw.mjs gateway"
timeout /t 10 /nobreak >nul

REM 3. 微信登录（会弹出二维码，用微信扫码）
echo [3/4] 微信登录 - 请在弹出窗口里用手机微信扫码 ...
echo       提示：扫码后 bot 账号上线，手机端另一个微信给它发消息即可测试
timeout /t 3 /nobreak >nul
tools\node-v22.23.2-win-x64\node.exe tools\node-v22.23.2-win-x64\node_modules\openclaw\openclaw.mjs channels login --channel openclaw-weixin

echo.
echo [4/4] 微信登录完成（或已取消）。三个窗口都不要关：
echo   - RoleMatrix-Server : 小R 核心（人格/记忆/表情包）
echo   - OpenClaw-Gateway  : 消息网关
echo   登录窗口可关闭
echo.
echo 测试方法：
echo   1. 手机 A 微信（bot 扫码的那个号）在电脑上已登录
echo   2. 用手机 B（另一个微信）给 bot 号发消息
echo   3. 小R 会通过 RoleMatrix 自动回复，偶尔发你收藏的表情包
pause
