@echo off
chcp 65001 >nul
title RoleMatrix 微信接入启动器
cd /d D:\RoleMatrix

echo ============================================
echo  RoleMatrix 微信接入 - 一键启动
echo ============================================
echo.
echo 需要的设备：
echo   - 平板：登录【微信小号】（用来扫码）
echo   - 手机：登录【微信大号】（用来发消息测试）
echo.
echo 接下来会依次弹出 3 个窗口，都别关：
echo   1. RoleMatrix-Server  小R 核心服务
echo   2. OpenClaw-Gateway   消息网关
echo   3. 微信扫码            显示二维码（用平板小号微信扫）
echo.

REM 1. 启动 RoleMatrix 核心服务（识图/记忆/表情包/大脑）
echo [1/3] 启动 RoleMatrix 核心服务 (127.0.0.1:8765) ...
start "RoleMatrix-Server" cmd /k ".venv\Scripts\python.exe -m uvicorn rolematrix.bridge.server:create_app --factory --host 127.0.0.1 --port 8765"
timeout /t 12 /nobreak >nul

REM 2. 启动 OpenClaw Gateway（加载微信/Telegram/桥接插件）
echo [2/3] 启动 OpenClaw Gateway ...
start "OpenClaw-Gateway" cmd /k "tools\node-v22.23.2-win-x64\node.exe tools\node-v22.23.2-win-x64\node_modules\openclaw\openclaw.mjs gateway"
timeout /t 10 /nobreak >nul

REM 3. 微信登录：独立窗口显示二维码
echo [3/3] 打开【微信扫码】窗口 ...
echo       请用【平板上的微信小号】扫描二维码
echo       如果二维码显示异常，把窗口里的 https://liteapp.weixin.qq.com/... 链接
echo       发到手机上打开，也可以扫码
timeout /t 3 /nobreak >nul
start "微信扫码" cmd /k "tools\node-v22.23.2-win-x64\node.exe tools\node-v22.23.2-win-x64\node_modules\openclaw\openclaw.mjs channels login --channel openclaw-weixin"

echo.
echo 扫码完成后：
echo   - 用手机【大号微信】给小号发消息，小R 就会自动回复
echo   - 想发图片/表情包测试也可以
echo.
echo 提示：登录窗口显示二维码时，扫码后会自动跳转，窗口出现"登录成功"字样
pause
