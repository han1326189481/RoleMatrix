@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
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
REM 从 openclaw.json 读取 DeepSeek key 注入环境变量（小R 嘴巴需要）
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-Content .openclaw\openclaw.json -Raw | ConvertFrom-Json).env.DEEPSEEK_API_KEY"`) do set "DEEPSEEK_API_KEY=%%i"
if not defined DEEPSEEK_API_KEY set "DEEPSEEK_API_KEY="
start "RoleMatrix-Server" cmd /k ".venv\Scripts\python.exe -m uvicorn rolematrix.bridge.server:create_app --factory --host 127.0.0.1 --port 8765"
timeout /t 12 /nobreak >nul

REM 2. 启动 OpenClaw Gateway（加载微信/Telegram/桥接插件）
echo [2/3] 启动 OpenClaw Gateway ...
start "OpenClaw-Gateway" cmd /k "tools\node-v22.23.2-win-x64\node.exe tools\node-v22.23.2-win-x64\node_modules\openclaw\openclaw.mjs gateway"
timeout /t 10 /nobreak >nul

REM 3. 微信登录：提取登录链接，用浏览器打开标准二维码
echo [3/3] 正在获取微信登录二维码 ...
echo       浏览器会自动打开二维码页面，用【平板上的微信小号】扫码
timeout /t 3 /nobreak >nul
set "WX_LOGIN_LOG=%TEMP%\wx_login_output.txt"
REM 后台启动 login（阻塞等待扫码），先把登录链接写到文件
start /b "" tools\node-v22.23.2-win-x64\node.exe tools\node-v22.23.2-win-x64\node_modules\openclaw\openclaw.mjs channels login --channel openclaw-weixin > "%WX_LOGIN_LOG%" 2>&1
REM 等 login 输出登录链接
timeout /t 8 /nobreak >nul
REM 从 login 输出中提取 liteapp 登录链接
set "WX_URL="
for /f "usebackq tokens=*" %%i in (`findstr /R "https://liteapp.weixin.qq.com/q/" "%WX_LOGIN_LOG%"`) do set "WX_URL=%%i"
if defined WX_URL (
  start "" "!WX_URL!"
  echo       已打开二维码页面（如果没弹出，手动复制这行到浏览器）：
  echo       !WX_URL!
) else (
  echo       未能自动提取登录链接，正在等 login 输出 ... 请稍候
  timeout /t 10 /nobreak >nul
  for /f "usebackq tokens=*" %%i in (`findstr /R "https://liteapp.weixin.qq.com/q/" "%WX_LOGIN_LOG%"`) do set "WX_URL=%%i"
  if defined WX_URL (
    start "" "!WX_URL!"
    echo       已打开二维码页面：!WX_URL!
  ) else (
    echo       仍未提取到链接，请打开文件 %WX_LOGIN_LOG% 查看二维码/链接
  )
)

echo.
echo 扫码完成后：
echo   - 用手机【大号微信】给小号发消息，小R 就会自动回复
echo   - 想发图片/表情包测试也可以
echo.
echo 提示：登录窗口显示二维码时，扫码后会自动跳转，窗口出现"登录成功"字样
pause
