@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === OVDeploy 上传到 GitHub ===
echo.
echo [1/3] 检查 GitHub 连接...
powershell -NoProfile -Command "try { $c=New-Object Net.Sockets.TcpClient('github.com',443); $c.Close(); exit 0 } catch { Write-Host '无法连接 github.com:443，请换网络或开 VPN'; exit 1 }"
if errorlevel 1 pause & exit /b 1
echo.
echo [2/3] GitHub 登录（浏览器授权）...
gh auth login --hostname github.com --git-protocol https --web
if errorlevel 1 pause & exit /b 1
echo.
echo [3/3] 创建公开仓库 OVDeploy 并推送...
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\push_to_github.ps1"
if errorlevel 1 pause & exit /b 1
echo.
echo 完成！
pause
