@echo off
setlocal

cd /d %~dp0\..

echo [build] building trendradar local image...
docker build -f docker/Dockerfile -t trendradar:local .
if errorlevel 1 exit /b 1

echo [build] building trendradar mcp local image...
docker build -f docker/Dockerfile.mcp -t trendradar-mcp:local .
if errorlevel 1 exit /b 1

echo [done] images:
docker images | findstr /I "trendradar trendradar-mcp"

endlocal

