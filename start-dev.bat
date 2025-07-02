@echo off
:: Start AlgoSpace Development Environment on Windows

echo Starting AlgoSpace Development Environment...

:: Check Docker status
docker info 2>nul | findstr "OSType: linux" >nul
if errorlevel 1 (
    echo ERROR: Docker is not in Linux mode!
    echo.
    echo Attempting to switch to Linux containers...
    "C:\Program Files\Docker\Docker\DockerCli.exe" -SwitchLinuxEngine 2>nul
    if errorlevel 1 (
        echo Failed to switch automatically. Please switch Docker to Linux containers manually.
        echo Right-click Docker icon in system tray and select "Switch to Linux containers..."
        pause
        exit /b 1
    )
    echo Waiting for Docker to switch modes...
    timeout /t 10 /nobreak >nul
)

:: Check if image exists
docker images | findstr algospace-env >nul
if errorlevel 1 (
    echo Building Docker image...
    docker build -t algospace-env -f Dockerfile.light .
    if errorlevel 1 (
        echo Failed with light Dockerfile, trying main Dockerfile...
        docker build -t algospace-env .
    )
)

:: Run container
echo Launching container...
docker run -it --rm ^
    -v "%cd%:/app" ^
    -w /app ^
    -p 8000:8000 ^
    -p 8888:8888 ^
    --name algospace-dev ^
    algospace-env bash

echo Development session ended.
pause