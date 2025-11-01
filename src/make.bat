@echo off
REM =====================================================
REM Build Python script into .exe with PyInstaller
REM Includes: TranningData, YOLO model, face_recognition models
REM =====================================================

REM --- Configuration ---
set SCRIPT_NAME=main.py
set DIST_FOLDER=dist
set TRANNING_DATA=TranningData
set YOLO_MODEL=yolov8n-pose.pt
set FACE_MODELS="D:\Arduino Related\FRAL\.venv\Lib\site-packages\face_recognition_models\models"

REM --- Clean previous builds ---
if exist build rmdir /s /q build
if exist %DIST_FOLDER% rmdir /s /q %DIST_FOLDER%
if exist %SCRIPT_NAME:.py=.exe% del /q %SCRIPT_NAME:.py=.exe%

REM --- Build .exe using PyInstaller ---
echo [INFO] Building %SCRIPT_NAME%...
pyinstaller --onefile ^
--add-data "%TRANNING_DATA%;%TRANNING_DATA%" ^
--add-data "%YOLO_MODEL%;." ^
--add-data %FACE_MODELS%;face_recognition_models\models ^
%SCRIPT_NAME%

REM --- Check if build succeeded ---
if exist "%DIST_FOLDER%\%SCRIPT_NAME:.py=.exe%" (
    echo [INFO] Build successful!
    echo [INFO] Running the .exe...
    "%DIST_FOLDER%\%SCRIPT_NAME:.py=.exe%"
) else (
    echo [ERROR] Build failed.
)

pause
