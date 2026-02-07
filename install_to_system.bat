@echo off
echo Installing ToneMatch AI VST3 to system folder...
echo.

set "SOURCE=E:\Users\Desktop\toneMatchAi\plugin\build\ToneMatchAI_artefacts\Release\VST3\ToneMatch AI.vst3"
set "DEST=C:\Program Files\Common Files\VST3\ToneMatch AI.vst3"

if not exist "%SOURCE%" (
    echo ERROR: Plugin not found at: %SOURCE%
    echo Please compile the plugin first.
    pause
    exit /b 1
)

echo Copying plugin...
if exist "%DEST%" (
    rmdir /s /q "%DEST%"
)
xcopy /E /I /Y "%SOURCE%" "%DEST%"

if exist "%DEST%" (
    echo.
    echo Deploying Python resources...
    set "RESOURCES=%DEST%\Contents\Resources"
    if not exist "%RESOURCES%" mkdir "%RESOURCES%"
    
    copy /Y "E:\Users\Desktop\toneMatchAi\plugin\Scripts\run_match.py" "%RESOURCES%\" >nul
    xcopy /E /I /Y "E:\Users\Desktop\toneMatchAi\src" "%RESOURCES%\src" >nul
    
    echo.
    echo ============================================================
    echo SUCCESS: Plugin installed to:
    echo %DEST%
    echo ============================================================
    echo.
    echo Plugin is ready to use in FL Studio!
) else (
    echo.
    echo ERROR: Failed to copy plugin
    echo You may need to run this script as Administrator
)

pause

