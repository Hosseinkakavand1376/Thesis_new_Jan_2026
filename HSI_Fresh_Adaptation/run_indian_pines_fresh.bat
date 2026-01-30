@echo off
echo ============================================
echo FRESH INDIAN PINES PIPELINE RUN
echo ============================================
echo Starting at: %date% %time%
echo.

cd /d C:\Users\hosse\Desktop\Thesis_Phase1_completed\HSI_Fresh_Adaptation

echo.
echo [1/4] Running SG + SVN...
echo ============================================
python WST_script.py --dataset indian_pines --preprocessing SG SVN
if %errorlevel% neq 0 (
    echo ERROR: SG+SVN failed!
    pause
    exit /b 1
)
echo SG+SVN completed at: %time%

echo.
echo [2/4] Running SG + MSC...
echo ============================================
python WST_script.py --dataset indian_pines --preprocessing SG MSC
if %errorlevel% neq 0 (
    echo ERROR: SG+MSC failed!
    pause
    exit /b 1
)
echo SG+MSC completed at: %time%

echo.
echo ============================================
echo ALL INDIAN PINES COMBINATIONS COMPLETE!
echo Finished at: %date% %time%
echo ============================================
pause
