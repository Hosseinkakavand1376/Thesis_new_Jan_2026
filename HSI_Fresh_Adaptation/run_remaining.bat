@echo off
echo ============================================
echo Running remaining preprocessing combinations
echo Started at: %date% %time%
echo ============================================

cd /d C:\Users\hosse\Desktop\Thesis_Phase1_completed\HSI_Fresh_Adaptation

echo.
echo [1/2] Running SG1 + SVN...
echo ============================================
python WST_script.py --dataset indian_pines --preprocessing SG1 SVN

echo.
echo [2/2] Running SG1 + MSC...
echo ============================================
python WST_script.py --dataset indian_pines --preprocessing SG1 MSC

echo.
echo ============================================
echo All preprocessing combinations completed!
echo Finished at: %date% %time%
echo ============================================
pause
