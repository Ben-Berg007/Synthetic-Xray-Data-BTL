@echo off
REM Seeds 5-9: same fixed rule as seeds 0-4 (500 LoRA steps), QC scores, then 1,000 images/class.
REM Run from D:\School\synthetic_data with the .venv activated:   scripts\run_seeds_5_9.bat
for %%S in (5 6 7 8 9) do (
  echo ===== seed %%S : train =====
  python scripts\03_train_lora.py --seed %%S --tag v2 --steps 500 --save_every 250 || goto :fail
  echo ===== seed %%S : QC scores =====
  python scripts\03b_eval_checkpoints.py --seed %%S --tag v2 || goto :fail
  echo ===== seed %%S : generate =====
  python scripts\04_generate.py --seed %%S || goto :fail
)
echo ===== classifiers, all 10 seeds =====
python scripts\05_train_classifiers.py --seeds 0 1 2 3 4 5 6 7 8 9 || goto :fail
echo ===== statistics =====
python scripts\06_stats.py || goto :fail
echo ALL DONE
goto :eof
:fail
echo FAILED - see the error above
exit /b 1
