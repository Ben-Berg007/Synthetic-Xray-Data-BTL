@echo off
REM Train, QC-score, and generate for seeds 1-4 with the rule fixed on seed 0 (500 steps), then generate seed 0.
REM Run from D:\School\synthetic_data with the .venv activated:   scripts\run_remaining_seeds.bat
for %%S in (1 2 3 4) do (
  echo ===== seed %%S : train =====
  python scripts\03_train_lora.py --seed %%S --tag v2 --steps 500 --save_every 250 || goto :fail
  echo ===== seed %%S : QC scores =====
  python scripts\03b_eval_checkpoints.py --seed %%S --tag v2 || goto :fail
)
for %%S in (0 1 2 3 4) do (
  echo ===== seed %%S : generate =====
  python scripts\04_generate.py --seed %%S || goto :fail
)
echo ALL DONE
goto :eof
:fail
echo FAILED - see the error above
exit /b 1
