# extra_scripts/ — archived & advanced scripts

You normally don't need anything in here. The four files you run are in the
**project root**: `RUN_tacview.py`, `RUN_wez_tacview.py`, `RUN_eval.py`,
`RUN_train.py`.

This folder holds the development/diagnostic/extra-generation scripts so the
root stays clean. Notes:

- `eval_notch.py` is the win-rate evaluator that `RUN_eval.py` calls — leave it
  here.
- The other `gen_*.py`, `missile_*.py`, `wez_sweep_fine.py`, `calibrate_*.py`,
  `test_*.py`, `smoke_*.py` are advanced/diagnostic tools.
- **Heads-up:** several of the older `gen_*.py` scripts have a hard-coded
  output path from the original Linux/WSL machine and will not write correctly
  on Windows as-is. The root `RUN_*.py` scripts do NOT have this problem (they
  use portable, relative paths). If you want to use one of these, run it from
  the project root and change its output path to a local folder.

To run any of these from the root with the ▶ button, open the file and edit its
output path first, or prefer the equivalent `RUN_*.py` at the root.
