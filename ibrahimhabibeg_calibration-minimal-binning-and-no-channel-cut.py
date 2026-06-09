! cp -r /kaggle/input/ariel-2025-repo/ariel-2025 .
! ln -s /usr/bin/python3 ariel-2025/.venv/bin/python
! ln -s ariel-2025/.venv/bin/python ariel-2025/.venv/bin/python3
! ln -s ariel-2025/.venv/bin/python ariel-2025/.venv/bin/python3.11


! cd ariel-2025/scripts && uv run sergei_pipeline.py  --input-data-folder /kaggle/input/ariel-data-challenge-2025 --output-data-folder /kaggle/working/calibrated --stop-at-calibration --binning 4


! rm -r ariel-2025


! ls /kaggle/working/calibrated

