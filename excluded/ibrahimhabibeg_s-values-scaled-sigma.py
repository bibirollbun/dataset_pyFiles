! cp -r /kaggle/input/ariel-pred/ariel-2025 ariel-2025


! export PYTHONPATH="$PYTHONPATH:/kaggle/input/ariel-pred/my-packages" && echo $PYTHONPATH && cd ariel-2025/scripts && python s_values_scaled_sigma.py --input-data-folder /kaggle/input/ariel-data-challenge-2025 --calibrated-data-folder /kaggle/working/calibrated --submission-file /kaggle/working/submission.csv --multiplier 0.945 --mean-sigma 0.00075

