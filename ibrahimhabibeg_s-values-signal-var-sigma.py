! cp -r /kaggle/input/ariel-pred/ariel-2025 ariel-2025


! export PYTHONPATH="$PYTHONPATH:/kaggle/input/ariel-pred/my-packages" && echo $PYTHONPATH && cd ariel-2025/scripts && python s_values_signal_var_sigma.py --input-data-folder /kaggle/input/ariel-data-challenge-2025 --calibrated-data-folder /kaggle/working/calibrated --submission-file /kaggle/working/submission.csv --multiplier 0.945 --mean-fgs-sigma 0.00090 --mean-airs-sigma 0.00070

