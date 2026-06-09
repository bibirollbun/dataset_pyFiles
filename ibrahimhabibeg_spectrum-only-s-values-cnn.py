! cp -r /kaggle/input/ariel-pred/ariel-2025 ariel-2025


! export PYTHONPATH="$PYTHONPATH:/kaggle/input/ariel-pred/my-packages" && echo $PYTHONPATH && cd ariel-2025/scripts && python spectrum_only_s_values_cnn.py --input-data-folder /kaggle/input/ariel-data-challenge-2025 --calibrated-data-folder /kaggle/input/calibration-minimal-binning-and-no-channel-cut/calibrated --submission-file /kaggle/working/submission.csv --trained-model-folder /kaggle/input/spectrum-only-s-values-cnn-training-only/model --device cuda

