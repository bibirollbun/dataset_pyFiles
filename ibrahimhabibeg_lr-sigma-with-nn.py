! cp -r /kaggle/input/ariel-pred/ariel-2025 ariel-2025


! export PYTHONPATH="$PYTHONPATH:/kaggle/input/ariel-pred/my-packages" && echo $PYTHONPATH && cd ariel-2025/scripts && python lr_sigma_with_nn.py --input-data-folder /kaggle/input/ariel-data-challenge-2025 --calibrated-data-folder /kaggle/input/calibration-minimal-binning-and-no-channel-cut/calibrated --submission-file /kaggle/working/submission.csv --trained-model-folder /kaggle/working/model --mean-fgs-sigma 0.00085 --mean-airs-sigma 0.00065

