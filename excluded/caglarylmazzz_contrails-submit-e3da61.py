! pip install -q --no-index --find-links=file:///kaggle/input/contrails-weights/pip \
    segmentation_models_pytorch
! cp -r /kaggle/input/contrails-src ./src


import numpy as np
import sys
import yaml
sys.path.append('/kaggle/working/src')
import unet5
import vit4
import unet1024
import util
from submit import write_submission


cfg = yaml.safe_load("""
input:
  weight: /kaggle/input/contrails-weights

unet5:
  batch_size: 1
  models:
    - name: maxvit_tiny
      encoder: maxvit_tiny_tf_512.in1k
      decoder_channels: [256, 128, 64, 32, 16]
      w: 1.0
      resize: 512
      tta: d4prob
      folds: [0,1,2,3,4]

vit4:
  batch_size: 2
  models:
    - name: vit4_1024
      encoder: maxvit_tiny_tf_512.in1k
      decoder_channels: [256, 128, 64, 32, 32]
      w: 0.6465366913319027
      #w: 1.0
      resize: 512
      tta: d4prob
      folds: [5, 7]

unet1024:
  batch_size: 2
  models:
    - name: unet1024
      encoder: maxvit_tiny_tf_512.in1k
      decoder_channels: [256, 128, 64, 32, 16]
      w: 0.4531613974630522
      resize: 1024
      tta: d4prob
      folds: [3, 4]
""")


data_type = 'test'
#data_type = 'validation'
debug=False

# Unet3 (Ash-color 3-channel image)
#preds = unet3.run(data_type, cfg, None, debug=debug)
#preds = unet4.run(data_type, cfg, None, debug=debug)
#preds = unet5.run(data_type, cfg, None, debug=debug)
#preds = unet1024.run(data_type, cfg, None, debug=debug)

# Ensemble
preds = vit4.run(data_type, cfg, None, debug=debug)
unet1024.run(data_type, cfg, preds, debug=debug)


util.check_preds_finite(preds)

# Write submission.csv
#th = 0.45
th = 0.5
submit = write_submission(preds, th, 'submission.csv')


! tail submission.csv

if data_type == 'validation':
    ! python3 src/score.py submission.csv




