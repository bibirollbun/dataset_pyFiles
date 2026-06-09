# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -q rfdetr==1.2.1 supervision==0.26.1 roboflow


from roboflow import Roboflow
rf = Roboflow(api_key="key")
project = rf.workspace("open-pel5l").project("vision")
version = project.version(2)
dataset = version.download("coco")


from rfdetr import RFDETRSmall

MODEL_PATH = '/content/output/checkpoint_best_total.pth'
model = RFDETRSmall(pretrain_weights=MODEL_PATH)
model.optimize_for_inference()



import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image

TEST_IMAGE_DIR = '/content/dataset/test/images'
SUBMISSION_FILE = 'submission.csv'
NUM_ROWS_REQUIRED = 541

test_images = sorted(list(Path(TEST_IMAGE_DIR).glob('*.jpg')))
if not test_images:
    raise FileNotFoundError(f"No test images found in {TEST_IMAGE_DIR}")

all_preds = []

for img_path in tqdm(test_images):
    img = Image.open(str(img_path))
    det = model.predict(img, threshold=0.001)

    for box, conf in zip(det.xyxy, det.confidence):
        xmin, ymin, xmax, ymax = box
        all_preds.append({
            'image_id': img_path.name,
            'xmin': int(xmin),
            'ymin': int(ymin),
            'xmax': int(xmax),
            'ymax': int(ymax),
            'confidence': conf
        })

df = pd.DataFrame(all_preds).sort_values(by='confidence', ascending=False)
df = df.head(NUM_ROWS_REQUIRED)

if len(df) < NUM_ROWS_REQUIRED:
    pad = NUM_ROWS_REQUIRED - len(df)
    dummy = pd.DataFrame([{
        'image_id': test_images[0].name, 'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1
    }] * pad)
    df = pd.concat([df.drop(columns=['confidence']), dummy], ignore_index=True)
else:
    df = df.drop(columns=['confidence'])

df[['image_id', 'xmin', 'ymin', 'xmax', 'ymax']].to_csv(SUBMISSION_FILE, index=False)
print("file created")


