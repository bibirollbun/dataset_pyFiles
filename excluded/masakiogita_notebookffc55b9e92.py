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


# 1. å¿…è¦�ãƒ©ã‚¤ãƒ–ãƒ©ãƒªã�®ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
import numpy as np
import pandas as pd
import os
from PIL import Image
from tensorflow.keras.models import load_model
import zipfile

# zipãƒ•ã‚¡ã‚¤ãƒ«ã�®ãƒ‘ã‚¹ï¼ˆKaggleå…¬å¼�ã‚³ãƒ³ãƒšã�®inputå†…ï¼‰
zip_test_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'

# å±•é–‹å…ˆã�®ãƒ‘ã‚¹ï¼ˆ/kaggle/working ã�«å±•é–‹ã�™ã‚‹ï¼‰
extract_test_path = '/kaggle/working'

with zipfile.ZipFile(zip_test_path, 'r') as zip_ref:
    zip_ref.extractall(extract_test_path)

print("âœ… test.zip å±•é–‹å®Œäº†")


# 2. ğŸ”½ ãƒ¢ãƒ‡ãƒ«ã�®ãƒ­ãƒ¼ãƒ‰ â†� ã‚³ã‚³ï¼�ï¼�
model = load_model('/kaggle/input/dvc/tensorflow2/default/1/datascience.h5')  # æ­£ã�—ã�„ãƒ‘ã‚¹ã�«ã�™ã‚‹

# 3. ãƒ†ã‚¹ãƒˆç”»åƒ�ã�®èª­ã�¿è¾¼ã�¿ï¼†å‰�å‡¦ç�†
test_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test'
test_images = []
image_ids = []

for fname in os.listdir(test_dir):
    img = Image.open(os.path.join(test_dir, fname)).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(128, 128, 1)
    test_images.append(img_array)
    image_ids.append(int(fname.split('.')[0]))

test_images = np.array(test_images).astype(np.float32)

# 4. äºˆæ¸¬å®Ÿè¡Œ
preds = model.predict(test_images).flatten()

# 5. æ��å‡ºç”¨CSVä½œæˆ�
submission = pd.DataFrame({'id': image_ids, 'label': preds})
submission = submission.sort_values('id')
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv ã‚’ä¿�å­˜ã�—ã�¾ã�—ã�Ÿ")

