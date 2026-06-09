# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:10]:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


input_path = '/kaggle/input/cassava-leaf-disease-classification'
train_csv_filename = 'train.csv'

train_csv = pd.read_csv(os.path.join(input_path, train_csv_filename))
train_csv


train_csv['label'].hist()


most_frequent = train_csv['label'].mode().iat[0]
most_frequent


sample_csv_filename = 'sample_submission.csv'

sample_csv = pd.read_csv(os.path.join(input_path, sample_csv_filename))
sample_csv


test_path = 'test_images'

test_image_ids = os.listdir(os.path.join(input_path, test_path))
test_image_ids


predicted_values = np.full(len(test_image_ids), most_frequent)
predicted_values


submission = pd.DataFrame({
    'image_id': test_image_ids,
    'label': predicted_values
})


submission.to_csv('submission.csv', index=False)




