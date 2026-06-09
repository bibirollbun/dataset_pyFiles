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


train = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')
test = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv')


# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer
import torch
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def get_embeddings(df):
    combined_text = df['anchor'] + ' [SEP] ' + df['target'] + ' [SEP] ' + df['context']
    embeddings = model.encode(combined_text.tolist(), batch_size=64, show_progress_bar=True)
    return embeddings


X = get_embeddings(train)
y = train['score']

# 4. Model: Ridge Regression (basit başlıyoruz)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = Ridge()
regressor.fit(X_train, y_train)


# Validation skoruna bakalım
y_pred = regressor.predict(X_val)
print('Validation MSE:', mean_squared_error(y_val, y_pred))



# 5. Test setine uygulayalım
X_test = get_embeddings(test)
test_preds = regressor.predict(X_test)

# 6. Submission dosyası oluştur
submission = pd.DataFrame({
    'id': test['id'],
    'score': test_preds
})

submission.to_csv('submission.csv', index=False)




























