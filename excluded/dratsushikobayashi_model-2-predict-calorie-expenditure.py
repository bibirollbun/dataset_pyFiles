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


# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# ãƒ‡ãƒ¼ã‚¿ã�®å…ˆé ­ç¢ºèª�
print(train.head())
print(train.info())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ç‰¹å¾´é‡�ã�¨ç›®çš„å¤‰æ•°
features = ['Sex', 'Weight', 'Age', 'Heart_Rate', 'Body_Temp']
X_train = train[features]
y_train = train['Calories']
X_test = test[features]

# ã‚«ãƒ†ã‚´ãƒªåˆ—ã�¨æ•°å€¤åˆ—ã�®åˆ†é›¢
categorical_cols = ['Sex']
numerical_cols = ['Weight', 'Age', 'Heart_Rate', 'Body_Temp']

# å‰�å‡¦ç�†ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # æ•°å€¤åˆ—ã�¯ã��ã�®ã�¾ã�¾é€šã�™
)

# ç·šå½¢å›�å¸°ãƒ¢ãƒ‡ãƒ«ã‚’å�«ã‚€ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# ï¼ˆç„¡ç�†ã�ªã‚¯ãƒªãƒƒãƒ—ã‚’é�¿ã�‘ã‚‹ã�Ÿã‚�ã�«ï¼‰å¯¾æ•°å¤‰æ�›ã�—ã�¦å­¦ç¿’
y_train_log = np.log1p(y_train)  # log(1 + y)
model.fit(X_train, y_train_log)

# äºˆæ¸¬ã�—ã€�é€†å¤‰æ�›ã�—ã�¦æ��å‡ºå½¢å¼�ã�«
test_preds_log = model.predict(X_test)
test_preds = np.expm1(test_preds_log)  # exp(pred) - 1

# å¿µã�®ã�Ÿã‚�äºˆæ¸¬å€¤ã‚’å®‰å…¨ã�«ã‚¯ãƒªãƒƒãƒ—ï¼ˆã‚ªãƒ—ã‚·ãƒ§ãƒ³ï¼‰
test_preds = np.clip(test_preds, 1, 314)

# ğŸ“¤ æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ä½œæˆ�
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_preds
})
submission.to_csv("/kaggle/working/submission_log_transform.csv", index=False)




