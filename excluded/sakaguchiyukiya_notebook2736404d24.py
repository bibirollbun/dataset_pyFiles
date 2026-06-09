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


import os
import pandas as pd
import keras_nlp
import keras_core as keras
from sklearn.model_selection import train_test_split

# ãƒ�ãƒƒã‚¯ã‚¨ãƒ³ãƒ‰ã‚’ TensorFlow ã�«è¨­å®š
os.environ["KERAS_BACKEND"] = "tensorflow"

# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_train.head(25)
##ãƒ†ã‚¹ãƒˆç”¨ãƒ•ã‚¡ã‚¤ãƒ«
##ç�½å®³ã�§ã�¯ã�ªã�„ã�®ã�Œï¼�
##ç�½å®³ã�Œï¼‘


df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df_test.head()
#print(len(df))
##æœ¬ç•ªç”¨ãƒ•ã‚¡ã‚¤ãƒ«


df_sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
df_sample.head()
##æ��å‡ºç”¨ãƒ•ã‚¡ã‚¤ãƒ«


# ãƒ†ã‚­ã‚¹ãƒˆã�®è¡¨ç¤ºå¹…ã‚’åºƒã�’ã‚‹ï¼ˆçœ�ç•¥ã�•ã‚Œã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹ï¼‰
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_train.head()
##ãƒ†ã‚¹ãƒˆç”¨ãƒ•ã‚¡ã‚¤ãƒ«

pd.set_option('display.max_colwidth', None)
df_train.head()


df_train.info()##ãƒ‡ãƒ¼ã‚¿ã�®æƒ…å ±
df_train.describe()##ãƒ‡ãƒ¼ã‚¿ã�®çµ±è¨ˆ
df_train.isnull().sum()##ä½•ä»¶æ¬ æ��ã�Œã�‚ã‚‹ã�‹


print(len(df_train))
df_train['target'].value_counts()
##ç�½å®³ã�§ã�¯ã�ªã�„ã�®ã�Œï¼�
##ç�½å®³ã�Œï¼‘


df_train.head(30)


df_train.head(25)


# æ¬ æ��ã‚’åŸ‹ã‚�ã‚‹ï¼ˆç©ºæ–‡å­—ï¼‰
df_train["text"] = df_train["text"].fillna("")
df_test["text"] = df_test["text"].fillna("")

# ãƒ‡ãƒ¼ã‚¿åˆ†å‰²
X_train, X_val, y_train, y_val = train_test_split(df_train["text"], df_train["target"], test_size=0.2, random_state=42)


classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=2,
    activation="softmax"
)

import tensorflow as tf  

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),  
    metrics=["accuracy"]
)


classifier.fit(
    X_train.to_list(), y_train,
    validation_data=(X_val.to_list(), y_val),
    epochs=2,
    batch_size=16
)


# ğŸ“¤ ãƒ†ã‚¹ãƒˆç”¨ã�®äºˆæ¸¬
test_probs = classifier.predict(df_test["text"].to_list(), batch_size=16)
test_preds = test_probs.argmax(axis=1)  # softmaxã�®æœ€å¤§å€¤ â†’ ã‚¯ãƒ©ã‚¹ç•ªå�·

# ğŸ“� æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ�
submission = pd.DataFrame({
    "id": df_test["id"],
    "target": test_preds
})

submission.to_csv("submission2.csv", index=False)




