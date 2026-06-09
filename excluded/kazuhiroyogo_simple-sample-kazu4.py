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


train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sample_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")


train_df


test_df


sample_df


# 
!ls ../input/jigsaw-toxic-comment-classification-challenge

# --- 2. spaCyãƒ¢ãƒ‡ãƒ«ã�®ãƒ­ãƒ¼ãƒ‰ ---
nlp = spacy.load("/kaggle/input/en-core-web/en_core_web_lg-3.1.0/en_core_web_lg/en_core_web_lg-3.1.0")

# --- 3. è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�®ãƒ™ã‚¯ãƒˆãƒ«åŒ– (é«˜é€ŸåŒ–) ---
print("Vectorizing train data...")
# ğŸ’¥ã€�ä¿®æ­£ã€‘nlp.pipe() ã‚’ä½¿ã�„ãƒ�ãƒƒãƒ�å‡¦ç�†ã�§é«˜é€ŸåŒ–
vectors_train = [doc.vector for doc in nlp.pipe(train_df['comment_text'], batch_size=50)]
X = np.array(vectors_train)
y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# --- 4. è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¨æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�®åˆ†å‰² ---
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

# --- 5. ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’ ---
print("Training model...")
# ğŸ’¥ã€�ä¿®æ­£ã€‘å¤‰æ•°å��ã‚’ estimator ã�«çµ±ä¸€ã€‚n_jobs=-1ã�§ä¸¦åˆ—å‡¦ç�†
estimator = RandomForestClassifier(random_state=0, n_jobs=-1)
estimator.fit(X_train, y_train)

# --- 6. è¨“ç·´ãƒ»æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�§ã�®äºˆæ¸¬ ---
pred_train = estimator.predict(X_train)
pred_valid = estimator.predict(X_valid)

# --- 7. è©•ä¾¡ ---
# 
print(f"Train MSE: {mean_squared_error(y_train, pred_train)}, Valid MSE: {mean_squared_error(y_valid, pred_valid)}")


# --- 8. ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®ãƒ™ã‚¯ãƒˆãƒ«åŒ– (é«˜é€ŸåŒ–) ---
print("Vectorizing test data...")
# ğŸ’¥ã€�ä¿®æ­£ã€‘X_test ã‚’å®šç¾©
vectors_test = [doc.vector for doc in nlp.pipe(test_df['comment_text'], batch_size=50)]
X_test = np.array(vectors_test)

# --- 9. ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�§ã�®äºˆæ¸¬ ---
print("Predicting on test data...")
# ğŸ’¥ã€�ä¿®æ­£ã€‘estimator ã‚’ä½¿ç”¨
pred_test = estimator.predict(X_test)

# --- 10. æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ� ---
# ğŸ’¥ã€�ä¿®æ­£ã€‘DataFrameã�®ä½œæˆ�æ–¹æ³•ã‚’ä¿®æ­£
submission_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sub_df = pd.DataFrame(pred_test, columns=submission_columns)
sub_df['id'] = test_df['id'].values # 
sub_df = sub_df[['id'] + submission_columns] # 

sub_df.to_csv("Kazu_submission7.csv", index=False)

