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


train_path = '/kaggle/input/recommendation-class-homework-2024-part-time/Tianchi_Train_uid_iid_rt.csv'
test_path  = '/kaggle/input/recommendation-class-homework-2024-part-time/Tianchi_Test_uid_iid.csv'

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

# Đảm bảo cột đúng tên
print(df_train.head())
print(df_test.head())



import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# ---- Đọc dữ liệu ----
min_rating, max_rating = df_train['ratings'].min(), df_train['ratings'].max()
reader = Reader(rating_scale=(min_rating, max_rating))

data = Dataset.load_from_df(
    df_train[['user_id', 'click_article_id', 'ratings']],
    reader
)

# ---- Huấn luyện SVD ----
algo = SVD(n_factors=500, reg_all=0.02, lr_all=0.005)
cross_validate(algo, data, measures=['RMSE','MAE'], cv=5, verbose=True)
trainset = data.build_full_trainset()
algo.fit(trainset)

# ---- Dự đoán ----
raw_preds = [
    algo.predict(uid, iid).est
    for uid, iid in zip(df_test['user_id'], df_test['click_article_id'])
]

# ---- Chuẩn hóa & chuyển nhãn 0/1 ----
preds_scaled = (np.array(raw_preds) - min_rating) / (max_rating - min_rating)
preds_scaled = np.clip(preds_scaled, 0.0, 1.0)
binary_preds = (preds_scaled >= 0.5).astype(int)

# ---- Tạo file nộp với 2 cột ID, ratings ----
# Nếu file test gốc đã có cột 'ID' thì dùng trực tiếp.
# Nếu chưa có, tạo ID = 0..N-1 để khớp sample_submission.
if 'ID' not in df_test.columns:
    df_test = df_test.copy()
    df_test['ID'] = range(len(df_test))

df_submit = pd.DataFrame({
    'ID': df_test['ID'],
    'ratings': binary_preds
})

assert len(df_submit) == 75652, f"Số dòng không khớp: {len(df_submit)}"

df_submit.to_csv('submission.csv', index=False)
print("✅ Saved submission.csv with columns:", df_submit.columns.tolist())


