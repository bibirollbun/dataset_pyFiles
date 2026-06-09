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


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


sample_submission.head()


train.head()


train.describe()


test.describe()


train.info()


test.info()


train.isnull().sum()


test.isnull().sum()





for col in train.select_dtypes(include=['object', 'bool']).columns:
    print(f"--- {col} ---")
    print(train[col].value_counts())
    print("\n")



for col in test.select_dtypes(include=['object', 'bool']).columns:
    print(f"--- {col} ---")
    print(test[col].value_counts())
    print("\n")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

cols = train.select_dtypes(include=['object', 'bool']).columns

sns.set(style="whitegrid")

n_cols = 4  # satır başına kaç grafik
n_rows = math.ceil(len(cols) / n_cols)

plt.figure(figsize=(5 * n_cols, 4 * n_rows))

for i, col in enumerate(cols, 1):
    plt.subplot(n_rows, n_cols, i)
    order = train.groupby(col)['accident_risk'].mean().sort_values().index
    sns.barplot(
        x=col,
        y='accident_risk',
        data=train,
        order=order,
        palette='viridis'
    )
    plt.title(f"{col} vs Ortalama Kaza Riski")
    plt.ylabel("Ortalama Risk")
    plt.xlabel(col)
    plt.xticks(rotation=30)

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Sayısal sütunları seç
cols = train.select_dtypes(include=['number']).columns.drop('accident_risk', errors='ignore')

sns.set(style="whitegrid")

n_cols = 4  # satır başına kaç grafik
n_rows = math.ceil(len(cols) / n_cols)

plt.figure(figsize=(5 * n_cols, 4 * n_rows))

for i, col in enumerate(cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.regplot(
        x=col,
        y='accident_risk',
        data=train,
        scatter_kws={'alpha': 0.4, 's': 20},
        line_kws={'color': 'red'},
        ci=None
    )
    plt.title(f"{col} vs Kaza Riski")
    plt.xlabel(col)
    plt.ylabel("Kaza Riski")

plt.tight_layout()
plt.show()



train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


train.head()


test.head()





from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


cat_features = train.select_dtypes(include=['object', 'bool']).columns.tolist() 
feature_cols = [ 'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday',
                'school_season', 'num_reported_accidents']
X = train[feature_cols]
y = train['accident_risk']


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=42
)



from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

best_rmse = float('inf')
best_depth = None
models = {}  # modelleri depth ile birlikte saklamak için

for depth in [4, 6, 8, 10]:
    model = CatBoostRegressor(
        iterations=3000,
        depth=depth,
        learning_rate=0.1,       # başlangıç lr
        loss_function='RMSE',
        eval_metric='RMSE',
        cat_features=cat_features,
        random_seed=42,
        verbose=50,
        use_best_model=True,
        early_stopping_rounds=50
    )
    
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(f"Depth={depth}, RMSE={rmse:.4f}")
    
    # modeli sakla
    models[depth] = model
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_depth = depth

print(f"Best depth: {best_depth}, Best RMSE: {best_rmse:.4f}")

# En iyi model
best_model = models[best_depth]

# Artık best_model ile test tahmini ve submission yapılabilir



test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")  # orijinal test verisi
feature_cols = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
                'weather', 'road_signs_present', 'public_road', 'time_of_day', 
                'holiday', 'school_season', 'num_reported_accidents']

X_test = test[feature_cols]

# 3️⃣ Tahmin yap
y_test_pred = best_model.predict(X_test)








# 4️⃣ Submission DataFrame oluştur
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': y_test_pred
})


# 5️⃣ Tahminleri 3 basamaklı yuvarla (isteğe bağlı)
submission['accident_risk'] = submission['accident_risk'].round(3)

# 6️⃣ CSV olarak kaydet
submission.to_csv('submission.csv', index=False)

print(submission.head())

