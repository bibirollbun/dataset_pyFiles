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


# Veri setlerini yükleyelim
train = pd.read_csv("/kaggle/input/molecular-machine-learning/train.csv")
test = pd.read_csv("/kaggle/input/molecular-machine-learning/test.csv")

# Özellik seçimi
features_to_drop = ["Batch_ID", "T80", "Smiles"]  # Batch_ID ve Smiles'ı çıkarıyoruz
X_train = train.drop(features_to_drop, axis=1)
y_train = np.log1p(train["T80"])  # T80 üzerinde log dönüşümü yapıyoruz
X_test = test.drop(features_to_drop, axis=1)



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
# 2. VERİ NORMALİZASYONU
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. HİPERPARAMETRE OPTİMİZASYONU
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': [3, 5, 7, 9],
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3)
}

# Modeli oluşturma ve RandomizedSearchCV ile hiperparametre arama
xgb_model = XGBRegressor()
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, verbose=2)
random_search.fit(X_train_scaled, y_train)

# En iyi parametrelerle modeli kuruyoruz
best_xgb_model = random_search.best_estimator_



# 4. ENSEMBLE MODEL (Stacking veya Voting Regressor)
# Farklı modelleri tanımlayalım
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)
ridge_model = Ridge(alpha=1.0)

# Birleştirme modelini oluşturuyoruz
ensemble_model = VotingRegressor(estimators=[('xgb', best_xgb_model), ('rf', rf_model), ('ridge', ridge_model)])

# Modeli eğitelim
ensemble_model.fit(X_train_scaled, y_train)

# 5. TAHMİN YAPMA
test_preds_log = ensemble_model.predict(X_test_scaled)
test_preds = np.expm1(test_preds_log)  # Log dönüşümünü tersine çeviriyoruz

# 6. SONUÇLARI KAYDETME
submission = test[['Batch_ID']].copy()  # Test setinden Batch_ID'yi alıyoruz
submission['T80'] = test_preds  # Tahminleri ekliyoruz

# Submission dosyasını kaydedelim
submission.to_csv("final_submission.csv", index=False)




