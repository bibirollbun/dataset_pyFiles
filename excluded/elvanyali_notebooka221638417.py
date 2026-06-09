import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

#train.head()
#test.head()
sample_submission.head()


import lightgbm as lgb
#LightGBM (Light Gradient Boosting Machine), Microsoft tarafından geliştirilen ağaç tabanlı bir makine öğrenimi modelidir.
from sklearn.model_selection import KFold
#Eğitim/validasyon bölmesi için KFold Cross-Validation
#KFold, eğitim verisini birden fazla parçaya bölüp her parçayı bir kez doğrulama (validation) için kullanmamıza olanak tanıyan bir çapraz doğrulama (cross-validation) yöntemidir.
from sklearn.metrics import mean_squared_error
#mean_squared_error (kısaca MSE), makine öğreniminde özellikle regresyon problemlerinde kullanılan bir hata ölçüsüdür.
import numpy as np

features = [col for col in train.columns if col not in ['id', 'rainfall']]
X = train[features]
y = train['rainfall']
X_test = test[features]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

#n_splits=5: veriyi 5 parçaya böl
#shuffle=True: veriyi karıştır (rastgele sıralama overfitting’i azaltır)
#random_state=42: aynı bölmeyi her seferinde elde etmek için sabit rastgelelik

preds = np.zeros(len(test))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val)


    model = lgb.train(
        params={
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt'
        },
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )

    preds += model.predict(X_test) / kf.n_splits


# Tahminleri submission dosyasına yaz
sample_submission['rainfall'] = preds

# Dosyayı kaydet
sample_submission.to_csv('submission.csv', index=False)

# Kontrol amaçlı ilk birkaç satırı tekrar göster
sample_submission.head()


sample_submission.tail()



import pandas as pd

data = {
    'id': [2190, 2191, 2192, 2193, 2194],
    'rainfall': [0.981988, 0.976433, 0.894068, 0.201917, 0.072932]
}

df = pd.DataFrame(data)
print(df)


import matplotlib.pyplot as plt
plt.hist(preds, bins=30)
plt.title('Tahmin Dağılımı')
plt.xlabel('Rainfall')
plt.ylabel('Frequency')
plt.show()

