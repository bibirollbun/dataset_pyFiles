VER = 37
FOLDS = 5
SEED = 42

import numpy as np, pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
print(f"Train shape: {train.shape}")
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
print(f"Test shape: {test.shape}")
test.head()


BASE = train.columns[1:-1].tolist()
CATS = train[BASE].select_dtypes(include=['object', 'bool']).columns.tolist()
TARGET = train.columns[-1]
print(f"BASE: {BASE}")
print(f"CATS: {CATS}")
print(f"TARGET: {TARGET}")


from sklearn.preprocessing import OrdinalEncoder, StandardScaler

enc = OrdinalEncoder(dtype=np.int8)
train[CATS] = enc.fit_transform(train[CATS])
test[CATS] = enc.transform(test[CATS])

sca = StandardScaler()
train[BASE] = sca.fit_transform(train[BASE])
test[BASE] = sca.transform(test[BASE])


"""
    The code of this cell is extract from 'Featrue Engineering for Machine Learning' by Alice Zheng
and Amanda Casari.
"""

from sklearn.cluster import KMeans

class KMeanFeaturizer:
    def __init__(self, k=100, target_scale=5.0, random_state=42):
        self.k = k
        self.target_scale = target_scale
        self.random_state = random_state

    def fit(self, X, y=None):
        if y is None:
            km_model = KMeans(
                n_clusters=self.k,
                n_init=20,
                random_state=self.random_state
            ).fit(X)

            self.km_model = km_model
            self.cluster_centers_ = km_model.cluster_centers_
            return self

        data_with_target = np.hstack((X, y[:,np.newaxis]*self.target_scale))
        
        km_model_pretrain = KMeans(
            n_clusters=self.k,
            n_init=20,
            random_state=self.random_state
        ).fit(data_with_target)
        
        km_model = KMeans(
            n_clusters=self.k,
            init=km_model_pretrain.cluster_centers_[:,:-1],
            n_init=1,
            max_iter=1
        ).fit(X)
        
        self.km_model = km_model
        self.cluster_centers_ = km_model.cluster_centers_
        return self

    def transform(self, X):
        clusters = self.km_model.predict(X)
        return clusters[:,np.newaxis]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


%%time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import time

oof = np.zeros(len(train))
pred = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for i, (train_idx, val_idx) in enumerate(kf.split(train), 1):
    print(f"Fold {i}")

    X_train = train.loc[train_idx, BASE].copy()
    y_train = train.loc[train_idx, TARGET]
    
    X_val = train.loc[val_idx, BASE].copy()
    y_val = train.loc[val_idx, TARGET]

    X_test = test[BASE].copy()

    start_time = time.time()

    print("KMeans Featurize...")
    
    km = KMeanFeaturizer(target_scale=10).fit(X_train, np.array(y_train))
    X_train_cluster = km.transform(X_train)
    X_val_cluster = km.transform(X_val)
    X_test_cluster = km.transform(X_test)

    print("done")

    sca = StandardScaler()
    X_train_cluster = sca.fit_transform(X_train_cluster)
    X_val_cluster = sca.transform(X_val_cluster)
    X_test_cluster = sca.transform(X_test_cluster)
    
    X_train = np.hstack((X_train, X_train_cluster))
    X_val = np.hstack((X_val, X_val_cluster))
    X_test = np.hstack((X_test, X_test_cluster))

    model = LinearRegression().fit(X_train, y_train)

    oof[val_idx] = model.predict(X_val)
    pred += model.predict(X_test)

    print(f"Fold {i} RMSE: {np.sqrt(mean_squared_error(y_val, oof[val_idx]))}")
    print(f"Fold {i} time: {time.time() - start_time}")

pred /= FOLDS


print(f"Overall RMSE: {np.sqrt(mean_squared_error(train[TARGET], oof))}")


np.save(f"oof_v{VER}", oof)

sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
sub[TARGET] = np.clip(pred, 0, 1)
sub.to_csv(f"pred_v{VER}.csv", index=False)
sub.head()




