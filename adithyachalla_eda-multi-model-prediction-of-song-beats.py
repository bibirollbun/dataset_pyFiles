# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
train


test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test


train.info() 


train.describe().T  


train.isna().sum()  


corr = train.corr(numeric_only=True)
target_corr = corr['BeatsPerMinute'].drop('BeatsPerMinute').sort_values(key=abs, ascending=False)
print(target_corr)


sns.heatmap(corr, cmap='coolwarm', center=0)
plt.show()


from sklearn.feature_selection import mutual_info_regression

X = train.drop(['id','BeatsPerMinute'], axis=1)
y = train['BeatsPerMinute']

mi = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print(mi_series)



sns.histplot(train['BeatsPerMinute'], bins=40, kde=True)
print("Skew:", train['BeatsPerMinute'].skew())


sns.boxplot(x=train['BeatsPerMinute'])


numeric_cols = train.drop(['id','BeatsPerMinute'], axis=1).columns
for col in numeric_cols:
    sns.histplot(train[col], kde=True)
    plt.title(col)
    plt.show()



for col in numeric_cols:
    sns.scatterplot(x=train[col], y=train['BeatsPerMinute'])
    plt.title(f"BPM vs {col}")
    plt.show()



top_feats = corr['BeatsPerMinute'].abs().sort_values(ascending=False).index[1:6]
sns.pairplot(train[top_feats.to_list() + ['BeatsPerMinute']], diag_kind='kde')



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(train[numeric_cols])
pca = PCA(n_components=2)
pc = pca.fit_transform(X_scaled)
plt.scatter(pc[:,0], pc[:,1], c=train['BeatsPerMinute'], cmap='coolwarm')
plt.colorbar(label='BPM')
plt.title("PCA colored by BPM")
plt.show()



train['Energy_per_min'] = train['Energy'] / (train['TrackDurationMs']/60000)
sns.scatterplot(x='Energy_per_min', y='BeatsPerMinute', data=train)



X = train.drop(['id','BeatsPerMinute'], axis=1)
y = train['BeatsPerMinute']



from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)



X_train_scaled


X_val_scaled


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
preds = ridge.predict(X_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print("Ridge RMSE:", rmse)



from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

alphas = np.logspace(-4, 4, 40)
ridge_cv = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
)
ridge_cv.fit(X_train, y_train)

best_alpha = ridge_cv.named_steps['ridgecv'].alpha_
print("Best alpha:", best_alpha)

val_preds = ridge_cv.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("Tuned Ridge RMSE:", rmse)



from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05,
                   max_depth=6, subsample=0.8, colsample_bytree=0.8,
                   random_state=42)
xgb.fit(X_train, y_train)
preds = xgb.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print("XGB RMSE:", rmse)



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

rf = RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
preds = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print("RandomForest RMSE:", rmse)



from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

lgbm = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgbm.fit(X_train, y_train)
preds = lgbm.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print("LightGBM RMSE:", rmse)



# Scale & predict on test
X_full = train.drop(['id','BeatsPerMinute'], axis=1)
scaler.fit(X_full)          # fit on all training data
test_scaled = scaler.transform(test.drop('id', axis=1))

final_preds = xgb.predict(test_scaled)
submission = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': final_preds})
submission.to_csv('submission.csv', index=False)



# Scale & predict on test
X_full = train.drop(['id','BeatsPerMinute'], axis=1)
scaler.fit(X_full)          # fit on all training data
test_scaled = scaler.transform(test.drop('id', axis=1))

final_preds = lgbm.predict(test_scaled)
submission = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': final_preds})
submission.to_csv('submission1.csv', index=False)



# Scale & predict on test
X_full = train.drop(['id','BeatsPerMinute'], axis=1)
scaler.fit(X_full)          # fit on all training data
test_scaled = scaler.transform(test.drop('id', axis=1))

final_preds = ridge.predict(test_scaled)
submission = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': final_preds})
submission.to_csv('submission2.csv', index=False)


from sklearn.ensemble import VotingRegressor
voting_reg = VotingRegressor(
    estimators=[
        ('ridge', ridge),
        ('lgbm', lgbm),
        ('xgb', xgb)
    ],
    weights=[1, 1, 1] 
)

# Fit on training data
voting_reg.fit(X_train, y_train)



# Validate
val_preds = voting_reg.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Voting Regressor RMSE: {rmse:.4f}")


# Predict on test data
test_preds = voting_reg.predict(test_scaled)
submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': test_preds    # or final_preds if that’s your variable name
})
submission.to_csv('submission2.csv', index=False)
print("submission3.csv saved with", len(submission), "rows.")


from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

ridge_base = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

bag_ridge = BaggingRegressor(
    base_estimator=ridge_base,
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
bag_ridge.fit(X_train, y_train)
preds = bag_ridge.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print("Bagging (Ridge) RMSE:", rmse)



from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

bag = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(
        max_depth=None,       # fully grown trees
        random_state=42
    ),
    n_estimators=100,         # number of trees
    max_samples=1.0,          # bootstrap sample size
    max_features=1.0,         # use all features per tree
    bootstrap=True,           # sample with replacement
    n_jobs=-1,                # use all CPU cores
    random_state=42
)
bag.fit(X_train, y_train)
preds = bag.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print("Bagging (Decision Tree) RMSE:", rmse)



# Scale & predict on test
X_full = train.drop(['id','BeatsPerMinute'], axis=1)
scaler.fit(X_full)          # fit on all training data
test_scaled = scaler.transform(test.drop('id', axis=1))
# Predict on test data
test_preds = bag.predict(test_scaled)
submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': test_preds    # or final_preds if that’s your variable name
})
submission.to_csv('submission4.csv', index=False)
print("submission3.csv saved with", len(submission), "rows.")


from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# log-spaced alpha values from 0.001 to 1000
alphas = np.logspace(-4, 4, 40)

# Build a pipeline to scale features and fit RidgeCV
ridge_cv = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
)

# Fit on the full training data (unscaled X)
ridge_cv.fit(X_train, y_train)

# Retrieve best alpha
best_alpha = ridge_cv.named_steps['ridgecv'].alpha_
print("Best alpha selected by RidgeCV:", best_alpha)

# Validation RMSE
from sklearn.metrics import mean_squared_error
val_preds = ridge_cv.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("Validation RMSE:", rmse)



# Fit on entire training set
ridge_cv.fit(X, y)

# Predict on test set
final_preds = ridge_cv.predict(test.drop('id', axis=1))

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': final_preds
})
submission.to_csv('submission_ridgecv.csv', index=False)
print("submission_ridgecv.csv saved with", len(submission), "rows.")



from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Define and fit the MLP
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64),  # you can adjust sizes
    activation='relu',
    solver='adam',
    alpha=1e-4,                    # L2 regularization
    learning_rate='adaptive',
    learning_rate_init=1e-3,
    max_iter=300,
    batch_size=512,
    early_stopping=True,           # stops when validation score stops improving
    n_iter_no_change=20,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

# Validation predictions and RMSE
val_preds = mlp.predict(X_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("MLP RMSE:", rmse)



# Scale & predict on test (for models that require scaling)
X_full = train.drop(['id','BeatsPerMinute'], axis=1)
y_full = train['BeatsPerMinute']

scaler = StandardScaler()
X_full_scaled  = scaler.fit_transform(X_full)
test_scaled    = scaler.transform(test.drop('id', axis=1))


test_preds = mlp.predict(test_scaled)
submission = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': test_preds})
submission.to_csv('submission_mlp.csv', index=False)
print("submission_mlp.csv saved with", len(submission), "rows.")

