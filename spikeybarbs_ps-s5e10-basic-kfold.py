from warnings import filterwarnings
filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

print('library loaded!')


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col = 'id')
X_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col = 'id')

print('train shape =', train.shape)
train.head()


train.info()


train.duplicated().sum()


train.drop_duplicates(inplace = True)


target = 'accident_risk'
X = train.drop(columns = [target])
y = train.accident_risk

categoric_columns = X.select_dtypes(include = 'object').columns.to_list()
numeric_columns = X.select_dtypes(exclude = 'object').columns.to_list()


fig, axes = plt.subplots(1, int(len(categoric_columns)), figsize=(5*len(categoric_columns), 4))
for ax, col in zip(axes, categoric_columns):
    sns.countplot(data=X, x=col, ax=ax)
    ax.set_title(f'Count of {col}')
plt.tight_layout()
plt.show()


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size = 0.8, test_size = 0.2, random_state = 42
)

categoric_transformer = Pipeline(steps = [
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

numeric_transformer = Pipeline(steps = [
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers = [
    ('cat', categoric_transformer, categoric_columns),
    ('num', numeric_transformer, numeric_columns)
])

lgbm_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.08,
    max_depth=-1,          
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

reg = Pipeline(steps = [
    ('preprocessing', preprocessor),
    ('model', lgbm_model)
])

reg.fit(X_train, y_train)
preds = reg.predict(X_valid)

print('RMSE =', mean_squared_error(preds, y_valid, squared = False))


print('Predicting data X_test')

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_test),))

for train_idx, val_idx in kfold.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    reg.fit(X_tr, y_tr)
    test_preds += reg.predict(X_test) / kfold.get_n_splits()
    
    preds = reg.predict(X_valid)
    print('RMSE =', np.sqrt(mean_squared_error(preds, y_valid)))
    
output = pd.DataFrame({
    'id': X_test.index,
    'accident_risk': test_preds
})

output.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")

