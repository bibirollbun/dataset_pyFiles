import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


df_train.shape


df_train.head()


df_train.columns


df_train.info()


df_train.describe().T


train = df_train.copy()
test = df_test.copy()


train.groupby('y')['duration'].describe()


import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x='y', y='duration', data=train)
plt.title("Checking if 'duration' leaks the target")
plt.show()


def preprocess(df):
    df = df.copy()
    
    df['pdays_was_contacted'] = (df['pdays'] != -1).astype(int)
    
    df['pdays_bin'] = pd.cut(df['pdays'], bins=[-2, 0, 100, 300, 900], labels=[0,1,2,3])
    
    df['balance_log'] = np.log1p(df['balance'])
    
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 
                   'aug', 'sep', 'oct', 'nov', 'dec']
    df['month'] = df['month'].str.lower()
    df['month'] = df['month'].map({m: i for i, m in enumerate(month_order)})
    
    return df


train = preprocess(train)
test = preprocess(test)


# Drop IDs (not predictive)
train.drop('id', axis=1, inplace=True)
test_ids = test['id']
test.drop('id', axis=1, inplace=True)


# Encode categorical columns
cat_cols = train.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = test[col].map(lambda s: '<UNK>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<UNK>')
    test[col] = le.transform(test[col])


X = train.drop('y', axis=1)
y = train['y']
X_test = test.copy()


# Model training using Stratified KFold + LightGBM
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nTraining fold {fold + 1}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=1000,
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / skf.n_splits


auc_score = roc_auc_score(y, oof_preds)
print(f"\nâœ… Overall ROC AUC Score: {auc_score:.5f}")


sample_submission


submission = pd.DataFrame({
    'id': sample_submission['id'],
    'y': test_preds
})
submission.to_csv('submission.csv', index=False)


submission




