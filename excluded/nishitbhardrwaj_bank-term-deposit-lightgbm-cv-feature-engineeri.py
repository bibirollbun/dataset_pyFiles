import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import category_encoders as ce



import pandas as pd

# Load competition data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# (Optional) Load UCI bank data for exploration/augmentation
bank = pd.read_csv('/kaggle/input/uci-ml-repository-bank-marketing/bank/bank.csv', sep=';')
bank_additional = pd.read_csv('/kaggle/input/uci-ml-repository-bank-marketing/bank-additional/bank-additional/bank-additional-full.csv', sep=';')



print(train.shape, test.shape)
print(train.dtypes)
print(train['y'].value_counts())
train.head()

bank.head()
bank_additional.head()



# Ffilling missing values if have any
train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)

# Identify categorical and numerical columns
cat_cols = train.select_dtypes(include='object').columns.tolist()
num_cols = train.select_dtypes(include='number').columns.drop(['id','y']).tolist()

# encoding labels
from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))



for col in cat_cols:
    freq = train[col].value_counts() / len(train)
    train[col + '_freq'] = train[col].map(freq)
    test[col + '_freq'] = test[col].map(freq)


if 'job' in train.columns and 'y' in bank.columns:
    uci_success_rate = bank.groupby('job')['y'].apply(lambda x: (x == 'yes').mean()).to_dict()
    train['job_uci_srate'] = train['job'].map(uci_success_rate)
    test['job_uci_srate'] = test['job'].map(uci_success_rate)



import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold  # Import KFold or StratifiedKFold

# Define the number of splits (folds) for cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the out-of-fold predictions and test set predictions
oof = np.zeros(len(train))
preds = np.zeros(len(test))

# Define the FEATURES variable (exclude 'y' column)
FEATURES = [col for col in train.columns if col != 'y']

# Start cross-validation
for fold, (trn_idx, val_idx) in enumerate(kf.split(train, train['y'])):
    model = lgb.LGBMClassifier(
        n_estimators=5000, learning_rate=0.01, num_leaves=32, 
        colsample_bytree=0.7, subsample=0.7, random_state=42
    )
    model.fit(
        train.iloc[trn_idx][FEATURES], train.iloc[trn_idx]['y'],
        eval_set=[(train.iloc[val_idx][FEATURES], train.iloc[val_idx]['y'])],
        callbacks=[early_stopping(stopping_rounds=300), log_evaluation(200)]
    )
    # Storing out-of-fold predictions
    oof[val_idx] = model.predict_proba(train.iloc[val_idx][FEATURES])[:, 1]
    
    # Averaging the test set predictions across all folds
    preds += model.predict_proba(test[FEATURES])[:, 1] / kf.n_splits



# Pseudo-label high-certainty predictions
pseudo_test = test.copy()
pseudo_test['y'] = (preds > 0.95).astype(int)
augmented = pd.concat([train, pseudo_test[pseudo_test['y'] == 1], pseudo_test[pseudo_test['y'] == 0]])

# Retrain with the augmented data (re-run model training as above)



submission = sample_submission.copy()
submission['y'] = preds
submission.to_csv('/kaggle/working/submission.csv', index=False)



lgb.plot_importance(model, max_num_features=20)
import matplotlib.pyplot as plt
plt.show()



from sklearn.metrics import roc_auc_score
print('CV ROC AUC:', roc_auc_score(train['y'], oof))



import matplotlib.pyplot as plt
lgb.plot_importance(model, max_num_features=20)  # `model` is from the last fold; you can also use averaged importances.
plt.show()


