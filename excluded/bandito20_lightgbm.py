import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df.drop('id', axis=1, inplace=True)
df.head()


num_cols = df.select_dtypes(include=['number']).columns.difference(['y'])
cat_cols = df.select_dtypes(include=['object']).columns


num_cols


import warnings
warnings.filterwarnings('ignore')


fig, ax = plt.subplots(4, 2, figsize=(10, 10))
ax = ax.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, ax=ax[i])
    ax[i].set_title(col)
plt.show()


cat_cols


for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col)
    plt.show()


for col in cat_cols:
    df[col] = df[col].astype('category')
df[cat_cols] = df[cat_cols].replace('unknown', np.nan)


df[cat_cols].isna().sum()


df['y'].value_counts()


plt.pie(df['y'].value_counts().values, labels=df['y'].value_counts().index, autopct='%1.1f%%')


X = df.drop('y', axis=1)
y = df['y']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


train_data = lgb.Dataset(X_train, y_train)

params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'scale_pos_weight': 7.28839
}

model = lgb.train(params, train_data, num_boost_round=100)


y_pred = model.predict(X_test)


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc})')
plt.plot([0, 1], [0, 1], 'k--')  #
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


lgb.plot_importance(model, max_num_features=20)


from bayes_opt import BayesianOptimization

def lgb_optimizer(num_leaves, learning_rate, max_depth, min_data_in_leaf, feature_fraction, lambda_l1, lambda_l2):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': int(num_leaves),
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'min_data_in_leaf': int(min_data_in_leaf),
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'verbose': -1
    }
    
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=100,
        nfold=5,
        stratified=True,
        metrics='auc',
        seed=42
    )
    
    # Check the actual key in cv_results (try printing cv_results.keys())
    # Common keys: 'auc-mean', 'valid auc-mean', 'binary_logloss-mean', etc.
    auc_key = [key for key in cv_results.keys() if 'auc' in key and 'mean' in key][0]
    return max(cv_results[auc_key])  # Return the best AUC score

# Define parameter bounds
pbounds = {
    'num_leaves': (15, 127),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 12),
    'min_data_in_leaf': (20, 100),
    'feature_fraction': (0.7, 1.0),
    'lambda_l1': (0, 100),
    'lambda_l2': (0, 100)
}

# Run Bayesian optimization
optimizer = BayesianOptimization(f=lgb_optimizer, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=10, n_iter=20)  
print("Best parameters:", optimizer.max['params'])


best_params = {'num_leaves': 57, 
              'learning_rate': 0.28570714885887566, 
              'max_depth': 10, 
              'min_data_in_leaf': 68, 
              'feature_fraction': 0.7468055921327309, 
              'lambda_l1': 15.599452033620265, 
              'lambda_l2': 5.8083612168199465}

model = lgb.train(
    best_params,
    train_data,
    num_boost_round=100,
)


y_pred = model.predict(X_test)


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc})')
plt.plot([0, 1], [0, 1], 'k--')  #
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_df.head()


X_test = test_df.drop('id', axis=1)
cat_cols = X_test.select_dtypes(include=['object']).columns
for col in cat_cols:
    X_test[col] = X_test[col].astype('category')
X_test[cat_cols] = X_test[cat_cols].replace('unknown', np.nan)
y_pred = model.predict(X_test)


submission = pd.DataFrame({"id": test_df["id"], "y": y_pred})
submission.to_csv("submission_bank.csv", index=False)

