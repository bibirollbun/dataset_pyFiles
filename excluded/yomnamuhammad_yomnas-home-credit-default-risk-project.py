import pandas as pd
import numpy as np


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """
    Down‑cast float64→float32 and int64→int32/16 to cut RAM ~50 %.
    Does NOT affect object / category columns.
    """
    for col in df.columns:
        t = df[col].dtype
        if t.kind in "iuf":
            df[col] = pd.to_numeric(
                df[col],
                downcast="float" if t.kind == "f" else "integer"
            )
    return df


data = '/kaggle/input/home-credit-default-risk'


train_data = reduce_mem(pd.read_csv(f"{data}/application_train.csv"))
test_data = reduce_mem(pd.read_csv(f"{data}/application_test.csv"))


bureau_data = reduce_mem(pd.read_csv(f"{data}/bureau.csv"))
previous_app_data = reduce_mem(pd.read_csv(f"{data}/previous_application.csv"))


print(train_data.info())


print(bureau_data.info())


print(previous_app_data.info())


bureau_data_num = bureau_data.drop(["CREDIT_ACTIVE","CREDIT_CURRENCY","CREDIT_TYPE"],axis=1)


import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation matrix
corr_matrix = bureau_data_num.corr()

# Set up the figure size
plt.figure(figsize=(12, 10))

# Create a heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)

# Add a title
plt.title("Correlation Matrix of Bureau Data", fontsize=16)

# Show the plot
plt.show()



cat_cols_prev_app = previous_app_data.select_dtypes(include= ['object', 'category'])
wo_cat_cols_prev_app = previous_app_data.drop(cat_cols_prev_app,axis=1)


import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation matrix
corr_matrix = wo_cat_cols_prev_app.corr()

# Set up the figure size
plt.figure(figsize=(12, 10))

# Create a heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)

# Add a title
plt.title("Correlation Matrix of Previous Application Data", fontsize=16)

# Show the plot
plt.show()


print(len(test_data))  #Checking Test Size: should be 48744


def bur_features(bureau_data):
    bureau_features = bureau_data.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'CREDIT_ACTIVE': lambda x: (x == 'Active').sum(),
    }).rename(columns={
        'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT', #total number of loans (ACTIVE & NOT ACTIVE) the client has done from previous banks
        'CREDIT_ACTIVE': 'BUREAU_ACTIVE_LOANS' #total number of ACTIVE loans the client has done from previous banks
    }).reset_index()

    bureau_debt_features = bureau_data.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT_SUM': ['mean', 'max', 'sum'], #Average:Average size of issued credits, MAX:Largest loan issued, total credit across all accounts.
        'AMT_CREDIT_SUM_DEBT': ['mean', 'max', 'sum'] #mean:Average remaining debt, max:Largest outstanding debt, sum:Total debt
    })
    bureau_debt_features.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_debt_features.columns]
    bureau_debt_features.reset_index(inplace=True)

    bureau_final = pd.merge(bureau_features, bureau_debt_features, on='SK_ID_CURR', how='left')
    return bureau_final  # Only return aggregated features



def prev_features(previous_app_data):
    prev_features = previous_app_data.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'NAME_CONTRACT_STATUS': lambda x: (x == 'Approved').sum(),
    }).rename(columns={
        'SK_ID_PREV': 'PREV_APPLICATION_COUNT', #count of all previous applications with
        'NAME_CONTRACT_STATUS': 'PREV_APPROVED_COUNT' #count of all previously APPROVED applications with
    }).reset_index()

    return prev_features  # Only return aggregated features



# Generate and merge clean features using the two functions above
bureau_final = bur_features(bureau_data)
previous_final = prev_features(previous_app_data)

# Merge into train and test
train_data = pd.merge(train_data, bureau_final, on='SK_ID_CURR', how='left')
train_data = pd.merge(train_data, previous_final, on='SK_ID_CURR', how='left')

test_data = pd.merge(test_data, bureau_final, on='SK_ID_CURR', how='left')
test_data = pd.merge(test_data, previous_final, on='SK_ID_CURR', how='left')

# Fill missing values
train_data.fillna(-999, inplace=True)
test_data.fillna(-999, inplace=True)



print(len(test_data))  #Test set size chekpoint; should be 48744


train_data


def drop_percentage_null_rows(df, threshold=50):

  """
    Drops columns from a DataFrame if their percentage of null values is above a given threshold.

    Parameters:
    - df (pd.DataFrame): input DataFrame.
    - threshold (float): The percentage (0-100) above which columns are dropped.

    Returns:
    - df_cleaned (pd.DataFrame): A new DataFrame with the high-null columns dropped.
    - dropped_columns (list): A list of the column names that were dropped.
    """
  # Calculate percentage of nulls
  null_percent = df.isnull().mean() * 100

  # Find columns to drop
  to_drop = null_percent[null_percent > threshold].index.tolist()

  # Drop them
  df_cleaned = df.drop(columns=to_drop)

  print(f"Dropped {len(to_drop)} columns with more than {threshold}% nulls.")
  return df_cleaned, to_drop



train_data, dropped_cols_train = drop_percentage_null_rows(train_data, threshold=50)
test_data = test_data.drop(columns=dropped_cols_train)


print(train_data.info())
print(test_data.info()) #making sure that both the test set and train set have the same number of columns


print(len(test_data)) # Test size check point should be 48744


train_data.isnull().sum() #check missing values in each column


train_data = train_data.dropna() #drop all rows containing missing values


train_data.isnull().sum() #check once again missing values in each column


train_data.duplicated().sum()


print(len(test_data))                # should be 48744


TARGET = "TARGET"
cat_cols=[c for c in train_data.columns if train_data[c].dtype=="object"]
num_cols=[c for c in train_data.columns if c not in cat_cols + ["SK_ID_CURR",TARGET]]


print(len(cat_cols))
print(len(num_cols))


# concat so we get identical dummy columns for train and test
full_cat = pd.concat([train_data[cat_cols], test_data[cat_cols]], axis=0)

dummies = pd.get_dummies(full_cat, dummy_na=False)

# split again
n_train = len(train_data)
train_dummies = dummies.iloc[:n_train].reset_index(drop=True)
test_dummies  = dummies.iloc[n_train:].reset_index(drop=True)


# drop raw categorical cols and add dummies
train_data = pd.concat([train_data.drop(columns=cat_cols).reset_index(drop=True),
                       train_dummies], axis=1)
test_data  = pd.concat([test_data.drop(columns=cat_cols).reset_index(drop=True),
                       test_dummies],  axis=1)


print(len(test_data))  #test size check: should be 48744


variances = train_data.var().sort_values(ascending=False) #finding the variances of all columns


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.histplot(variances, bins=50, kde=True)
plt.title("Distribution of Feature Variance")
plt.xlabel("Variance")
plt.ylabel("Number of Features")
plt.grid(True)
plt.show()



X_train = train_data.drop(columns=[TARGET, "SK_ID_CURR"])
y_train = train_data[TARGET]

sk_ids = test_data['SK_ID_CURR'].copy()
X_test  = test_data.drop(columns=["SK_ID_CURR"])


print(len(X_test))  #Test size checkpoint should be 48744


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
!pip install -q optuna
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score



"""def objective(trial): #Optuna calls repeatedly to test different parameter combinations
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500), #number of trees
        'max_depth': trial.suggest_int('max_depth', 3, 10), # Maximum depth of each tree
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), # tells Optuna to test learning rates from 0.01 to 0.3.
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), #prevents overfitting
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), #Prevents overfitting by using different features per tree.
        'gamma': trial.suggest_float('gamma', 0, 5), #Minimum loss reduction required to make a split
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), #Minimum total weight (sum of sample weights) needed to make a split
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0), #L1 regularization on weights
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0), #L2 regularization on weights
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'auc'
    }

    model = XGBClassifier(**params) #defining the model

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Keep it 3 for speed
    score = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1) #evaluate using cross-validation and return the AUC score.

    return score.mean()
"""



#study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42)) #study is the optimization process zy Job kda aw experiment
#study.optimize(objective, n_trials=20)  # Runs objective 20 times and keep track of the best scores and parameters


# Best parameters found from Optuna tuning
best_params = {
    'n_estimators': 425,
    'max_depth': 8,
    'learning_rate': 0.04323444823345436,
    'subsample': 0.7183238342717091,
    'colsample_bytree': 0.7476569144728966,
    'gamma': 2.793707349238791,
    'min_child_weight': 5,
    'reg_alpha': 3.687326698058209,
    'reg_lambda': 3.3561848367527585,
    'use_label_encoder': False,
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': 42
}

#Initialze and train the model
model = XGBClassifier(**best_params)
model.fit(X_train, y_train)



test_pred = model.predict_proba(X_test)[:, 1] #returns probabilities for class 0 and class 1
#[:, 1] gives only the probability of the positive class (defaulting on a loan) kaggle wants it

submission = pd.DataFrame({
    'SK_ID_CURR': sk_ids,
    'TARGET': test_pred
})
submission.to_csv("submission.csv", index=False)



submission.shape  # should be 48744




