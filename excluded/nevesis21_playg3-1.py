import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier,XGBRegressor
import xgboost
from sklearn.model_selection import KFold



df= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
df.head()



df.info()



df.isnull().sum()


df.describe()





# Identify feature columns (excluding 'day' and the last column)
feature_columns = df.columns[2:-1]  # Exclude the first column ('day') and the last column (target)

# Define number of subplots based on feature columns
num_cols = len(feature_columns)
fig, axes = plt.subplots(num_cols, 1, figsize=(8, 4 * num_cols))

# Ensure `axes` is always iterable
if num_cols == 1:
    axes = [axes]  

# Loop through selected feature columns and create subplots
for i, col in enumerate(feature_columns):
    sns.lineplot(x=df['day'], y=df[col], ax=axes[i])
    axes[i].set_title(f'Day vs {col}')
    axes[i].set_xlabel('Day')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()



 pd.option_context('mode.use_inf_as_na', True)


# Fix the dataset before plotting
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Now plot safely
sns.lineplot(x=df['day'], y=df['pressure'])



plt.figure(figsize=(10,8))
sns.countplot(x=df['rainfall'])
plt.xlabel(" 1/0")
plt.ylabel("Counts")
plt.title("count of 1/0")
plt.show()


correlation_matrix=df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',cbar=True)
plt.show()


df['avg_temp'] = (df['maxtemp']+df['mintemp'])/2
df['cloud*humidity']=df['cloud']*df['humidity']
df['mintemp*dewpoint']=df['mintemp'] * df['dewpoint']
df['temperature*dewpoint'] = df['temparature']*df['dewpoint']


df_test['avg_temp'] = (df_test['maxtemp']+df_test['mintemp'])/2
df_test['cloud*humidity']=df_test['cloud']*df_test['humidity']
df_test['mintemp*dewpoint']=df_test['mintemp'] * df_test['dewpoint']
df_test['temperature*dewpoint'] = df_test['temparature']*df_test['dewpoint']


df_test.columns


RMV = ['rainfall','id']
FEATURES=[col for col in df.columns if not col in RMV]
print("out features are:")
print(FEATURES)


import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(df))
pred_xgb = np.zeros(len(df_test))
auc_scores = []  # Move outside the loop to store AUC across folds

for i, (train_index, test_index) in enumerate(kf.split(df)):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)

    x_train = df.loc[train_index,FEATURES].copy()
    y_train = df.loc[train_index,"rainfall"]    
    x_valid = df.loc[test_index,FEATURES].copy()
    y_valid = df.loc[test_index,"rainfall"]
    x_test = df_test[FEATURES].copy()

    model = xgb.XGBClassifier(
        max_depth=8,
        colsample_bytree=0.9, 
        subsample=0.9, 
        n_estimators=10_000,  
        learning_rate=0.1, 
        eval_metric="auc",
        early_stopping_rounds=100,
        alpha=1
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100 
    )
       # INFER OOF
    oof_xgb[test_index] = model.predict_proba(x_valid)[:,1]
    # INFER TEST
    pred_xgb += model.predict_proba(x_test)[:,1]

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS





from sklearn.metrics import roc_auc_score
true = df.rainfall.values
m = roc_auc_score(true, oof_xgb)
print(f"XGBoost CV Score AUC = {m:.3f}")




feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  
plt.show()


import pandas as pd

submission = pd.DataFrame({
    "id": df_test["id"],  # Assuming your test set has an ID column
    "rainfall": pred_xgb  # Predicted probability values
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")





