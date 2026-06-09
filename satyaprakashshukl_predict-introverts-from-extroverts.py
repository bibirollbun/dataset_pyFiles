import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df_sub=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
df_train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_train.columns


df_train.head()


df_test.head()


df_test.shape,df_train.shape


df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])


df_train.info()


df_train.shape


df_train.isnull().sum()


df_train.describe()


import seaborn as sns


missing_train = df_train.isna().mean() * 100
missing_test = df_test.isna().mean() * 100

print("Columns in df_train with more than 10% missing values:")
print(missing_train[missing_train > 0])

print("\nColumns in df_test with more than 10% missing values:")
print(missing_test[missing_test > 0])


import matplotlib.pyplot as plt


missing_values = df_train.isnull().sum()
missing_values = missing_values[missing_values > 0]

if not missing_values.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Missing Values')
    plt.title('Missing Values per Feature')
    plt.tight_layout()
    plt.show()
else:
    print("✅ No missing values found in the dataset.")


!pip install dython


from dython.nominal import associations

associations_df = associations(df_train[:10000], nominal_columns='all', plot=False)
corr_matrix = associations_df['corr']
plt.figure(figsize=(20, 8))
plt.gcf().set_facecolor('#FFFDD0') 
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix including Categorical Features')
plt.show()


from sklearn.preprocessing import OrdinalEncoder



cat_cols_train = df_train.select_dtypes(include=['object']).columns
cat_cols_train = cat_cols_train[cat_cols_train != 'Personality']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

df_train[cat_cols_train] = ordinal_encoder.fit_transform(df_train[cat_cols_train].astype(str))
df_test[cat_cols_train] = ordinal_encoder.transform(df_test[cat_cols_train].astype(str))


le = LabelEncoder()
df_train['Personality'] = le.fit_transform(df_train['Personality'])


df_train.isnull().sum()


#df_train['avg_social_activity'] = df_train[['Friends_circle_size', 'Social_event_attendance']].mean(axis=1)
#df_test['avg_social_activity'] = df_test[['Friends_circle_size', 'Social_event_attendance']].mean(axis=1)
#df_train['introversion_index'] = df_train['Time_spent_Alone'] * df_train['Drained_after_socializing']
#df_test['introversion_index'] = df_test['Time_spent_Alone'] * df_test['Drained_after_socializing']
drop_cols = ['Stage_fear', 'Going_outside', 'Post_frequency']
df_train = df_train.drop(columns=drop_cols)
df_test = df_test.drop(columns=drop_cols)


y = df_train['Personality'] 
X = df_train.drop(['Personality'],axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42,stratify=y)


Xgb_params ={
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,   
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
test_preds_accum = np.zeros(len(df_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n🔁 Fold {fold+1}")
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**Xgb_params)
    model.fit(
        X_train_fold, y_train_fold,
        early_stopping_rounds=100,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )
    
    val_preds = model.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, val_preds)
    val_scores.append(acc)
    print(f"Validation Accuracy: {acc:.4f}")   
  
    test_preds_accum += model.predict(df_test)


submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')



test_preds_final = np.round(test_preds_accum / 5).astype(int)
submission['Personality'] = le.inverse_transform(test_preds_final)
submission.to_csv('submission.csv', index=False)

print(f"\n✅ Mean CV Accuracy: {np.mean(val_scores):.4f}")


submission['Personality'].hist()

