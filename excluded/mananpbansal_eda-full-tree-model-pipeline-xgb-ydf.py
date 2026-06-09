!pip install ydf


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from collections import Counter
import lightgbm as lgb
import xgboost as xgb


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.info()


train.head()


target_col = "Personality"
num_col = train.select_dtypes(include=['float64', 'int64']).columns
cat_col = train.select_dtypes(include=['object']).columns


print(f"Target Column: {target_col}\nNumeric Column: {num_col}\nCat Column: {cat_col}")


for col in train.columns:
    if train[col].isnull().sum() != 0:
        null_pct = (train[col].isnull().sum() / train.shape[0]) * 100
        print(f"{col} has {null_pct:.2f}% null values")
        if pd.api.types.is_numeric_dtype(train[col]):
            skew_val = train[col].skew()
            print(f"Skewness of {col}: {skew_val:.2f}")
        else:
            print(f"{col} is non-numeric — skewness not applicable.")


'''
train['Time_spent_Alone'] = train['Time_spent_Alone'].fillna(train['Time_spent_Alone'].median())

for col in ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
    train[col] = train[col].fillna(train[col].mean())
'''


'''
for col in ['Stage_fear', 'Drained_after_socializing']:
    train[col] = train[col].fillna(train[col].mode()[0])
'''


sns.countplot(x='Personality', data=train)
plt.title("Distribution of Personality Types")
plt.ylabel("Count")
plt.show()

print(train['Personality'].value_counts(normalize=True))


numeric_cols = train.select_dtypes(include=['float64', 'int64']).drop(columns=['id'])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.show()


sns.boxplot(x='Personality', y='Time_spent_Alone', data=train)
plt.title("Time Spent Alone vs Personality Type")
plt.show()

sns.violinplot(x='Personality', y='Post_frequency', data=train)
plt.title("Post Frequency by Personality Type")
plt.show()


sns.countplot(x='Stage_fear', hue='Personality', data=train)
plt.title("Stage Fear Distribution by Personality Type")
plt.xticks(rotation=45)
plt.show()

sns.countplot(x='Drained_after_socializing', hue='Personality', data=train)
plt.title("Drained After Socializing by Personality Type")
plt.xticks(rotation=45)
plt.show()


sns.boxplot(data=train, x='Personality', y='Friends_circle_size')
plt.title("Friends Circle Size Outliers")
plt.show()


col = "Time_spent_Alone"
Q1 = train[col].quantile(0.25)
Q3 = train[col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
outlier_percent = (len(outliers) / train.shape[0]) * 100

print(f"{col}: {len(outliers)} outliers ({outlier_percent:.2f}%)")


train['Time_spent_Alone'].value_counts().sort_index()


train["Time_spent_Alone"] = np.log1p(train["Time_spent_Alone"])


Q1 = train['Time_spent_Alone'].quantile(0.25)
Q3 = train['Time_spent_Alone'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mean_value = train['Time_spent_Alone'].mean()

train['Time_spent_Alone'] = train['Time_spent_Alone'].apply(
    lambda x: mean_value if x < lower_bound or x > upper_bound else x
)

mean_test_value = test['Time_spent_Alone'].mean()
test['Time_spent_Alone'] = test['Time_spent_Alone'].apply(
    lambda x: mean_test_value if x < lower_bound or x > upper_bound else x
)


'''le1 = LabelEncoder()
le2 = LabelEncoder()
train['Stage_fear'] = le1.fit_transform(train['Stage_fear'])
train['Drained_after_socializing'] = le2.fit_transform(train['Drained_after_socializing'])'''


train['Personality'] = train['Personality'].map({'Introvert': 1, 'Extrovert': 0})


#train = train.drop('Time_spent_Alone_log', axis=1)


#train = train.drop('id', axis=1)


y = train['Personality']
X = train.drop('Personality', axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


counter = Counter(y_train)
neg, pos = counter[0], counter[1]

scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")


xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    tree_method="hist",
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=8, 
    subsample=0.7,
    colsample_bytree=1,
    max_bin=1024,
    scale_pos_weight=2.84,
    eval_metric='logloss',  
    early_stopping_rounds=100,
    n_jobs=-1
)


xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)


lgbParams = {'n_estimators': 5000,
             'max_depth': 16, 
             'learning_rate': 0.0005,
             'min_child_weight': 3.43,
             'min_child_samples': 216, 
             'subsample': 0.782,
             'subsample_freq': 4, 
             'colsample_bytree': 1, 
             'num_leaves': 25,
             'scale_pos_weight': 2.84,
             'force_col_wise':'true',
             'verbose':-1,
             'early_stopping_rounds':300,
             #'is_unbalance': True
            }


lgb_model=lgb.LGBMClassifier(**lgbParams)
lgb_model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)])
lgb.plot_importance(lgb_model, importance_type="gain", figsize=(8,6), max_num_features=12, color = "black",
                    title="LightGBM Feature Importance (Gain)")
plt.show()


import ydf

X_train['Personality'] = y_train
new_train = X_train.copy()

model_gbt = ydf.GradientBoostedTreesLearner(
    label="Personality",
    task=ydf.Task.CLASSIFICATION,
    include_all_columns=True,
    max_depth=20,
    min_examples=10,
    num_trees=500,
    shrinkage=0.05,
    subsample=0.8,
    categorical_algorithm="CART",
    allow_na_conditions=True,
    missing_value_policy="GLOBAL_IMPUTATION",
    validation_ratio=0.1,
    early_stopping="LOSS_INCREASE",
    early_stopping_initial_iteration=30,
    early_stopping_num_trees_look_ahead=50,
    random_seed=42,
).train(new_train)


model_rfl = ydf.CartLearner(
    label="Personality",
    task=ydf.Task.CLASSIFICATION,
    
    max_depth=20,                          
    min_examples=10,                     
    max_num_nodes=None,                 
    
    include_all_columns=True,
    allow_na_conditions=True,             
    missing_value_policy="GLOBAL_IMPUTATION",  
    
    categorical_algorithm="CART",        
    max_vocab_count=2000,
    min_vocab_frequency=5,
).train(new_train)


models = {
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'GradientBoostedTrees': model_gbt,
    'RandomForest': model_rfl
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_val)
    y_pred = (y_pred[:] > threshold).astype(int)

    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'], ax=axes[i])
    axes[i].set_title(f'{name} (Acc: {acc:.4f})')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

plt.tight_layout()
plt.show()


#test = test.drop('id', axis=1)
test.info()


#num_cols = ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

#test['Time_spent_Alone'] = test['Time_spent_Alone'].fillna(test['Time_spent_Alone'].median())

#for col in num_cols:
 #   test[col] = test[col].fillna(test[col].mean()) 

#test['Stage_fear'] = test['Stage_fear'].fillna(test['Stage_fear'].mode()[0])
#test['Drained_after_socializing'] = test['Drained_after_socializing'].fillna(test['Drained_after_socializing'].mode()[0])


'''test['Stage_fear'] = le1.transform(test['Stage_fear'])
test['Drained_after_socializing'] = le2.transform(test['Drained_after_socializing'])'''


test['Time_spent_Alone'] = np.log1p(test['Time_spent_Alone'])
#test = test.drop('Time_spent_Alone', axis=1)


num_col


#test[num_col] = scaler.transform(test[num_col])
#test[num_col] = np.log1p(test[num_col])


test.info()


preds = xgb_model.predict(test)
preds1 = lgb_model.predict(test)


preds_gbt = model_gbt.predict(test)
threshold = 0.5
preds_gbt = (preds_gbt[:] > threshold).astype(int)


preds_rfl = model_rfl.predict(test)
threshold = 0.5
preds_rfl = (preds_rfl[:] > threshold).astype(int)


submission['Personality'] = preds_rfl
submission['Personality'] = submission['Personality'].map({1:'Introvert', 0:'Extrovert'})
submission.to_csv('submission_rand_forest.csv', index=None)


submission['Personality'] = preds_gbt
submission['Personality'] = submission['Personality'].map({1:'Introvert', 0:'Extrovert'})
submission.to_csv('submission_ydf.csv', index=None)


submission['Personality'] = preds
submission['Personality'] = submission['Personality'].map({1:'Introvert', 0:'Extrovert'})
submission.to_csv('submission_xgb.csv', index=None)


submission['Personality'] = preds1
submission['Personality'] = submission['Personality'].map({1:'Introvert', 0:'Extrovert'})
submission.to_csv('submission_lgb.csv', index=None)


majority_preds = np.round((preds_gbt + preds_rfl + preds) / 3).astype(int)

submission['Personality'] = majority_preds
submission['Personality'] = submission['Personality'].map({1:'Introvert', 0:'Extrovert'})
submission.to_csv('submission.csv', index=None)

