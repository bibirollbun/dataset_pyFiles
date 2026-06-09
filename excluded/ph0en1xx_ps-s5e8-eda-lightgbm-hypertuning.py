import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
print(f"train shape : {train.shape}")
print(f"Test  shape : {test.shape}")


train.info()


train.head()


train.describe()


categorical = train.select_dtypes(include=['object']).columns
categorical


numerical = train.select_dtypes(include=['number']).columns
numerical


train.isnull().sum()


train.duplicated().sum()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,6))
sns.countplot(data=train,x='y')
sns.set_style('whitegrid')
plt.ylabel('Number of clients')
plt.xlabel('Target')
plt.tight_layout()
plt.show()


for col in categorical:
    print(f"\nUnique values in '{col}':")
    print(train[col].nunique())


for col in numerical.drop(['id','y']):
    plt.figure(figsize=(8, 6))
    sns.set_style('whitegrid')
    sns.histplot(train[col], kde=True)
    plt.xlabel=(col)
    plt.tight_layout()
    plt.show()
    print(col + ' stats: ','\n')
    print(train[col].describe(),'\n')


for col in categorical:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train,x=col,hue='y')
    sns.set_style('whitegrid')
    plt.title('Distribution by target')
    plt.legend(title='Target (y)', labels = ['No','Yes'])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(train[numerical.drop('id')].corr(),cmap='coolwarm', annot=True)
plt.title('Correlation between numerical features + target')


plt.figure(figsize=(8,8))
unknown_counts = train[categorical].isin(['unknown']).sum()
sns.barplot(x=unknown_counts.index, y=unknown_counts.values)
sns.set_style('whitegrid')
plt.ylabel('Number of unknown values')
plt.title('Unknown values per categorical feature')


train_og = train.copy()


from sklearn.preprocessing import LabelEncoder
for col in categorical:
    #fit encoder on training data only(preventing data leak)
    le = LabelEncoder()
    le.fit(train[col].astype(str))
    
    #transform train and test
    train[f'{col}_encoded'] = le.transform(train[col].astype(str))
    test[f'{col}_encoded'] = le.transform(test[col].astype(str))

# Drop original categorical columns from both
train_clean = train.drop(columns=categorical)
test_clean = test.drop(columns=categorical)


#education-job alignment  
train_clean['edu_job_match'] = train_clean['education_encoded'] * 10 + train_clean['job_encoded']

#balance categories
train_clean['balance_category'] = pd.cut(train_clean['balance'], 
                                       bins=[-np.inf, 0, 1000, 5000, np.inf], 
                                       labels=[0, 1, 2, 3])

#age groups
train_clean['age_group'] = pd.cut(train_clean['age'], 
                                bins=[0, 25, 35, 50, 65, 100], 
                                labels=[0, 1, 2, 3, 4]) #max age in this dataset is 95
#age-job interaction
train_clean['age_job_interaction'] = train_clean['age_group'].astype(int) * 100 + train_clean['job_encoded']

#campaign intensity  
train_clean['campaign_intensity'] = train_clean['campaign'] > train_clean['campaign'].median()

#previous contact success indicator
train_clean['had_previous_success'] = (train_clean['poutcome_encoded'] == 1).astype(int)

# Campaign burden
train_clean['high_campaign_pressure'] = (train_clean['campaign'] > 3).astype(int)
train_clean['campaign_per_previous'] = train_clean['campaign'] / (train_clean['previous'] + 1)  # Avoid div by 0



#education-job alignment  
test_clean['edu_job_match'] = test_clean['education_encoded'] * 10 + test_clean['job_encoded']

#balance categories
test_clean['balance_category'] = pd.cut(test_clean['balance'], 
                                       bins=[-np.inf, 0, 1000, 5000, np.inf], 
                                       labels=[0, 1, 2, 3])
#age groups
test_clean['age_group'] = pd.cut(test_clean['age'], 
                                bins=[0, 25, 35, 50, 65, 100], 
                                labels=[0, 1, 2, 3, 4]) #max age in this dataset is 95
#age-job interaction
test_clean['age_job_interaction'] = test_clean['age_group'].astype(int) * 100 + test_clean['job_encoded']

#campaign intensity  
test_clean['campaign_intensity'] = test_clean['campaign'] > test_clean['campaign'].median()

#previous contact success indicator
test_clean['had_previous_success'] = (test_clean['poutcome_encoded'] == 1).astype(int)

# Campaign burden
test_clean['high_campaign_pressure'] = (test_clean['campaign'] > 3).astype(int)
test_clean['campaign_per_previous'] = test_clean['campaign'] / (test_clean['previous'] + 1)  # Avoid div by 0


print("Unique month_encoded values:", sorted(train_clean['month_encoded'].unique()))


def create_cyclical_month(month_encoded):
    #since months are zero indexed - we need to add one to them
    month_num = month_encoded + 1
    
    month_sin = np.sin(2 * np.pi * month_num / 12)
    month_cos = np.cos(2 * np.pi * month_num / 12)
    return month_sin, month_cos

# Apply to both datasets
train_clean['month_sin'], train_clean['month_cos'] = create_cyclical_month(train_clean['month_encoded'])
test_clean['month_sin'], test_clean['month_cos'] = create_cyclical_month(test_clean['month_encoded'])







#realistic model without data leakage (duration is only known after the call ends,and in real prediction scenarios - you need to decide before/during the call)
features_realistic = [col for col in train_clean.columns 
                     if col not in ['duration', 'y']]
X_realistic = train_clean[features_realistic]

#competition model (with duration) 
features_competition = [col for col in train_clean.columns 
                       if col != 'y']
X_competition = train_clean[features_competition]

y = train_clean['y']


from sklearn.model_selection import train_test_split

#realistic
X_train_real, X_val_real, y_train, y_val = train_test_split(
    X_realistic, y, test_size=0.2, random_state=42, stratify=y
)

#competition 
X_train_comp, X_val_comp, y_train, y_val = train_test_split(
    X_competition, y, test_size=0.2, random_state=42, stratify=y
)


import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Realistic model
lgb_real = lgb.LGBMClassifier(random_state=101)
lgb_real.fit(X_train_real, y_train)
pred_real = lgb_real.predict_proba(X_val_real)[:, 1]
score_real = roc_auc_score(y_val, pred_real)

# Competition model
lgb_comp = lgb.LGBMClassifier(random_state=101)
lgb_comp.fit(X_train_comp, y_train)
pred_comp = lgb_comp.predict_proba(X_val_comp)[:, 1]
score_comp = roc_auc_score(y_val, pred_comp)

print(f"Realistic model AUC: {score_real:.4f}")
print(f"Competition model AUC: {score_comp:.4f}")


import matplotlib.pyplot as plt

# Compare what features matter in each model
feature_imp_real = lgb_real.feature_importances_
feature_imp_comp = lgb_comp.feature_importances_



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Realistic model
pd.Series(lgb_real.feature_importances_, index=X_realistic.columns).nlargest(10).plot(kind='barh', ax=ax1, title='Realistic Model')

# Competition model  
pd.Series(lgb_comp.feature_importances_, index=X_competition.columns).nlargest(10).plot(kind='barh', ax=ax2, title='Competition Model')

plt.tight_layout()
plt.show()


from sklearn.model_selection import RandomizedSearchCV

#hyperparameter tuning
param_dist = {
    'num_leaves': [20, 31, 50, 100, 150],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'min_data_in_leaf': [5, 10, 20, 50, 100],
    'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
}

lgb_random = RandomizedSearchCV(
    lgb.LGBMClassifier(random_state=42, verbose=-1),
    param_dist,
    n_iter=50,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)


lgb_random.fit(X_train_comp, y_train)

#get the best model 
best_lgb = lgb_random.best_estimator_
pred_comp = best_lgb.predict_proba(X_val_comp)[:, 1]
score_comp = roc_auc_score(y_val, pred_comp)

print(f"Best params: {lgb_random.best_params_}")
print(f"Realistic model AUC: {score_comp:.4f}")


features_competition = [col for col in train_clean.columns if col != 'y']
X_test_comp = test_clean[features_competition]
test_predictions = best_lgb.predict_proba(X_test_comp)[:, 1]


submission = pd.DataFrame({
    'id': test_clean['id'], 
    'y': test_predictions
})


submission.to_csv('submission.csv', index=False)
submission.head()

