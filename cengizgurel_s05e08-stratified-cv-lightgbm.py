import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import lightgbm as lgb
pd.set_option('display.max_columns', None)


#Load training and test datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

#Display the shapes
print(f'train_df shape:{train_df.shape}')
print(f'test_df shape:{test_df.shape}')


train_df.head(10)


#Check missing values for train and test datasets
print(train_df.isna().sum())
print('\n')
print(test_df.isna().sum())


#Count unique values for numeric features in the training set
print("Number of unique values in numeric columns:")
print(train_df.select_dtypes(include='number').nunique())

print("\n")

#Count unique values for categorical features in the training set
print("Number of unique values in categorical columns:")
print(train_df.select_dtypes(include='object').nunique())


#Check the data types
train_df.dtypes


#Figure size for the plot
plt.figure(figsize=(6,6))

#Create a pie chart
counts = train_df['y'].value_counts().sort_index()  # ensures 0 comes before 1
labels = ['Class 0', 'Class 1']

plt.pie(counts, labels=labels, autopct='%1.1f%%', colors = ['#a9a9a9', '#e74c3c'])
plt.title('Target y ')


#List of categorical features
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'month', 'contact', 'poutcome']

#Figure size for the plots
fig = plt.figure(figsize=(20, 5 * len(categorical_features)))  

for i, var_name in enumerate(categorical_features):
    ax = fig.add_subplot(len(categorical_features), 1, i + 1)
    sns.countplot(data=train_df, x=var_name, hue='y', ax=ax, palette = ['#a9a9a9', '#e74c3c'])
    ax.set_title(var_name)

plt.tight_layout()
plt.show()


#Figure size for the plot
plt.figure(figsize=(6, 6))

#Histogram of ages
sns.histplot(data=train_df, x='age', binwidth=2, kde=True, hue=train_df['y'], palette = ['#000000', '#e74c3c'])

plt.title('Age Distribution by Subscription Status (y)', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()


#Figure size for the plot
plt.figure(figsize=(6, 6))

#Histogram of balances
sns.histplot(data=train_df, x='balance', bins=100, hue=train_df['y'], palette = ['#000000', '#e74c3c'])

plt.title('Balance Distribution by Subscription Status (y)', fontsize=16)
plt.xlabel('Balance', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()


#Figure size for the plot
plt.figure(figsize=(6, 6))

#Histogram of days
sns.histplot(data=train_df, x='day', binwidth=1, kde=True, hue=train_df['y'], palette = ['#000000', '#e74c3c'])

plt.title('Day Distribution by Subscription Status (y)', fontsize=16)
plt.xlabel('Day', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()


#Figure size for the plot
plt.figure(figsize=(6, 6))

#Histogram of Durations
sns.histplot(data=train_df, x='duration', bins=100, hue=train_df['y'], palette = ['#000000', '#e74c3c'])

plt.title('Duration Distribution by Subscription Status (y)', fontsize=16)
plt.xlabel('Duration', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()


#Figure size for the plot
plt.figure(figsize=(6, 6))

#Histogram of campaigns
sns.histplot(data=train_df, x='campaign', bins=50, hue=train_df['y'], palette = ['#000000', '#e74c3c'])

plt.title('Campaign Distribution by Subscription Status (y)', fontsize=16)
plt.xlabel('Campaign', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()


#Figure size for the plot
plt.figure(figsize=(6, 6))

#Histogram of pdays
sns.histplot(data=train_df, x='pdays', bins=100, hue=train_df['y'], palette = ['#000000', '#e74c3c'])

plt.title('pdays Distribution by Subscription Status (y)', fontsize=16)
plt.xlabel('pdays', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()


#Figure size for the plot
plt.figure(figsize=(6, 6))

#Histogram of previous
sns.histplot(data=train_df, x='previous', bins=50, hue=train_df['y'], palette = ['#000000', '#e74c3c'])

plt.title('Previous Distribution by Subscription Status (y)', fontsize=16)
plt.xlabel('Previous', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()


#Correlation heatmap for numeric features
corr = train_df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='RdBu')

plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
plt.show()


#Remove outliers
train_df = train_df[(train_df['balance'] > -3000) & (train_df['balance'] < 78000)]
train_df = train_df[train_df['duration'] < 3750]
train_df = train_df[train_df['previous'] < 75]
train_df = train_df[train_df['campaign'] < 55]


#Save ids for later use
train_id = train_df['id']
test_id = test_df['id']

#Drop 'id' columns
train_df.drop('id', axis=1, inplace=True)
test_df.drop('id', axis=1, inplace=True)

#Store the number of rows in train and test sets for later use
ntrain = train_df.shape[0]
ntest = test_df.shape[0]

#Concatenating train and test for a consistent feature engineering
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['y'], axis=1, inplace=True)

#Save target variable
y_train = train_df['y']

#Check combined datasets shape
print(f"all_data shape: {all_data.shape}")


#'month' map
month_map = {'jan':1, 'feb':2, 'mar':2, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
all_data['month_num'] = all_data['month'].map(month_map)

#Drop after mapping
all_data.drop('month', axis=1, inplace=True)


#Aggregate 'duration' stats by 'default'
agg_default_duration = all_data.groupby('default')['duration'].agg(['mean', 'max', 'min', 'count'])
agg_default_duration.columns = ['default_duration_mean', 'default_duration_max', 'default_duration_min', 'default_duration_count']
all_data = all_data.merge(agg_default_duration, on='default', how='left')

#Aggregate 'duration' stats by 'housing'
agg_housing_duration = all_data.groupby('housing')['duration'].agg(['mean', 'max', 'min', 'count'])
agg_housing_duration.columns = ['housing_duration_mean', 'housing_duration_max', 'housing_duration_min', 'housing_duration_count']
all_data = all_data.merge(agg_housing_duration, on='housing', how='left')

#Aggregate 'duration' stats by 'loan'
agg_loan_duration = all_data.groupby('loan')['duration'].agg(['mean', 'max', 'min', 'count'])
agg_loan_duration.columns = ['loan_duration_mean', 'loan_duration_max', 'loan_duration_min', 'loan_duration_count']
all_data = all_data.merge(agg_loan_duration, on='loan', how='left')


#Apply log transformation
all_data['log_balance'] = np.log1p(all_data['balance'])
all_data['log_duration'] = np.log1p(all_data['duration'])
all_data['log_campaign'] = np.log1p(all_data['campaign'])
all_data['log_pdays'] = np.log1p(all_data['pdays'] + 1)
all_data['log_previous'] = np.log1p(all_data['previous'])

#Drop after transformation
all_data.drop(['balance', 'duration', 'campaign', 'pdays', 'previous'], axis=1, inplace=True)


#List of features
le_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']

le = LabelEncoder()

for i in le_cols:
    all_data[i] = le.fit_transform(all_data[i])


all_data.head()


#Split the combined dataset
train_df_new = all_data[:ntrain]
test_df_new = all_data[ntrain:]


X = train_df_new
y = y_train


#Model parameters
params = {
    'n_estimators': 30000,
    'learning_rate': 0.06,
    'learning_rate': 0.06,
    'num_leaves': 100,
    'max_depth': 10,
    'min_child_samples': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'reg_lambda': 0.3,
    'reg_alpha': 0.8,
    'max_bin': 4801,
    'random_state': 2003,
    'boosting_type': 'gbdt',
    'eval_metric': 'auc',
    'class_weight': 'balanced',
    'metric': 'auc',
    'verbosity': -1
}


n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=12)

y_prob = np.zeros(len(test_df_new))
val_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"\nTraining fold {fold + 1}/{n_splits}")

    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = X.iloc[val_index], y.iloc[val_index]

    model = LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(300),
            lgb.log_evaluation(500)
        ]
    )

    # Predict on validation set for scoring
    val_pred = model.predict_proba(X_val)[:, 1]
    fold_score = roc_auc_score(y_val, val_pred)
    val_scores.append(fold_score)
    print(f"Fold {fold + 1} AUC: {fold_score:.4f}")

    # Aggregate test predictions
    y_prob += model.predict_proba(test_df_new)[:, 1] / n_splits

print(f"\nMean AUC across folds: {np.mean(val_scores):.6f}")


sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
sub['y'] = y_prob
sub.to_csv("submission.csv", index=False)
sub.head()

