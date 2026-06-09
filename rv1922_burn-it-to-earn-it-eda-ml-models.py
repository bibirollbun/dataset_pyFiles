import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt  
import optuna
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio  
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import make_scorer, mean_squared_log_error
import time
import xgboost as xgb
import plotly.figure_factory as ff  
pio.renderers.default = 'iframe_connected'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.head()


train.info()


train.describe().round(2)


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())

print("Number of Rows:",train.shape[0])

print("Number of Columns:",train.shape[1])


object_column_names = train.select_dtypes(include=['object']).columns
print("Object Column Names:", object_column_names.tolist())


numerical_column_names = train.select_dtypes(include=['number']).columns
print("Numerical Column Names:", numerical_column_names.tolist())


ice_palette = ['#e0f7fa', '#b2ebf2', '#4dd0e1', '#00acc1', '#007c91']
cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


plt.figure(figsize=(10, 5))
sns.histplot(train['Calories'], bins=50, kde=True, color='#4dd0e1', edgecolor='black')
plt.title('Distribution of Calories Burned', fontsize=16, weight='bold')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


sex_count = train['Sex'].value_counts().reset_index()
sex_count.columns = ['Sex','Count']

fig = px.pie(
    sex_count,
    names ='Sex',
    values = "Count",
    color = 'Sex',
    color_discrete_sequence = ice_palette,
    title = "Sex Distribution"
)
fig.update_traces(textinfo = 'percent+label')
fig.update_layout(width = 500, height=500)

fig.show()


fig, axes = plt.subplots(nrows=len(cols), ncols=1, figsize=(8, 18))

for i, col in enumerate(cols):
    sns.histplot(train[col], bins=50, kde=True, ax=axes[i], color=ice_palette[i % len(ice_palette)], edgecolor='black', linewidth=0.5)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


fig = px.scatter(
    train,
    x='Heart_Rate',
    y='Calories',
    color='Sex',
    title='Calories Burned vs Heart Rate',
    labels={
        'Heart_Rate': 'Heart Rate (bpm)',
        'Calories': 'Calories Burned (kcal)',
        'Sex': 'Gender'
    },
     color_discrete_sequence=ice_palette
)
fig.show()


fig = px.scatter(
    train,
    x='Height',
    y='Weight',
    color='Sex',
    title='Height & Weight vs Heart Rate',
    labels={
        'Height': 'Height',
        'Weight': 'Weight',
        'Sex': 'Gender'
    },
    color_discrete_sequence=ice_palette
)

fig.show()


sorted_data = train.sort_values('Duration')
fig = px.line(
    sorted_data,
    x='Duration',
    y='Calories',
    markers=True,
    title='Calories Burned vs Exercise Duration',
    color_discrete_sequence=ice_palette,
    labels={'Duration': 'Exercise Duration (minutes)', 'Calories': 'Calories Burned'}
)

fig.update_layout(
    width=700,
    height=400,
    font=dict(size=14),
    plot_bgcolor='white',
    title_x=0.5
)

fig.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Violin Plot: Body_Temp vs Sex
sns.violinplot(x='Sex', y='Body_Temp', data=train, ax=axes[0], palette=ice_palette)
axes[0].set_title('Sex vs Body Temp')
axes[0].set_xlabel('Sex')
axes[0].set_ylabel('Body Temperature')

# 2. Violin Plot: Heart_Rate vs Sex
sns.violinplot(x='Sex', y='Heart_Rate', data=train, ax=axes[1], palette=ice_palette)
axes[1].set_title('Sex vs Heart Rate')
axes[1].set_xlabel('Sex')
axes[1].set_ylabel('Heart Rate')

# 3. Violin Plot: Calories vs Sex
sns.violinplot(x='Sex', y='Calories', data=train, ax=axes[2], palette=ice_palette)
axes[2].set_title('Sex vs Calories')
axes[2].set_xlabel('Sex')
axes[2].set_ylabel('Calories Burned')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


sns.kdeplot(train, x="Calories", hue="Sex", fill=True)


for col in cols:
    print(f"\nAverage {col} by Sex:")
    print(train.groupby('Sex')[col].mean().round(2))
    print("-"*30)


corr = train[cols].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


le =  LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])


# BASIC PHYSIOLOGICAL FEATURES

train['BMI'] = train['Weight'] / (train['Height'] / 100) ** 2
train['BSA'] = 0.007184 * (train['Weight'] ** 0.425) * (train['Height'] ** 0.725)

train['BMR_Male'] = 88.362 + (13.397 * train['Weight']) + (4.799 * train['Height']) - (5.677 * train['Age'])
train['BMR_Female'] = 447.593 + (9.247 * train['Weight']) + (3.098 * train['Height']) - (4.330 * train['Age'])

train['BMR'] = np.where(train['Sex'] == 'male', train['BMR_Male'], train['BMR_Female'])

train['Max_HR'] = 220 - train['Age']
train['HR_Reserve'] = train['Max_HR'] - train['Heart_Rate']
train['HR_Percentage'] = (train['Heart_Rate'] / train['Max_HR']) * 100


# INTERACTION FEATURES

train['Intensity_Score'] = train['Heart_Rate'] * train['Duration']

train['Weight_Duration'] = train['Weight'] * train['Duration']
train['BMI_Duration'] = train['BMI'] * train['Duration']

train['HR_per_Weight'] = train['Heart_Rate'] / train['Weight']

train['Temp_HR_Ratio'] = train['Body_Temp'] / train['Heart_Rate']

train['Age_Weight'] = train['Age'] * train['Weight']
train['Age_HR'] = train['Age'] * train['Heart_Rate']


# NON-LINEAR TRANSFORMATIONS
train['Log_Duration'] = np.log1p(train['Duration'])  
train['Log_Weight'] = np.log1p(train['Weight'])
train['Log_HR'] = np.log1p(train['Heart_Rate'])

train['Sqrt_Duration'] = np.sqrt(train['Duration'])
train['Sqrt_Weight'] = np.sqrt(train['Weight'])
train['Sqrt_Age'] = np.sqrt(train['Age'])

train['Duration_Squared'] = train['Duration'] ** 2
train['HR_Squared'] = train['Heart_Rate'] ** 2
train['Weight_Squared'] = train['Weight'] ** 2
train['BMI_Squared'] = train['BMI'] ** 2

train['Duration_Cubed'] = train['Duration'] ** 3
train['HR_Cubed'] = train['Heart_Rate'] ** 3


# ADVANCED NON-LINEAR FEATURES 
train['Exp_HR_Norm'] = np.exp((train['Heart_Rate'] - train['Heart_Rate'].mean()) / train['Heart_Rate'].std())
train['Exp_Duration_Norm'] = np.exp((train['Duration'] - train['Duration'].mean()) / train['Duration'].std())

train['Sin_HR'] = np.sin(2 * np.pi * (train['Heart_Rate'] - train['Heart_Rate'].min()) / 
                         (train['Heart_Rate'].max() - train['Heart_Rate'].min()))
train['Cos_Duration'] = np.cos(2 * np.pi * (train['Duration'] - train['Duration'].min()) / 
                               (train['Duration'].max() - train['Duration'].min()))

train['Weight_Height_Ratio'] = train['Weight'] / train['Height']
train['HR_Temp_Product'] = train['Heart_Rate'] * train['Body_Temp']


le =  LabelEncoder()
test['Sex'] = le.fit_transform(test['Sex'])


# BASIC PHYSIOLOGICAL FEATURES

test['BMI'] = test['Weight'] / (test['Height'] / 100) ** 2
test['BSA'] = 0.007184 * (test['Weight'] ** 0.425) * (test['Height'] ** 0.725)

test['BMR_Male'] = 88.362 + (13.397 * test['Weight']) + (4.799 * test['Height']) - (5.677 * test['Age'])
test['BMR_Female'] = 447.593 + (9.247 * test['Weight']) + (3.098 * test['Height']) - (4.330 * test['Age'])

test['BMR'] = np.where(test['Sex'] == 'male', test['BMR_Male'], test['BMR_Female'])

test['Max_HR'] = 220 - test['Age']
test['HR_Reserve'] = test['Max_HR'] - test['Heart_Rate']
test['HR_Percentage'] = (test['Heart_Rate'] / test['Max_HR']) * 100


# INTERACTION FEATURES

test['Intensity_Score'] = test['Heart_Rate'] * test['Duration']

test['Weight_Duration'] = test['Weight'] * test['Duration']
test['BMI_Duration'] = test['BMI'] * test['Duration']

test['HR_per_Weight'] = test['Heart_Rate'] / test['Weight']

test['Temp_HR_Ratio'] = test['Body_Temp'] / test['Heart_Rate']

test['Age_Weight'] = test['Age'] * test['Weight']
test['Age_HR'] = test['Age'] * test['Heart_Rate']


# NON-LINEAR TRANSFORMATIONS

test['Log_Duration'] = np.log1p(test['Duration'])  
test['Log_Weight'] = np.log1p(test['Weight'])
test['Log_HR'] = np.log1p(test['Heart_Rate'])

test['Sqrt_Duration'] = np.sqrt(test['Duration'])
test['Sqrt_Weight'] = np.sqrt(test['Weight'])
test['Sqrt_Age'] = np.sqrt(test['Age'])

test['Duration_Squared'] = test['Duration'] ** 2
test['HR_Squared'] = test['Heart_Rate'] ** 2
test['Weight_Squared'] = test['Weight'] ** 2
test['BMI_Squared'] = test['BMI'] ** 2

test['Duration_Cubed'] = test['Duration'] ** 3
test['HR_Cubed'] = test['Heart_Rate'] ** 3


# ADVANCED NON-LINEAR FEATURES
test['Exp_HR_Norm'] = np.exp((test['Heart_Rate'] - test['Heart_Rate'].mean()) / test['Heart_Rate'].std())
test['Exp_Duration_Norm'] = np.exp((test['Duration'] - test['Duration'].mean()) / test['Duration'].std())

test['Sin_HR'] = np.sin(2 * np.pi * (test['Heart_Rate'] - test['Heart_Rate'].min()) / 
                        (test['Heart_Rate'].max() - test['Heart_Rate'].min()))
test['Cos_Duration'] = np.cos(2 * np.pi * (test['Duration'] - test['Duration'].min()) / 
                              (test['Duration'].max() - test['Duration'].min()))

test['Weight_Height_Ratio'] = test['Weight'] / test['Height']
test['HR_Temp_Product'] = test['Heart_Rate'] * test['Body_Temp']


train.head()


target_col = 'Calories'

num_cols = train.select_dtypes(include=['number']).columns

target_corr = train[num_cols].corr()[target_col].sort_values(ascending=False)

print("Correlation of Numerical Features with Target ('Calories'):\n")
print(target_corr)


drop_col = 'Calories'
X = train.drop(columns=[drop_col])
y = np.log1p(train[drop_col].values)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def objective(trial):
    params = {
        'verbosity': 0,
        'random_state': 42,
        'tree_method': 'hist',
        'booster': 'gbtree',
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'lambda': trial.suggest_float('lambda', 1e-4, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-4, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
    }

    model = xgb.XGBRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle_scores = []

    for train_idx, valid_idx in kf.split(X):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]

        model.fit(X_train_fold, y_train_fold)

        preds = model.predict(X_valid_fold)
        preds = np.maximum(preds, 0)  # prevent negative predictions before log1p inverse
        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid_fold), np.expm1(preds)))
        rmsle_scores.append(rmsle)

    return np.mean(rmsle_scores)


#study = optuna.create_study(direction='minimize')  # minimizing RMSLE
#study.optimize(objective, n_trials=5, timeout=3600)

#print("Best RMSLE: {:.5f}".format(study.best_value))
#print("Best Parameters:", study.best_params)


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


xgb_params = {
    'grow_policy': 'lossguide',
    'lambda': 0.02046735930064088,
    'alpha': 0.0005215538755506407,
    'colsample_bytree': 0.5630947597366918,
    'subsample': 0.6991548866344726,
    'learning_rate': 0.03414808413524688,
    'n_estimators': 1420,
    'max_depth': 9,
    'min_child_weight': 10,
    'gamma': 0.0843351689779594,
    'verbosity': 0,
    'random_state': 42
}

# Initialize KFold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize arrays
oof_preds_log = np.zeros(len(train))
test_preds_log = np.zeros(len(test))

# Loop through folds
for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              early_stopping_rounds=50,
              verbose=False)

    # Predict on validation and test
    oof_preds_log[valid_idx] = model.predict(X_valid)
    test_preds_log += model.predict(test) / n_splits

    # Calculate and print fold RMSLE
    oof_valid_exp = np.expm1(oof_preds_log[valid_idx])
    y_valid_exp = np.expm1(y_valid)
    fold_score = rmsle(y_valid_exp, oof_valid_exp)
    print(f"Fold {fold + 1} RMSLE: {fold_score:.5f}")

# Final out-of-fold performance
oof_preds_exp = np.expm1(oof_preds_log)
true_exp = np.expm1(y)
final_rmsle = rmsle(true_exp, oof_preds_exp)
print(f"\nOverall OOF RMSLE: {final_rmsle:.5f}")


test_preds_combined_log = test_preds_log

df_submission = submission.copy()

df_submission[drop_col] = np.expm1(test_preds_combined_log)

df_submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
print(df_submission.head())


plt.figure(figsize=(10, 6))
plt.hist(df_submission[drop_col], bins=50, color='mediumvioletred', edgecolor='black')
plt.title("Distribution of Predicted Prices")
plt.xlabel("Predicted Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

