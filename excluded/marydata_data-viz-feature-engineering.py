import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor as xgb
from xgboost.callback import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import optuna


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv").set_index('id')
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv").set_index('id')

columns = train.columns.tolist()
train.head()


print(f"Is na in train {train.shape}: \n ============ \n {train.isna().sum().sum()}")
print(f"Is na in test {test.shape}: \n ============ \n {test.isna().sum().sum()}")


train.info()


train.describe()


def preparing_data(train):
    print(f"Missing Value : {train.isna().sum().sum()}\n")
    print("=" * 75)
    print(f"Duplicate value : {train.duplicated().sum().sum()}")
    print("=" * 75)

def print_count_values(dataframe, columns):
    for column in columns:
        count_value  = dataframe[column].value_counts()
        print(f"{column} number of occurences : {count_value}")
        print("=" * 75)
    
def print_nunique_values(dataframe, columns):
    for column in columns:
        unique_values = dataframe[column].nunique()
        print(f"{column} unique value : {unique_values}")
        print("=" * 75)


preparing_data(train)
preparing_data(test)


print_count_values(train, columns[0:1])
print_count_values(test, columns[0:1])


print_nunique_values(train, columns)
print_nunique_values(test, columns[0:6])


train.hist(figsize=(20, 20), bins=50, xlabelsize=8, ylabelsize=8, grid=True, edgecolor="black")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.suptitle("Histogram of all features", fontsize=20)
plt.xlabel("Value", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


sns.heatmap(train.drop('Sex', axis=1).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


#pairplot
sns.pairplot(train, diag_kind='kde')
plt.suptitle("Pairplot of all features", fontsize=20)
plt.show()


plt.figure(figsize=(10, 6))
train['Sex'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title("Pie Chart of Sex")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Distribution of Variables by Sex', fontsize=16, y=1.02)

# Flatten axes for easier iteration
axes = axes.flatten()

# Variables to plot (excluding 'Sex')
numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# Create boxplots
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=train, x='Sex', y=col, ax=axes[i])
    axes[i].set_title(f'{col} Distribution by Sex')
    axes[i].set_xlabel('Sex')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()


sex_mean = train.groupby('Sex').mean()[['Height', 'Weight', 'Calories']]
fig, ax = plt.subplots(figsize=(10, 6))
sex_mean.plot(kind='bar', ax=ax)
ax.set_title('Average Height, Weight, and Calories')
ax.set_ylabel('Average Value')
ax.set_xlabel('Sex')
plt.xticks(rotation=0)
plt.show()


bins = [0, 20, 40, 60, 80, 100]
labels = ['0-19', '20-39', '40-59', '60-79', '80+']
age_set = train.copy()
age_set['Age_Group'] = pd.cut(age_set['Age'], bins=bins, labels=labels)

age_sex_calories = age_set.groupby(['Sex', 'Age_Group'])['Calories'].sum().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))
age_sex_calories.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Calories by grouped Age')
ax.set_xlabel('section of age')
ax.set_ylabel('Calories')
plt.show()


def feature_engineering(data):
    data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
    
    data['Work_intensity'] = data['Heart_Rate'] / (data['Duration'])
    
    data['King_Age'] = pd.cut(data['Age'], bins=[0, 20, 35, 50, 65, 100],
                               labels=['Teenager', 'Young Adult', 'Adult', 'Middle Age', 'Senior'])
    
    def heart_rate_classifier(hr):
        if hr < 90:
            return 'Low'
        elif hr < 120:
            return 'Moderate'
        else:
            return 'High'
    
    data['H_Rate'] = data['Heart_Rate'].apply(heart_rate_classifier)
    return data  


train = feature_engineering(train)
test = feature_engineering(test) 
cols = train.columns.tolist()


print_nunique_values(train,columns=cols)


print_count_values(train, columns=cols[9:])


plt.bar(train['King_Age'].value_counts().index, train['King_Age'].value_counts(), color='skyblue')
plt.title('Count of King_Age')
plt.xlabel('King_Age')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


plt.bar(train['H_Rate'].value_counts().index, train['H_Rate'].value_counts(), color='lightgreen')
plt.title('Count of Heart Rate')
plt.xlabel('Heart Rate')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


plt.bar(train['BMI'].value_counts().index, train['BMI'].value_counts(), color='salmon')
plt.title('Count of BMI')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(train.select_dtypes(include=['int64', 'float64']).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


col_encoded = ['Sex', 'King_Age', 'H_Rate']
for column in col_encoded:
    le = LabelEncoder()
    train[column] = le.fit_transform(train[column])
    test[column] = le.transform(test[column])
train


def optimisation(data):
    for col in data.select_dtypes(include=['int64']).columns:
        data[col] = pd.to_numeric(data[col], downcast='integer')
    for col in data.select_dtypes(include=['float64']).columns:
        data[col] = pd.to_numeric(data[col], downcast='float')
    return data


train = optimisation(train)
test = optimisation(test)
train.info()


def add_pairwise_combinations(df, columns=None, operations=['sum', 'diff', 'prod', 'ratio'], row_stats=None):
    """
    Creates pairwise combinations and optional row-wise statistics.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to use. If None, all numeric columns except 'Calories'.
        operations (list): Pairwise operations: 'sum', 'diff', 'prod', 'ratio'
        row_stats (list): Row-wise stats to calculate: 'mean', 'std', 'min', 'max', 'median', 'count'

    Returns:
        df (pd.DataFrame): Updated DataFrame
    """
    if columns is None:
        columns = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'Calories']

    # Pairwise combinations
    for col1, col2 in tqdm(combinations(columns, 2), total=len(columns)*(len(columns)-1)//2):
        if 'sum' in operations:
            df[f'{col1}_{col2}_sum'] = df[col1] + df[col2]
        if 'diff' in operations:
            df[f'{col1}_{col2}_diff'] = df[col1] - df[col2]
            df[f'{col2}_{col1}_diff'] = df[col2] - df[col1]
        if 'prod' in operations:
            df[f'{col1}_{col2}_prod'] = df[col1] * df[col2]
        if 'ratio' in operations:
            df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-6)
            df[f'{col2}_{col1}_ratio'] = df[col2] / (df[col1] + 1e-6)

    # Row-wise statistics
    if row_stats:
        row_data = df[columns]
        if 'mean' in row_stats:
            df['row_mean'] = row_data.mean(axis=1)
        if 'median' in row_stats:
            df['row_median'] = row_data.median(axis=1)

    return df


train = add_pairwise_combinations(train)
test = add_pairwise_combinations(test)


def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

# Custom RMSLE eval function for XGBoost (for early stopping)
def rmsle_xgb_eval(preds, dtrain):
    labels = dtrain.get_label()
    score = rmsle(labels, preds)
    return 'rmsle', score


# Split features and target
X = train.drop('Calories', axis=1)
y = train['Calories']

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store validation scores
scores = []

# Loop over folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Initialize model
    model = xgb(
        tree_method='hist',
        enable_categorical=True,
        device='cuda',
        max_depth=9,
        colsample_bynode=0.3,
        subsample=0.8,
        n_estimators=50_000,
        learning_rate=0.01,
        min_child_weight=10,
    )
    
    # Fit with early stopping using custom RMSLE
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=rmsle_xgb_eval,
        early_stopping_rounds=500,
        verbose=False
    )
    
    # Predict and evaluate using RMSLE
    y_pred = model.predict(X_val)
    score = rmsle(y_val, y_pred)
    print(f"RMSLE: {score:.4f}")
    scores.append(score)

# Average score across folds
print(f"\nAverage RMSLE: {np.mean(scores):.4f}")


# submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
# test_pred = model.predict(test)
# submission['Calories'] = test_pred
# submission.to_csv('Playground_XBG_may_2025.csv',index = False)
# submission.head()


pipeline = Pipeline([
    ('model', CatBoostRegressor(silent=True))
])
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
        'depth': trial.suggest_int('depth', 3, 10, step=1),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, step=1),
        'random_strength': trial.suggest_float('random_strength', 1, 10, step=1),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1, step=0.1),
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 50, 200, step=25)
    }
    
    model = CatBoostRegressor(**params, silent=True)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("Best parameters: ", best_params)


model = CatBoostRegressor(**best_params, silent=True)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
rmsle = rmsle(y_test, y_pred)
print(f"RMSLE: {rmsle}")


submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
test_pred = model.predict(test)
submission['Calories'] = test_pred
submission.to_csv('Playground_CATBOOST_may_2025.csv',index = False)
submission.head()

