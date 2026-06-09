import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import XGBRegressor

import optuna
from optuna.samplers import TPESampler

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


sns.set_theme(style="darkgrid", palette="deep", font="Arial", font_scale=1.2)


train.info()


train.head()


train = train.drop('id', axis=1)
test_ids = test.pop('id')


train.head()


train.describe()


def sex_category(sex):
    if sex == 'female':
        return 0
    else:
        return 1

train['Sex'] = train['Sex'].apply(sex_category)
test['Sex'] = test['Sex'].apply(sex_category)


train.head()


sns.histplot(data=train, x='Height', bins=35)


def age_category(age):
    if age < 18:
        return 0
    elif 18 <= age < 50:
        return 1
    else:
        return 2

train['Age Group'] = train['Age'].apply(age_category)
test['Age Group'] = test['Age'].apply(age_category)


train.head()


def height_category(height):
    if height < 164:
        return 0
    elif 164 <= height < 174:
        return 1
    elif 174 <= height < 185:
        return 2
    else:
        return 3

train['Height Group'] = train['Height'].apply(height_category)
test['Height Group'] = test['Height'].apply(height_category)


def weight_category(weight):
    if weight < 63:
        return 0
    elif 63 <= weight < 74:
        return 1
    elif 74 <= weight < 87:
        return 2
    else:
        return 3

train['Weight Group'] = train['Weight'].apply(weight_category)
test['Weight Group'] = test['Weight'].apply(weight_category)


def heart_rate_category(hr):
    if hr < 88:
        return 0
    elif 88 <= hr < 95:
        return 1
    elif 95 <= hr < 103:
        return 2
    else:
        return 3

train['Heart Rate Group'] = train['Heart_Rate'].apply(heart_rate_category)
test['Heart Rate Group'] = test['Heart_Rate'].apply(heart_rate_category)


def body_temp_category(temp):
    if temp < 39.6:
        return 0
    elif 39.6 <= temp < 40.3:
        return 1
    elif 40.3 <= temp < 40.7:
        return 2
    else:
        return 3

train['Body Temp Group'] = train['Body_Temp'].apply(body_temp_category)
test['Body Temp Group'] = test['Body_Temp'].apply(body_temp_category)


calories = train.pop('Calories')
train['Calories'] = calories


train.head()


corr_matrix = train.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='magma')
plt.title("Correlation heatmap")
plt.tight_layout()
plt.show()


import matplotlib.cm as cm

def plot_distribution_and_boxplots(train_df, test_df, target_column,
                                   palette="plasma", title_fontsize=16,
                                   label_fontsize=12, grid_style="--", grid_alpha=0.5):

    numeric_cols_train = set(train_df.select_dtypes(include=['number']).columns)
    numeric_cols_test = set(test_df.select_dtypes(include=['number']).columns)
    common_numeric_cols = list(numeric_cols_train & numeric_cols_test)

    if target_column in common_numeric_cols:
        common_numeric_cols.remove(target_column)

    if not common_numeric_cols:
        print("No cross features.")
        return

    cmap = cm.get_cmap(palette)
    colors = [cmap(0.3), cmap(0.5), cmap(0.75)]

    fig, axes = plt.subplots(len(common_numeric_cols), 3, figsize=(20, len(common_numeric_cols) * 4))

    for i, col in enumerate(common_numeric_cols):
        sns.histplot(train_df[col], bins=30, kde=True, color=colors[0], ax=axes[i, 0])
        axes[i, 0].set_title(f'Distribution of "{col}" (Train)', fontsize=title_fontsize)
        axes[i, 0].set_xlabel(col, fontsize=label_fontsize)
        axes[i, 0].set_ylabel('Frequency', fontsize=label_fontsize)
        axes[i, 0].grid(True, linestyle=grid_style, alpha=grid_alpha)

        sns.boxplot(x=train_df[col], color=colors[1], ax=axes[i, 1])
        axes[i, 1].set_title(f'Boxplot of "{col}" (Train)', fontsize=title_fontsize)
        axes[i, 1].set_xlabel(col, fontsize=label_fontsize)
        axes[i, 1].grid(True, linestyle=grid_style, alpha=grid_alpha)

        sns.boxplot(x=test_df[col], color=colors[2], ax=axes[i, 2])
        axes[i, 2].set_title(f'Boxplot of "{col}" (Test)', fontsize=title_fontsize)
        axes[i, 2].set_xlabel(col, fontsize=label_fontsize)
        axes[i, 2].grid(True, linestyle=grid_style, alpha=grid_alpha)

    plt.tight_layout()
    plt.show()


columns = train.columns.tolist()

for column in columns:
    if column != 'Calories':
        plot_distribution_and_boxplots(train, test, column)
    else:
        pass


y = train.pop('Calories')
X = train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'random_state': 42
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return rmsle(y_test, y_pred) 

pbar = None

def callback(study, trial):
    pbar.update(1)

best_params = {
    'n_estimators': 252, 
    'max_depth': 10, 
    'learning_rate': 0.07400916835251589, 
    'subsample': 0.9281982878733435, 
    'colsample_bytree': 0.9522953863115337, 
    'min_child_weight': 5, 
    'gamma': 1.9023714771457327, 
    'reg_alpha': 6.639043794554961, 
    'reg_lambda': 7.699039618845956
}

best_value = 0.06059518985303128

print("Best params:", best_params)
print("Best RMSLE:", best_value)


model = XGBRegressor(**best_params, random_state=42)
model.fit(X, y)

preds = model.predict(test)

submission = pd.DataFrame({
    'id': test_ids,
    'Calories': preds
})


submission.head()


submission.to_csv('submission.csv', index=False)

