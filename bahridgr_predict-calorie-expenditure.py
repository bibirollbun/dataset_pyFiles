import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
import optuna
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

train_id = train["id"].copy()
test_id = test["id"].copy()

train_df = train.drop("id",axis=1)
test_df = test.drop("id", axis=1)


# Checking the basic properties of data 
def check_data(dataframe):
    print("########################## HEAD ##########################")
    print(dataframe.head(3))
    print("########################## ISNULL(?) ##########################")
    print(dataframe.isna().sum())
    print("########################## INFO ##########################")
    print(dataframe.info())
    print("########################## SHAPE ##########################")
    print(dataframe.shape)
    print("########################## DESCRİBE ##########################")
    print(dataframe.describe([0.1, 0.25, 0.5, 0.75, 0.90, 0.99]).T)


check_data(test_df)


check_data(train_df)


def plot_numerical_distributions(df, numerical_cols):
    """
    Shows histogram and skewness values of given numeric columns as subplots.

    Args:
        df (pd.DataFrame): Data framework.
        numerical_cols (list): List of numeric columns to be plotted histograms.
        skew_columns (list): list of columns with skewness not between -0.5 and 0.5
    """
    skew_columns = []
    num_cols = len(numerical_cols)
    num_rows = (num_cols + 1) // 2  # Set the number of rows
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 6 * num_rows))
    axes = axes.flatten() # Convert 2-dimensional axis array to one dimensional

    for i, col in enumerate(numerical_cols):
        sns.histplot(x=df[col].dropna(), kde=True, bins=50, ax=axes[i])
        axes[i].set_title(f"{col} Distribution")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

        sk = skew(df[col].dropna())
        print(f"Skewness for {col}: {sk:.2f}")

        if sk > 1 or sk < -1:
            skew_columns.append(col)

    # Clear unused subplots
    if num_cols % 2 != 0:
        fig.delaxes(axes[-1])


    plt.tight_layout()
    plt.show()
    return skew_columns

numerical_col_train = [col for col in train_df.columns if train_df[col].dtype != 'O']
skew_list_train = plot_numerical_distributions(train_df, numerical_col_train)


numerical_col_test = [col for col in test_df.columns if test_df[col].dtype != 'O']
skew_list_test = plot_numerical_distributions(test_df, numerical_col_test)


correlation_matrix_train = train_df[numerical_col_train].corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix_train, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features for Train Dataset')
plt.show()



correlation_matrix_test = test_df[numerical_col_test].corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix_test, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features for Test Dataset')
plt.show()



train_df['New_HeartRateMinute'] = train_df['Heart_Rate'] / train_df['Duration']

train_df['New_HeartRateDuration'] = train_df['Heart_Rate'] * train_df['Duration']

train_df['New_DurationBodyTemp'] = train_df['Duration'] * train_df['Body_Temp']

train_df['New_AgeHeartRate'] = train_df['Age'] * train_df['Heart_Rate']

train_df['New_AgeBodyTemp'] = train_df['Age'] * train_df['Body_Temp']


train_df['New_BodyArea'] = train_df['Height'] * train_df['Weight']

train_df['New_DurationCategory'] = pd.qcut(train_df['Duration'], q=4, labels=['Sedentary', 'Lightly_Active', 'Moderately_Active', 'Very_Active'])


bins = [20, 30, 45, 60, 70, 80]
labels = ['Young', 'Adult', 'Middle_Aged', 'Old_Age', 'Old']
train_df['New_AgeCategory'] = pd.cut(train_df['Age'], bins=bins, labels=labels, right=False)


train_df.head()


test_df['New_HeartRateMinute'] = test_df['Heart_Rate'] / test_df['Duration']

test_df['New_HeartRateDuration'] = test_df['Heart_Rate'] * test_df['Duration']

test_df['New_DurationBodyTemp'] = test_df['Duration'] * test_df['Body_Temp']

test_df['New_AgeHeartRate'] = test_df['Age'] * test_df['Heart_Rate']

test_df['New_AgeBodyTemp'] = test_df['Age'] * test_df['Body_Temp']


test_df['New_BodyArea'] = test_df['Height'] * test_df['Weight']

test_df['New_DurationCategory'] = pd.qcut(test_df['Duration'], q=4, labels=['Sedentary', 'Lightly_Active', 'Moderately_Active', 'Very_Active'])


bins = [20, 30, 45, 60, 70, 80]
labels = ['Young', 'Adult', 'Middle_Aged', 'Old_Age', 'Old']
test_df['New_AgeCategory'] = pd.cut(test_df['Age'], bins=bins, labels=labels, right=False)


test_df.head()


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


label_encoder(train_df, 'Sex')

categorical_cols_train = [col for col in train_df.columns if 2 < train_df[col].nunique() < 10]
train_df = one_hot_encoder(train_df,categorical_cols_train, drop_first=True)


scaler = StandardScaler()
scale_cols_train = [col for col in train_df.columns if train_df[col].nunique() > 20 and train_df[col].dtypes != 'O']
scale_cols_train.remove('Calories')

train_df[scale_cols_train] = scaler.fit_transform(train_df[scale_cols_train])


label_encoder(test_df, 'Sex')

categorical_cols_test = [col for col in test_df.columns if 2 < test_df[col].nunique() < 10]
test_df = one_hot_encoder(test_df,categorical_cols_test, drop_first=True)


scaler = StandardScaler()
scale_cols_test = [col for col in test_df.columns if test_df[col].nunique() > 20 and test_df[col].dtypes != 'O']

test_df[scale_cols_test] = scaler.fit_transform(test_df[scale_cols_test])


train_df.head()


test_df.head()


def log_rmse_func(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return - np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


log_rmse_scorer = make_scorer(log_rmse_func, greater_is_better=True)


def compare_models(X, y, random_state=42, cv=3):
    # Models
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=random_state),
        "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=random_state),
        "Bagging Regressor": BaggingRegressor(n_estimators=50, random_state=random_state, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=50, random_state=random_state, verbosity=0, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=50, random_state=random_state, verbose=-1, n_jobs=-1),
        "CatBoost": CatBoostRegressor(verbose=False, random_state=random_state)
    }

    # Model scores
    results = []
    for name, model in models.items():
        try:
            score = -cross_val_score(model, X, y, scoring=log_rmse_scorer, cv=cv).mean()
            results.append({
                'Model': name,
                'Log RMSE': round(score, 4)
            })
        except Exception as e:
            results.append({
                'Model': name,
                'Log RMSE': f"Error: {e}"
            })

    return pd.DataFrame(results).sort_values("Log RMSE", ascending=True).reset_index(drop=True)


X = train_df.drop(['Calories'], axis=1)
y = train_df["Calories"]

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)

results = compare_models(X_train, y_train)
print(results)


catboost_model = CatBoostRegressor(random_state=42, verbose=False)


# hyperparameters optimization
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 700, 1200),
        "depth": trial.suggest_int("depth", 8, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 100, 300),
        "verbose": 0,
        "loss_function": "RMSE",
        "random_seed": 42
    }

    model = CatBoostRegressor(**params)

    # 5-Fold Cross Validation ve negatif RMSE
    scores = cross_val_score(model, X_train, y_train,
                             scoring=log_rmse_scorer,
                             cv=5, n_jobs=-1)

    return np.mean(scores)

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=30)


best_params = {'iterations': 1190,
               'depth': 12,
               'learning_rate': 0.08111993275911336,
               'l2_leaf_reg': 4.006075660366656,
               'bagging_temperature': 0.5861436328046665,
               'border_count': 109
               }

final_model = catboost_model.set_params(**best_params).fit(X_train, y_train)


y_pred_val = final_model.predict(X_val)

log_rmse = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(y_pred_val)))
print(f"Validation score:{log_rmse}")


y_pred_test = final_model.predict(test_df)

submission = pd.DataFrame({
    'id':test_id,
    'Calories':np.maximum(0, y_pred_test)
})


submission.to_csv("submission.csv", index=False)
submission.head()

