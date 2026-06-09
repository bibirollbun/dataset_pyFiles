!pip install optuna


import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/innovative-ai-challenge-2024/train.csv')
test = pd.read_csv('/kaggle/input/innovative-ai-challenge-2024/test.csv')
sub = pd.read_csv('/kaggle/input/innovative-ai-challenge-2024/sample_submission.csv')
train.drop(columns=['State'], inplace=True)
test.drop(columns=['State'], inplace=True)


print(train.shape, test.shape, sub.shape)


train.head()


# converting categorical features to numerical features:

categorical_features = ['Crop_Type', 'Soil_Type']
encoders = {}
for feature in categorical_features:
    encoders[feature] = LabelEncoder()
    train[feature] = encoders[feature].fit_transform(train[feature])
    test[feature] = encoders[feature].transform(test[feature])


def analyze_feature_importance(train_df, target='Crop_Yield (kg/ha)'):
    """
    Analyze and visualize feature importance using Random Forest
    """
    X = train_df.drop(columns=[target, 'id'])
    y = train_df[target]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return importance, rf_model

def plot_feature_importance(importance_df):
    """
    Create visualizations for feature importance
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance for Crop Yield Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    return plt

def create_correlation_heatmap(train_df):
    """
    Create a correlation heatmap for numerical features
    """
    plt.figure(figsize=(10, 8))
    correlation = train_df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.show()

    return plt


importance_df, rf_model = analyze_feature_importance(train)
plot_feature_importance(importance_df)
create_correlation_heatmap(train)


print("\nFeature Importance Rankings:")
print(importance_df)


### using the concept of cyclic features to handle time based data i.e. Year, and then polynomial features

def create_cyclical_features(df):
    df = df.copy()
    def normalize_year(col):
        return (col - col.min()) / (col.max() - col.min())
    norm_year = normalize_year(df['Year'])
    df['year_cycle_sin'] = np.sin(2 * np.pi * norm_year)
    df['year_cycle_cos'] = np.cos(2 * np.pi * norm_year)
    df['year_cycle_sin2'] = np.sin(4 * np.pi * norm_year)
    df['year_cycle_sin_half'] = np.sin(np.pi * norm_year)
    df['year_cycle_combined'] = df['year_cycle_sin'] * df['year_cycle_cos']
    df['year_cycle_square'] = np.sign(np.sin(2 * np.pi * norm_year))
    return df

train = create_cyclical_features(train)
test = create_cyclical_features(test)

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
feature_cols = [
    'Year','Irrigation_Area','Rainfall','Crop_Type','Soil_Type',
    'year_cycle_sin','year_cycle_cos','year_cycle_sin2','year_cycle_sin_half',
    'year_cycle_combined','year_cycle_square'
]
X_poly = poly.fit_transform(train[feature_cols])
y = train['Crop_Yield (kg/ha)']
X_test_poly = poly.transform(test[feature_cols])


def objective(trial):

    n_estimators = trial.suggest_int('n_estimators', 50, 150, step=25)  # Reduced range, increased step
    max_depth = trial.suggest_int('max_depth', 5, 15, step=2)  # Reduced range, increased step
    min_samples_split = trial.suggest_int('min_samples_split', 2, 8, step=2)  # Reduced range
    max_features = trial.suggest_float('max_features', 0.4, 0.9, step=0.2)  # Reduced range
    bag_n_estimators = trial.suggest_int('bag_n_estimators', 20, 60, step=20)  # Reduced range

    base_model = ExtraTreesRegressor(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        n_jobs=-1
    )

    model = BaggingRegressor(
        estimator=base_model,
        n_estimators=bag_n_estimators,
        random_state=42,
        n_jobs=-1 # Use all available cores
    )

    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_idx, val_idx in k_fold.split(X_poly):
        X_tr, X_val = X_poly[train_idx], X_poly[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, val_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15, show_progress_bar=True)
best_params = study.best_params


final_base = ExtraTreesRegressor(
    random_state=42,
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    max_features=best_params['max_features']
)
final_model = BaggingRegressor(
    estimator=final_base,
    n_estimators=best_params['bag_n_estimators'],
    random_state=42
)


loo = LeaveOneOut()
final_mse_scores = []
test_preds = []
for fold_idx, (train_idx, val_idx) in enumerate(loo.split(X_poly), 1):
    X_tr, X_val = X_poly[train_idx], X_poly[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    final_model.fit(X_tr, y_tr)
    val_pred = final_model.predict(X_val)
    fold_mse = mean_squared_error(y_val, val_pred)
    final_mse_scores.append(fold_mse)
    test_pred = final_model.predict(X_test_poly)
    test_preds.append(test_pred)

avg_mse = np.mean(final_mse_scores)
std_mse = np.std(final_mse_scores)
final_test_prediction = np.mean(test_preds, axis=0)


print(f"Best Params: {best_params}")
print(f"LOO Average MSE: {avg_mse:.4f}")
print(f"LOO Std MSE: {std_mse:.4f}")


sub['Target'] = final_test_prediction
sub


sub.to_csv('ss.csv', index=False)

