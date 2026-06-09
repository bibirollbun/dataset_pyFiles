import pandas as pd
import gc
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
from scipy.stats import pearsonr
from IPython.display import display
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor

warnings.filterwarnings('ignore')


df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')


df.head(5)


train_df = df.iloc[:500000]
test_df = df.iloc[500000:]
train_df = train_df.astype('float32')
test_df = test_df.astype('float32')


train_df.replace([np.inf, -np.inf], 0, inplace=True)

x_train = train_df.drop(columns = ['label'])
y_train = train_df['label']
x_test = test_df.drop(columns = ['label'])
y_test = test_df['label']


def find_cols_with_zero_variance(x_train):
    zero_var_cols = x_train.columns[x_train.nunique() <= 1].tolist()
    print("Columns with 0 variance:", zero_var_cols)
    print(f"{len(zero_var_cols)} col skipped")
    return zero_var_cols
    
def find_cols_with_low_correlation(x_train, y_train, threshold=0.01):
    correlations = x_train.corrwith(y_train)
    low_corr_cols = correlations[correlations.abs() < threshold].index.tolist()
    print("Columns with low correlation:", low_corr_cols)
    print(f"{len(low_corr_cols)} features dropped with correlation below {threshold}.")
    return low_corr_cols

def find_top_features_with_xgboost(x_train, x_test, y_train, top_n=50):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=0
    )
    model.fit(x_train, y_train)

    importances = model.feature_importances_
    feature_names = x_train.columns

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    top_features = importance_df.head(top_n)['feature'].tolist()
    dropped_features = importance_df.tail(len(importance_df) - top_n)['feature'].tolist()

    print(f"Selected top {top_n} features based on XGBoost importance.")
    print(f"Dropped features: {dropped_features}")

    return top_features

def select_top_features_with_xgboost(data, top_features):
    return data[top_features]

def find_highly_correlated_features(x_train, threshold=0.98):
    corr_matrix = x_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropping {len(high_corr_cols)} highly correlated features (threshold > {threshold})")
    print(f"Columns dropped: {high_corr_cols}")
    return high_corr_cols

def drop_features(data, to_drop):
    return data.drop(columns=to_drop)

def process(x_train, x_test, y_train):
    zero_var_cols = find_cols_with_zero_variance(x_train)
    x_train = drop_features(x_train, zero_var_cols)
    x_test = drop_features(x_test, zero_var_cols)
    
    low_corr_cols = find_cols_with_low_correlation(x_train, y_train, threshold=0.01) 
    x_train = drop_features(x_train, low_corr_cols)
    x_test = drop_features(x_test, low_corr_cols)
    
    selected_features = find_top_features_with_xgboost(x_train, x_test, y_train, top_n=300)
    x_train = select_top_features_with_xgboost(x_train, selected_features)
    x_test = select_top_features_with_xgboost(x_test, selected_features)

    high_corr_cols = find_highly_correlated_features(x_train, 0.98)
    x_train = drop_features(x_train, high_corr_cols)
    x_test = drop_features(x_test, high_corr_cols)
    
    return x_train, x_test, y_train

def train_fit_xgboost(x_train , y_train):
    XGBR = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.1,
            random_state=42,
    )
    XGBR.fit(x_train , y_train)
    return XGBR

def predict_res(model, x_test):
    y_pred = model.predict(x_test)
    return y_pred

"""xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.1,
    #max_depth=6,
    random_state=42,
)

# LightGBM
lgb_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# CatBoost
cat_model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    verbose=0,
    random_state=42
)

# Ensemble: Stacking
stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator=Ridge(),
    passthrough=True,
    cv=3
)

print("Training Stacking Model...")
stacking_model.fit(x_train, y_train)"""


def find_errors_and_plot(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    corr_coef, p_value = pearsonr(y_test, y_pred)
    
    results = pd.DataFrame({
        'Model': [f'{model_name}'],
        'MAE': [mae],
        'MSE': [mse],
        'RMSE': [rmse],
        'R2': [r2],
        'Pearson Correlation Coefficient':[corr_coef],
        'P value':[p_value],
    })
    display(results)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_test), size=100, replace=False)
    
    y_test_sample = y_test.iloc[sample_indices].reset_index(drop=True)
    y_pred_sample = pd.Series(y_pred[sample_indices])
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_sample, label='Actual', marker='o')
    plt.plot(y_pred_sample, label='Predicted', marker='x')
    plt.title("Actual vs Predicted (100 Random Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


x_train, x_test, y_train = process(x_train, x_test, y_train)


def plot_feature_vs_label(x, y, features):
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=x[feature], y=y, s=10, alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('label')
        plt.title(f"{feature} vs label")
        plt.tight_layout()
        plt.show()


#correlation_series = x_train.corrwith(y_train).abs().sort_values(ascending=False)
#top_corr_features = correlation_series.head(300).index.tolist()
#plot_feature_vs_label(x_train, y_train, features=top_corr_features)


print(len(x_train.columns))

XGBR = train_fit_xgboost(x_train , y_train)

#y_pred = predict_res(stacking_model, x_test)
y_pred = predict_res(XGBR, x_test)

find_errors_and_plot(y_test, y_pred, 'XGB')


!pip install shap
import shap


def shap_analysis(x_train, model, max_display=20, sample_size=1000):
    sampled_x = x_train.sample(sample_size, random_state=42)
    explainer = shap.Explainer(model)
    shap_values = explainer(sampled_x)

    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, sampled_x, plot_type="bar", max_display=max_display)
    shap.summary_plot(shap_values, sampled_x, max_display=max_display)
    shap.plots.waterfall(shap_values[100])

shap_analysis(x_train, XGBR, max_display=50, sample_size=1000)


del x_train
del y_train
del x_test
del y_test
del df
gc.collect()
%system free -m


## SUBMISSON
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')

train_df = train_df.astype('float32')
test_df = test_df.astype('float32')

train_df.replace([np.inf, -np.inf], 0, inplace=True)
test_df.replace([np.inf, -np.inf], 0, inplace=True)

x_train = train_df.drop(columns = ['label'])
y_train = train_df['label']
x_test = test_df.drop(columns = ['label'])
y_test = test_df['label']

del train_df
del test_df
gc.collect()

x_train, x_test, y_train = process(x_train, x_test, y_train)

# Train
XGBR = train_fit_xgboost(x_train , y_train)

# Test
#y_pred = predict_res(stacking_model, x_test)
y_pred = predict_res(XGBR, x_test)
y_pred


sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sample_submission['prediction'] = y_pred
sample_submission.to_csv('submission.csv',index = False )


sample_submission

