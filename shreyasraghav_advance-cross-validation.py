import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/test.csv")
sample_submission = pd.read_csv(r"/kaggle/input/playground-series-s5e1/sample_submission.csv")

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("sample_submission shape :",sample_submission.shape)


train_data.isna().sum().sort_values(ascending=False)


train_data = train_data.drop_duplicates()
train_data = train_data.dropna()
print("train_data shape :",train_data.shape)


test_data.isna().sum().sort_values(ascending=False)


import numpy as np
import pandas as pd

def date_feature_engineering(df):
    """
    Performs feature engineering on a date column in the given DataFrame.
    Adds various date-related features to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with additional date features.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Quarter'] = df['date'].dt.quarter
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.day_name()
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Cyclical Features
    df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 365.0)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    df['year_sin'] = np.sin(2 * np.pi * df['Year'] / 7.0)
    df['year_cos'] = np.cos(2 * np.pi * df['Year'] / 7.0)

    # Group Calculation
    df['Group'] = (df['Year'] - 2010) * 48 + df['Month'] * 4 + df['Day'] // 7
    
    return df



train_data = date_feature_engineering(train_data)
test_data = date_feature_engineering(test_data)


train_data.drop('date',axis=1,inplace=True)
test_data.drop('date',axis=1,inplace=True)


print("train data shape :", train_data.shape)
print("test data shape :", test_data.shape)


train_data = train_data.drop('id', axis = 1)
num_cols = list(train_data.select_dtypes(exclude=['object']).columns.difference(['num_sold']))
cat_cols = list(train_data.select_dtypes(include=['object']).columns)

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns.difference(['id']))
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# Initialize LabelEncoder
label_encoders = {col: LabelEncoder() for col in cat_cols}

# Apply LabelEncoder to each categorical column
for col in cat_cols:
    train_data[col] = label_encoders[col].fit_transform(train_data[col])
    test_data[col] = label_encoders[col].transform(test_data[col])
    


train_data = train_data.dropna()
train_data.shape


from sklearn.model_selection import train_test_split
X = train_data.drop(['num_sold'], axis=1)
y = train_data['num_sold']
test = test_data.drop(['id'],axis=1)

# Split datainto training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import optuna
import mlflow
from typing import Dict, List, Tuple, Optional
from lightgbm.callback import early_stopping as lgb_early_stopping
import plotly.express as px
import plotly.graph_objects as go

class CustomLearningRateScheduler:
    """Custom learning rate scheduler with cosine annealing and warmup"""
    def __init__(self, initial_lr: float, min_lr: float, warmup_epochs: int, total_epochs: int):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

class CustomMetrics:
    """Custom evaluation metrics for model assessment"""
    @staticmethod
    def weighted_mape(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        if weights is None:
            weights = np.ones_like(y_true)
        return np.average(np.abs((y_true - y_pred) / y_true) * 100, weights=weights)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
        errors = y_true - y_pred
        mask = np.abs(errors) <= delta
        squared_loss = 0.5 * errors ** 2
        linear_loss = delta * np.abs(errors) - 0.5 * delta ** 2
        return np.mean(np.where(mask, squared_loss, linear_loss))

class ModelLogger:
    """Logger for model parameters and performance metrics"""
    def __init__(self, log_dir: str = "model_logs"):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_history = []
        os.makedirs(self.log_dir, exist_ok=True)  # Ensure the directory exists
        
    def log_parameters(self, parameters: Dict) -> None:
        with open(f"{self.log_dir}/params_{self.timestamp}.json", "w") as f:
            json.dump(parameters, f, indent=4)
    
    def log_metrics(self, metrics: Dict) -> None:
        self.metrics_history.append(metrics)
        
    def save_metrics(self) -> None:
        pd.DataFrame(self.metrics_history).to_csv(
            f"{self.log_dir}/metrics_{self.timestamp}.csv", index=False
        )


def advanced_cross_val_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame,
    n_splits: int = 7,
    parameters: Dict = None,
    use_lr_scheduler: bool = True,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 100
) -> Tuple[float, np.ndarray, pd.DataFrame]:
    """
    Advanced cross-validation framework with learning rate scheduling and comprehensive logging
    """
    warnings.filterwarnings('ignore')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_all = []
    preds = []
    feature_importance = pd.DataFrame()
    logger = ModelLogger()
    
    # Initialize learning rate scheduler
    if use_lr_scheduler:
        lr_scheduler = CustomLearningRateScheduler(
            initial_lr=parameters['learning_rate'],
            min_lr=parameters['learning_rate'] * 0.01,
            warmup_epochs=50,
            total_epochs=parameters['n_estimators']
        )
    
    # Log parameters
    logger.log_parameters(parameters)
    
    # Store validation predictions for stacking
    oof_predictions = np.zeros(len(X))
    
    for fold, (train_index, valid_index) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")
        
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        # Calculate sample weights (optional)
        weights = 1 / np.log1p(y_train)  # Example weighting scheme
        
        # Initialize model
        model = LGBMRegressor(random_state=42, **parameters)
        
        # Training with callback for learning rate scheduling
        if use_lr_scheduler:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric='mape',
                sample_weight=weights,
                callbacks=[
                    lgb_early_stopping(early_stopping_rounds),  # Use callback for early stopping
                    lambda env: setattr(
                        env.model, 'learning_rate',
                        lr_scheduler(env.iteration)
                    )
                ]
            )
        else:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric='mape',
                sample_weight=weights,
                callbacks=[
                    lgb_early_stopping(early_stopping_rounds)  # Use callback for early stopping
                ]
        )
        # Predictions
        valid_preds = model.predict(X_valid)
        oof_predictions[valid_index] = valid_preds
        test_preds = model.predict(test)
        preds.append(test_preds)
        
        # Calculate metrics
        fold_metrics = {
            'fold': fold,
            'mape': CustomMetrics.weighted_mape(y_valid, valid_preds),
            'rmse': CustomMetrics.rmse(y_valid, valid_preds),
            'r2': r2_score(y_valid, valid_preds),
            'huber': CustomMetrics.huber_loss(y_valid, valid_preds)
        }
        metrics_all.append(fold_metrics)
        logger.log_metrics(fold_metrics)
        
        # Feature importance
        fold_importance = pd.DataFrame({
            'feature': X.columns,
            f'importance_fold_{fold}': model.feature_importances_
        })
        feature_importance = pd.concat([feature_importance, fold_importance], axis=1)
        
        print(f"Fold {fold} Metrics:")
        for metric, value in fold_metrics.items():
            if metric != 'fold':
                print(f"{metric.upper()}: {value:.4f}")
    
    # Calculate and display overall results
    mean_metrics = pd.DataFrame(metrics_all).mean()
    std_metrics = pd.DataFrame(metrics_all).std()
    print("\nOverall Cross-validation Results:")
    for metric in ['mape', 'rmse', 'r2', 'huber']:
        print(f"{metric.upper()}: {mean_metrics[metric]:.4f} (+/- {std_metrics[metric]:.4f})")
    
    # Save metrics
    logger.save_metrics()
    
    # Calculate feature importance statistics
    importance_cols = [col for col in feature_importance.columns if 'importance' in col]
    feature_importance['mean_importance'] = feature_importance[importance_cols].mean(axis=1)
    feature_importance['std_importance'] = feature_importance[importance_cols].std(axis=1)
    feature_importance = feature_importance.sort_values('mean_importance', ascending=False)
    
    # Visualizations
    plot_learning_curves(metrics_all)
    plot_feature_importance(feature_importance)
    plot_prediction_distribution(y, oof_predictions)
    
    return mean_metrics['mape'], np.mean(preds, axis=0), feature_importance

def plot_learning_curves(metrics_all: List[Dict]) -> None:
    """Plot learning curves for different metrics"""
    metrics_df = pd.DataFrame(metrics_all)
    plt.figure(figsize=(15, 5))
    for metric in ['mape', 'rmse', 'r2']:
        plt.plot(metrics_df['fold'], metrics_df[metric], label=metric.upper())
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Learning Curves Across Folds')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    """Plot feature importance with error bars"""
    plt.figure(figsize=(12, 8))
    
    # Extract feature names and importance values
    features = feature_importance['feature'].values.tolist()
    importances = feature_importance['mean_importance'].values.tolist()
    stds = feature_importance['std_importance'].values.tolist()
    
    # Create positions for bars
    positions = range(len(features))
    
    # Create horizontal bar plot with error bars
    plt.barh(positions, importances, xerr=stds, 
            height=0.7, alpha=0.8, capsize=5)
    
    # Set y-ticks
    plt.yticks(positions, features)
    
    # Customize the plot
    plt.xlabel('Mean Feature Importance')
    plt.title('Feature Importance with Standard Deviation')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()




def plot_prediction_distribution(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """Plot true vs predicted value distribution"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values Distribution')
    plt.show()

# Example usage:
parameters = {
    'n_estimators': 3946,
    'learning_rate': 0.10203344298643195,
    'max_depth': 12,
    'num_leaves': 20,
    'min_child_samples': 39,
    'subsample': 0.7786665459484634,
    'colsample_bytree': 0.7352055562065795,
    'reg_alpha': 0.2840216195298897,
    'reg_lambda': 6.583320975256993,
    'verbosity': -1
}

# Run advanced cross-validation
average_mape, lgb_preds, feature_importance = advanced_cross_val_lgbm(
    X=X,
    y=y,
    test=test,
    parameters=parameters,
    use_lr_scheduler=True
)

# Create submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'num_sold': np.expm1(lgb_preds).round()
})
submission.to_csv('submission_lgb_advanced.csv', index=False)

