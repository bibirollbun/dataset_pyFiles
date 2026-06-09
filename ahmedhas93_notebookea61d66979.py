# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder





df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")


df.drop(columns=['id'] , inplace=True)
# remove = ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id"]
# df = df.drop(columns=remove)


y = df['accident_risk']
X = df.drop(columns = ['accident_risk'])


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import warnings

def plot_complete_distribution(series, feature_name):
    """Create comprehensive distribution plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Complete Distribution Analysis: {feature_name}', 
                 fontsize=16, y=1.02)
    
    # Histogram with KDE
    axes[0, 0].hist(series, bins=30, alpha=0.7, density=True, 
                   color='skyblue', edgecolor='black')
    sns.kdeplot(series, ax=axes[0, 0], color='red', linewidth=2)
    axes[0, 0].set_title(f'Histogram with KDE\n{feature_name}')
    axes[0, 0].set_ylabel('Density')
    
    # Box plot
    axes[0, 1].boxplot(series, vert=True)
    axes[0, 1].set_title(f'Box Plot\n{feature_name}')
    axes[0, 1].set_ylabel('Values')
    
    # Q-Q plot
    stats.probplot(series, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title(f'Q-Q Plot\n{feature_name}')
    
    # Violin plot
    sns.violinplot(y=series, ax=axes[1, 0])
    axes[1, 0].set_title(f'Violin Plot\n{feature_name}')
    axes[1, 0].set_ylabel('Values')
    
    # ECDF plot
    x = np.sort(series)
    y = np.arange(1, len(x)+1) / len(x)
    axes[1, 1].plot(x, y, marker='.', linestyle='none')
    axes[1, 1].set_title(f'ECDF Plot\n{feature_name}')
    axes[1, 1].set_xlabel('Values')
    axes[1, 1].set_ylabel('ECDF')
    
    # Distribution comparison with normal
    axes[1, 2].hist(series, bins=30, alpha=0.7, density=True, 
                   color='skyblue', edgecolor='black', 
                   label='Data')
    # Plot normal distribution for comparison
    x_norm = np.linspace(series.min(), series.max(), 100)
    y_norm = stats.norm.pdf(x_norm, series.mean(), series.std())
    axes[1, 2].plot(x_norm, y_norm, 'r-', linewidth=2, 
                   label='Normal dist')
    axes[1, 2].set_title(f'Comparison with Normal Distribution\n{feature_name}')
    axes[1, 2].set_xlabel('Values')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()

# Plot distributions for all features
plot_complete_distribution(df['accident_risk'].clip(upper=0.8), 'accident_risk')


le = LabelEncoder()





X_train , X_test , y_train , y_test = train_test_split(X , y)


# for i in X.select_dtypes(exclude=['int' , 'float']).columns :
#     le.fit(X_train[i])
#     X_test[i] = le.transform(X_test[i])
#     X_train[i] = le.transform(X_train[i])


X


import optuna
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ModelSelectionPipeline:
    def __init__(self, X_train, X_test, y_train, y_test, categorical_features):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        self.categorical_features = categorical_features
        self.study = None
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data_for_xgboost(self, X_train, X_test=None):
        """Prepare data ensuring train/test have same columns"""
        if X_test is None:
            # Training only
            return pd.get_dummies(X_train, columns=self.categorical_features, drop_first=True)
        else:
            # Ensure same columns for train and test
            X_train_prep = pd.get_dummies(X_train, columns=self.categorical_features, drop_first=True)
            X_test_prep = pd.get_dummies(X_test, columns=self.categorical_features, drop_first=True)
            
            # Align columns
            X_test_prep = X_test_prep.reindex(columns=X_train_prep.columns, fill_value=0)
            return X_train_prep, X_test_prep
    
    def prepare_data_for_lightgbm(self, X):
        """Prepare data for LightGBM (Categorical Data Types)"""
        X_prepared = X.copy()
        
        # Convert to category dtype for LightGBM
        for col in self.categorical_features:
            if col in X_prepared.columns:
                X_prepared[col] = X_prepared[col].astype('category')
                
        return X_prepared
    
    def objective(self, trial):
        """Optuna objective function for model selection"""
        
        model_name = trial.suggest_categorical('model', [ 'lightgbm' , 'xgboost' ])
        
        if model_name == 'xgboost':
            # XGBoost parameters
            params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 700, 800),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.4, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0, 1),
                'random_state': 42,
                'verbosity': 0
            }
            
            # Prepare data for XGBoost
            X_train_prepared = self.prepare_data_for_xgboost(self.X_train)
            model = XGBRegressor(**params)
            
        else:  # lightgbm
            # LightGBM parameters
            params = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 700, 800),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 18),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 12, 150),
                'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0, 1),
                'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
                'random_state': 42,
                'verbosity': -1
            }
            
            # Prepare data for LightGBM
            X_train_prepared = self.prepare_data_for_lightgbm(self.X_train)
            model = LGBMRegressor(**params)
        
        # Calculate RMSE using cross-validation
        try:
            mse_scores = -cross_val_score(
                model, X_train_prepared, self.y_train,
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                error_score='raise'
            )
            rmse_scores = np.sqrt(mse_scores)
            rmse = rmse_scores.mean()
            
            # Store additional information
            trial.set_user_attr('model_name', model_name)
            trial.set_user_attr('rmse_std', rmse_scores.std())
            trial.set_user_attr('cv_scores', rmse_scores.tolist())
            
        except Exception as e:
            # Penalize failed trials
            rmse = 1e10
            trial.set_user_attr('failed', True)
        
        return rmse
    
    def optimize(self, n_trials=100):
        """Run the optimization process"""
        print("Starting model selection optimization...")
        
        self.study = optuna.create_study(
            study_name="xgboost_vs_lightgbm",
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        self.study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nOptimization completed!")
        print(f"Best model: {self.study.best_trial.user_attrs['model_name']}")
        print(f"Best RMSE: {self.study.best_value:.4f}")
        
        return self.study
    
    def train_best_model(self):
        """Train the best model found during optimization"""
        if self.study is None:
            raise ValueError("Run optimize() first!")
        
        best_params = self.study.best_params
        self.best_model_name = self.study.best_trial.user_attrs['model_name']
        
        print(f"\nTraining best model: {self.best_model_name}")
        
        if self.best_model_name == 'xgboost':
            # Filter and prepare XGBoost parameters
            xgb_params = {k.replace('xgb_', ''): v for k, v in best_params.items() 
                         if k.startswith('xgb_')}
            xgb_params['random_state'] = 42
            xgb_params['verbosity'] = 0
            
            self.best_model = XGBRegressor(**xgb_params)
            
            # Prepare data for XGBoost
            X_train_prepared , X_test_prepared=prepare_data_for_xgboost(self.X_train,self.X_test)
            
        else:  # lightgbm
            # Filter and prepare LightGBM parameters
            lgb_params = {k.replace('lgb_', ''): v for k, v in best_params.items() 
                         if k.startswith('lgb_')}
            lgb_params['random_state'] = 42
            lgb_params['verbosity'] = -1
            
            self.best_model = LGBMRegressor(**lgb_params)
            
            # Prepare data for LightGBM
            X_train_prepared = self.prepare_data_for_lightgbm(self.X_train)
            X_test_prepared = self.prepare_data_for_lightgbm(self.X_test)
        
        # Train the model
        self.best_model.fit(X_train_prepared, self.y_train)
        
        return X_test_prepared
    
    def evaluate(self):
        """Evaluate the best model on test set"""
        if self.best_model is None:
            raise ValueError("Train the best model first!")
        
        # Prepare test data based on model type
        if self.best_model_name == 'xgboost':
            X_train_prepared , X_test_prepared=prepare_data_for_xgboost(self.X_train,self.X_test)
        else:
            X_train_prepared = self.prepare_data_for_lightgbm(self.X_train)
            X_test_prepared = self.prepare_data_for_lightgbm(self.X_test)
        
            

        
        # Predictions
        y_pred = self.best_model.predict(X_test_prepared)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"FINAL EVALUATION - {self.best_model_name.upper()}")
        print(f"{'='*50}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MSE:  {mse:.4f}")
        print(f"Test R²:   {r2:.4f}")
        
        # Compare with baseline (mean predictor)
        baseline_rmse = np.sqrt(mean_squared_error(self.y_test, [self.y_test.mean()] * len(self.y_test)))
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        print(f"Improvement over baseline: {improvement:.1f}%")
        
        return {
            'model': self.best_model_name,
            'rmse': rmse,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred
        }
    
    def analyze_optimization(self):
        """Analyze the optimization results"""
        if self.study is None:
            raise ValueError("Run optimize() first!")
        
        print(f"\n{'='*50}")
        print(f"OPTIMIZATION ANALYSIS")
        print(f"{'='*50}")
        
        # Model distribution
        model_counts = {'xgboost': 0, 'lightgbm': 0}
        model_performance = {'xgboost': [], 'lightgbm': []}
        
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                model_name = trial.user_attrs.get('model_name')
                if model_name in model_counts:
                    model_counts[model_name] += 1
                    model_performance[model_name].append(trial.value)
        
        print(f"\nModel Distribution:")
        for model, count in model_counts.items():
            if count > 0:
                avg_rmse = np.mean(model_performance[model])
                std_rmse = np.std(model_performance[model])
                print(f"  {model:10s}: {count:3d} trials | Avg RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
        
        # Best parameters
        print(f"\nBest Parameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance from the best model"""
        if self.best_model is None:
            raise ValueError("Train the best model first!")
        
        # Prepare feature names based on model type
        if self.best_model_name == 'xgboost':
            X_prepared = self.prepare_data_for_xgboost(self.X_train)
            feature_names = X_prepared.columns.tolist()
        else:
            feature_names = self.X_train.columns.tolist()
        
        # Get importance scores
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            print(f"\nTop {top_n} Feature Importances:")
            for i, row in importance_df.iterrows():
                print(f"  {row['feature']:20s}: {row['importance']:.4f}")
            
            return importance_df
        else:
            print("Feature importance not available for this model")
            return None

# Example usage with your data
def run_complete_pipeline(X_train, X_test, y_train, y_test, categorical_features, n_trials=100):
    """Run the complete model selection pipeline"""
    
    # Initialize pipeline
    pipeline = ModelSelectionPipeline(
        X_train, X_test, y_train, y_test, categorical_features
    )
    
    # Step 1: Optimize model selection and hyperparameters
    study = pipeline.optimize(n_trials=n_trials)
    
    # Step 2: Train the best model
    pipeline.train_best_model()
    
    # Step 3: Evaluate on test set
    results = pipeline.evaluate()
    
    # Step 4: Analyze optimization results
    pipeline.analyze_optimization()
    
    # Step 5: Get feature importance
    importance_df = pipeline.get_feature_importance()
    
    return pipeline, results, importance_df


 categorical_features=X.select_dtypes(exclude=['int' , 'float']).columns


# Quick usage with your data:
pipeline, results, importance_df = run_complete_pipeline(
    X_train, X_test, (y_train), (y_test), 
    categorical_features=categorical_features, 
    n_trials=30
)


pipeline.best_model


def prepare_data_for_lightgbm( X ,categorical_features ):
    """Prepare data for LightGBM (Categorical Data Types)"""
    X_prepared = X.copy()
    
    # Convert to category dtype for LightGBM
    for col in categorical_features:
        if col in X_prepared.columns:
            X_prepared[col] = X_prepared[col].astype('category')
            
    return X_prepared



categorical_features


X_prepared =  prepare_data_for_lightgbm( X , categorical_features)


best_model=pipeline.best_model

# best_model=LGBMRegressor(colsample_bytree=0.7635792237136688,
#               learning_rate=0.029953341889868593, max_depth=16,
#               min_child_samples=11, n_estimators=4606, num_leaves=120,
#               random_state=42, reg_alpha=0.6933834712965824,
#               reg_lambda=0.23265021630938104, subsample=0.7675550493966354,
#               verbosity=-1)


# y_prepared = np.log1p(y)


best_model.fit(X_prepared,y)


X_prepared.columns.tolist()


from sklearn.metrics import mean_squared_error



# np.sqrt(mean_squared_error(y_test,np.expm1(results['predictions'])))


df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


df_test


submission_indes = df_test['id']
df_test.drop(columns=['id'] , inplace=True)


df_test


X_test_prepared = prepare_data_for_lightgbm( df_test , categorical_features)


np.log1p(y)


# pd.Series(np.expm1(best_model.predict(X_test_prepared)) , index=submission_indes).to_csv("submission.csv")


pd.Series(best_model.predict(X_test_prepared) , index=submission_indes).to_csv("submission.csv")

