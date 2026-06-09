import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

columns = train.columns.tolist()
train.head()


train_id = train.drop('id', axis=1)
test_id = test.drop('id', axis=1)


print(f"Is na in train {train.shape}: \n ============ \n {train.isna().sum().sum()}")
print(f"Is na in test {test.shape}: \n ============ \n {test.isna().sum().sum()} \n ============ \n")
print(f"  Info data set {train.info()}")


def preparing_data(train):
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
print_count_values(train_id, columns[1:])


print_nunique_values(train_id, columns[1:])


train.hist(figsize=(20, 20), bins=50, xlabelsize=8, ylabelsize=8, grid=True, edgecolor="black")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.suptitle("Histogram of all features", fontsize=20)
plt.xlabel("Value", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


data_cat = train.select_dtypes(include="object")
data_num = train.select_dtypes(include=['int', 'float'])

data_cat


sns.heatmap(data_num.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# sns.pairplot(train, diag_kind='kde')
# plt.suptitle("Pairplot of all features", fontsize=20)
# plt.show()


plt.figure(figsize=(10, 6))
for i in data_cat.columns:
    plt.figure(figsize=(10, 6))
    data_cat[i].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f"Pie Chart of {i}")
    plt.axis('equal')
    plt.show()


# fig, axes = plt.subplots(2,3, figsize=(20, 12))
# fig.suptitle("Distribution of varianbles by Fertilizer name", fontsize=16, y=1.02)

# axes = axes.flatten()
# num_col = data_num.columns.tolist()

# for i, col in enumerate(num_col):
#     sns.boxplot(data = train, x='Fertilizer Name', y=col, ax=axes[i])
#     axes[i].set_title(f'{col} distribution by Fertilizer Name')
#     axes[i].set_xlabel('Fertilizer Name')
#     axes[i].set_ylabel(col)

# plt.tight_layout()
# plt.show()


# fertilizer_name_mean = train.groupby("Fertilizer Name").mean()['Temparature	Humidity','Moisture','Soil Type','Crop Type','Nitrogen','Potassium','Phosphorous']
# fig, ax = plt.subplots(figsize=(10, 6))
# fertilizer_name_mean.plot(kind='bar', ax=ax)
# ax.set_title("Avrage of other variable")
# ax.set_ylabel("Avarage value")
# ax.set_xlabel("fertilizer_name")
# plt.xticks(rotation=0)
# plt.show()


train.describe()


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')


# Custom transformer for label encoding
class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
    
    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            X_encoded[col] = self.encoders[col].transform(X[col])
        return X_encoded


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


class OptimizedEnsemble:
    def __init__(self, train_data, test_data, n_trials=100, cv_folds=5):
        self.train = train_data
        self.test = test_data
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        
        # Initialize label encoders
        self.fertilizer_le = LabelEncoder()
        self.target_encoded = self.fertilizer_le.fit_transform(self.train['Fertilizer Name'])
        
        # Feature columns
        self.numerical_features = ['Temparature', 'Humidity', 'Moisture', 
                                 'Nitrogen', 'Potassium', 'Phosphorous']
        self.categorical_features = ['Soil Type', 'Crop Type']
        
        # Prepare data
        self.X_train = self.train[self.numerical_features + self.categorical_features]
        self.y_train = self.target_encoded
        self.X_test = self.test[self.numerical_features + self.categorical_features]
        
        # Setup preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.numerical_features),
                ('cat', MultiLabelEncoder(), self.categorical_features)
            ]
        )
        
        # Best parameters storage
        self.best_xgb_params = None
        self.best_lgb_params = None
        self.best_models = {}
        
    def create_xgb_pipeline(self, trial):
        """Create XGBoost pipeline with Optuna trial parameters"""
        params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.3, 1.0),
            'gamma': trial.suggest_float('xgb_gamma', 0, 2.0),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
            'objective': 'multi:softprob',
            'num_class': len(np.unique(self.y_train)),
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'random_state': 42,
            'verbosity': 0
        }
        
        xgb_classifier = XGBClassifier(**params)
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', xgb_classifier)
        ])
        
        return pipeline
    
    def create_lgb_pipeline(self, trial):
        """Create LightGBM pipeline with Optuna trial parameters"""
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(self.y_train)),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 15),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 10, 300),
            'min_data_in_leaf': trial.suggest_int('lgb_min_data_in_leaf', 5, 100),
            'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.3, 1.0),
            'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lgb_lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lgb_lambda_l2', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        lgb_classifier = LGBMClassifier(**params)
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', lgb_classifier)
        ])
        
        return pipeline
    
    def objective_xgb(self, trial):
        """Objective function for XGBoost optimization"""
        pipeline = self.create_xgb_pipeline(trial)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                               cv=cv, scoring='neg_log_loss', n_jobs=-1)
        
        return -scores.mean()  # Return negative because Optuna minimizes
    
    def objective_lgb(self, trial):
        """Objective function for LightGBM optimization"""
        pipeline = self.create_lgb_pipeline(trial)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                               cv=cv, scoring='neg_log_loss', n_jobs=-1)
        
        return -scores.mean()  # Return negative because Optuna minimizes
    
    def optimize_xgb(self):
        """Optimize XGBoost hyperparameters"""
        print("Optimizing XGBoost hyperparameters...")
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(self.objective_xgb, n_trials=self.n_trials)
        
        self.best_xgb_params = study.best_params
        print(f"Best XGBoost CV Score: {study.best_value:.6f}")
        print(f"Best XGBoost Parameters: {self.best_xgb_params}")
        
        return study
    
    def optimize_lgb(self):
        """Optimize LightGBM hyperparameters"""
        print("Optimizing LightGBM hyperparameters...")
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(self.objective_lgb, n_trials=self.n_trials)
        
        self.best_lgb_params = study.best_params
        print(f"Best LightGBM CV Score: {study.best_value:.6f}")
        print(f"Best LightGBM Parameters: {self.best_lgb_params}")
        
        return study
    
    def train_final_models(self):
        """Train final models with optimized parameters"""
        if self.best_xgb_params is None or self.best_lgb_params is None:
            raise ValueError("Please run optimization first!")
        
        print("Training final optimized models...")
        
        # Create XGBoost pipeline with best parameters
        xgb_params = {k.replace('xgb_', ''): v for k, v in self.best_xgb_params.items()}
        xgb_params.update({
            'objective': 'multi:softprob',
            'num_class': len(np.unique(self.y_train)),
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'random_state': 42,
            'verbosity': 0
        })
        
        xgb_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', XGBClassifier(**xgb_params))
        ])
        
        # Create LightGBM pipeline with best parameters
        lgb_params = {k.replace('lgb_', ''): v for k, v in self.best_lgb_params.items()}
        lgb_params.update({
            'objective': 'multiclass',
            'num_class': len(np.unique(self.y_train)),
            'random_state': 42,
            'verbosity': -1,
            'force_col_wise': True
        })
        
        lgb_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', LGBMClassifier(**lgb_params))
        ])
        
        # Fit models
        xgb_pipeline.fit(self.X_train, self.y_train)
        lgb_pipeline.fit(self.X_train, self.y_train)
        
        self.best_models = {
            'xgb': xgb_pipeline,
            'lgb': lgb_pipeline
        }
        
        print("Final models trained successfully!")
    
    def predict_ensemble(self, weights=None):
        """Generate ensemble predictions"""
        if not self.best_models:
            raise ValueError("Please train final models first!")
        
        if weights is None:
            weights = [0.5, 0.5]  # Equal weights by default
        
        # Get predictions from both models
        pred_xgb = self.best_models['xgb'].predict_proba(self.X_test)
        pred_lgb = self.best_models['lgb'].predict_proba(self.X_test)
        
        # Ensemble prediction
        pred_ensemble = weights[0] * pred_xgb + weights[1] * pred_lgb
        
        # Get top 3 predictions
        top_3_indices = np.argsort(pred_ensemble, axis=1)[:, -3:][:, ::-1]
        
        # Convert back to fertilizer names
        top_3_names = []
        for row in top_3_indices:
            names = self.fertilizer_le.inverse_transform(row)
            top_3_names.append(' '.join(names))
        
        return top_3_names
    
    def create_submission(self, filename='optimized_submission.csv', weights=None):
        """Create submission file"""
        predictions = self.predict_ensemble(weights)
        
        submission = pd.DataFrame({
            "id": self.test["id"],
            "Fertilizer Name": predictions
        })
        
        submission.to_csv(filename, index=False)
        print(f"Submission saved to {filename}")
        
        return submission
    
    def run_full_optimization(self):
        """Run complete optimization pipeline"""
        print("Starting full optimization pipeline...")
        
        # Optimize both models
        xgb_study = self.optimize_xgb()
        lgb_study = self.optimize_lgb()
        
        # Train final models
        self.train_final_models()
        
        # Create submission
        submission = self.create_submission()
        
        print("Optimization pipeline completed!")
        
        return {
            'xgb_study': xgb_study,
            'lgb_study': lgb_study,
            'submission': submission
        }


# Initialize the optimizer
optimizer = OptimizedEnsemble(train, test, n_trials=50, cv_folds=5)

# Run full optimization
results = optimizer.run_full_optimization()

optimizer.optimize_xgb()
optimizer.optimize_lgb() 
optimizer.train_final_models()
submission = optimizer.create_submission()

submission_weighted = optimizer.create_submission(
    filename='weighted_submission.csv', 
    weights=[0.6, 0.4]  # 60% XGB, 40% LGB
)





