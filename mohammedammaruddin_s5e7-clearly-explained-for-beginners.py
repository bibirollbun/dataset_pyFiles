import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


# dir(train)


def dataset_insights(**datasets):
    import pandas as pd
    import numpy as np

    for name, df in datasets.items():
        print(f"\n{'='*40}")
        print(f"ğŸ“˜ Dataset: {name}")
        print(f"{'='*40}\n")
        
        print("ğŸ”¹ FIRST 5 ROWS:")
        print(df.head(), "\n")
        
        print("ğŸ”¹ SHAPE:")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
        
        print("ğŸ”¹ DATA TYPES:")
        print(df.dtypes, "\n")
        
        print("ğŸ”¹ NULL VALUES PER COLUMN:")
        print(df.isnull().sum(), "\n")
        
        print("ğŸ”¹ DUPLICATE ROWS:")
        print(df.duplicated().sum(), "\n")
        
        print("ğŸ”¹ UNIQUE VALUES PER COLUMN:")
        print(df.nunique(), "\n")
        
        print("ğŸ”¹ SUMMARY STATISTICS (Numerical Features):")
        print(df.describe(), "\n")
        
        print("ğŸ”¹ VALUE COUNTS FOR CATEGORICAL FEATURES:")
        for col in df.select_dtypes(include='object'):
            print(f"\nColumn: {col}")
            print(df[col].value_counts().head(5))  # Show top 5 for brevity
        
        print("\nğŸ”¹ CORRELATION MATRIX (Top 5 Pairs):")
        corr_pairs = df.corr(numeric_only=True).abs().unstack().sort_values(ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1].drop_duplicates().head(5)
        print(corr_pairs, "\n")
        
        print("ğŸ”¹ SKEWNESS (Numerical Features):")
        print(df.skew(numeric_only=True).sort_values(ascending=False), "\n")

        numeric_df = df.select_dtypes(include=[np.number])

        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        # Broadcast using column-wise comparison
        outlier_mask = (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
        outliers = outlier_mask.sum()

        print("ğŸ”¹ OUTLIERS DETECTED (IQR Method):")
        print(outliers[outliers > 0], "\n")




dataset_insights(train=train)




def plot_dataset_insights(df, max_categories=10):
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(include=['object', 'category'])

    plt.style.use("seaborn-v0_8-darkgrid")
    
    # 1. VALUE COUNTS FOR CATEGORICAL FEATURES
    for col in categorical_df.columns:
        plt.figure(figsize=(8, 4))
        value_counts = df[col].value_counts().head(max_categories)
        sns.barplot(x=value_counts.values, y=value_counts.index, palette='Set2')
        plt.title(f"Top {max_categories} Value Counts for '{col}'")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

    # 2. CORRELATION MATRIX
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    # 3. SKEWNESS OF NUMERICAL FEATURES
    skewness = numeric_df.skew().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=skewness.index, y=skewness.values, palette='viridis')
    plt.xticks(rotation=45)
    plt.title("Skewness of Numeric Features")
    plt.ylabel("Skewness")
    plt.tight_layout()
    plt.show()

    # 4. OUTLIERS USING BOX PLOTS (IQR Method)
    for col in numeric_df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(x=df[col], color='salmon')
        plt.title(f"Box Plot of '{col}' (Outlier Detection)")
        plt.tight_layout()
        plt.show()



plot_dataset_insights(train)


train['Time_spent_Alone'].fillna(train['Time_spent_Alone'].mean(), inplace=True)


train = train[train['Time_spent_Alone'] <= 6.0]


count = (train['Time_spent_Alone'] > 9.0).sum()
print("Rows where Time_spent_Alone > 8:", count)


plt.figure(figsize=(7, 4))
sns.boxplot(x=train['Time_spent_Alone'], color='salmon')
plt.title(f"Box Plot of  (Outlier Detection)")
plt.tight_layout()
plt.show()


def select_important_features(X, y, model):
    selector = SelectFromModel(estimator=model)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    return X[selected_features], selected_features



def get_models():
    return {
        'xgb': XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', 
                             use_label_encoder=False, eval_metric='logloss'),
        'lgbm': LGBMClassifier(device='gpu'),
        'catboost': CatBoostClassifier(task_type='GPU', verbose=0),

        # New CPU models (no GPU args)
        # 'svm': SVC(probability=True),  # Add tuned params later if needed
        'rf': RandomForestClassifier(),
        'logreg': LogisticRegression(max_iter=500)
    }



# def optimize_model(trial, X, y, model_name):
#     if model_name == 'xgb':
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#             'max_depth': trial.suggest_int('max_depth', 3, 10),
#             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#             'gamma': trial.suggest_float('gamma', 0, 0.4),
#             'tree_method': 'gpu_hist',
#             'predictor': 'gpu_predictor',
#             'use_label_encoder': False,
#             'eval_metric': 'logloss'
#         }
#         model = XGBClassifier(**params)

#     elif model_name == 'lgbm':
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#             'num_leaves': trial.suggest_int('num_leaves', 20, 80),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#             'device': 'gpu'
#         }
#         model = LGBMClassifier(**params)

#     elif model_name == 'catboost':
#         params = {
#             'iterations': trial.suggest_int('iterations', 100, 500),
#             'depth': trial.suggest_int('depth', 4, 10),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#             'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 5.0),
#             'task_type': 'GPU',
#             'verbose': 0
#         }
#         model = CatBoostClassifier(**params)

#     score = cross_val_score(model, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=5)).mean()
#     return score




def optimize_model(trial, X, y, model_name):
    if model_name == 'xgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.4),
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params)

    elif model_name == 'lgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'device': 'gpu'
        }
        model = LGBMClassifier(**params)

    elif model_name == 'catboost':
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 5.0),
            'task_type': 'GPU',
            'verbose': 0
        }
        model = CatBoostClassifier(**params)

    # elif model_name == 'svm':
    #     params = {
    #         'C': trial.suggest_float('C', 0.1, 10),
    #         'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
    #         'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
    #     }
    #     model = SVC(**params, probability=True)  # CPU only

    elif model_name == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        model = RandomForestClassifier(**params)  # CPU only

    elif model_name == 'logreg':
        params = {
            'C': trial.suggest_float('C', 0.01, 10),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
            'penalty': trial.suggest_categorical('penalty', ['l2'])
        }
        model = LogisticRegression(**params, max_iter=500)  # CPU only

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Evaluation
    score = cross_val_score(model, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=5)).mean()
    return score



def run_pipeline(X, y):
    models = get_models()
    final_results = {}

    for name, model in models.items():
        print(f"\nğŸ”� Optimizing and selecting features using {name.upper()}")

        # Feature Selection
        X_sel, selected_features = select_important_features(X, y, model)
        print(f"âœ… Selected top {len(selected_features)} features using {name.upper()}")

        # Optuna optimization
        def objective(trial):
            return optimize_model(trial, X_sel, y, name)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)

        best_score = study.best_value
        best_params = study.best_params

        final_results[name] = {
            'score': best_score,
            'params': best_params,
            'features': selected_features
        }

    return final_results



# def preprocess_data(df, is_train=True):
#     df = df.copy()
    
#     # Convert Yes/No to 1/0
#     yes_no_cols = ['Stage_fear', 'Drained_after_socializing']
#     for col in yes_no_cols:
#         if col in df.columns:
#             df[col] = df[col].map({'Yes': 1, 'No': 0})
    
#     # Fill missing values with mean for numeric columns
#     for col in df.select_dtypes(include='number'):
#         df[col] = df[col].fillna(df[col].mean())

#     # Convert Personality only in training data
#     if is_train and 'Personality' in df.columns:
#         df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})
    
#     return df
# train = preprocess_data(train)
# test = preprocess_data(test)


def preprocess_data(df, is_train=True):
    df = df.copy()

    # Map Yes/No to 1/0 (safe conversion)
    yes_no_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})

    # Handle Personality label (only for training)
    if is_train and 'Personality' in df.columns:
        df['Personality'] = df['Personality'].replace({'Introvert': 0, 'Extrovert': 1, 
                                                        'introvert': 0, 'extrovert': 1})

    # Fill missing numeric values
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].mean())

    return df

train = preprocess_data(train, is_train=True)
test = preprocess_data(test, is_train=False)


train.dtypes


def engineer_features(df):
    df = df.copy()
    # Behavioral ratios
    df['Social_Energy_Ratio'] = df['Social_event_attendance'] / (df['Time_spent_Alone'] + 1e-5)
    df['Interaction_Recovery'] = df['Drained_after_socializing'] * df['Time_spent_Alone']
    # Social activity index
    df['Social_Activity_Index'] = (0.3*df['Social_event_attendance'] + 
                                   0.2*df['Going_outside'] + 
                                   0.5*df['Post_frequency'])
    # Friend density
    df['Friend_Density'] = df['Friends_circle_size'] / (df['Social_event_attendance'] + 1)
    return df

train = engineer_features(train)
test = engineer_features(test)


train.shape


# train.head()


# Assuming you have train.csv and test.csv already loaded
train = preprocess_data(train, is_train=True)
test = preprocess_data(test, is_train=False)

# Separate features and target
X = train.drop(columns=['Personality'])
y = train['Personality']

# Run your ML pipeline
results = run_pipeline(X, y)



# Show summary
for model, result in results.items():
    print(f"\nModel: {model.upper()}")
    print("Score:", result['score'])
    print("Top Features:", result['features'].tolist())
    print("Best Params:", result['params'])


from sklearn.base import clone

# Step 1: Get the best model by score
best_model_name = max(results, key=lambda k: results[k]['score'])
print(f"\nğŸ”¥ Best model selected: {best_model_name.upper()}")

# Step 2: Load top features and best params
top_features = results[best_model_name]['features']
best_params = results[best_model_name]['params']

# Step 3: Clone and configure model
model = get_models()[best_model_name]
best_model = clone(model).set_params(**best_params)

# Step 4: Fit on full training data using top features
best_model.fit(X[top_features], y)

# Step 5: Prepare test data
X_test = test[top_features]

# Step 6: Predict (0 = Introvert, 1 = Extrovert)
y_pred = best_model.predict(X_test)

# Step 7: Map predictions back to labels
label_map = {0: 'Introvert', 1: 'Extrovert'}
predicted_labels = pd.Series(y_pred).map(label_map)

# Step 8: Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': predicted_labels
})

# Step 9: Save CSV for Kaggle submission
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file 'submission.csv' is ready for upload to Kaggle!")



print(submission.head())
print(submission['Personality'].value_counts())
print(submission.shape)





