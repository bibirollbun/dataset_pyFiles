pip install scikit-learn==1.3.2, xgboost==1.7.6, lightgbm==4.3.0, imbalanced-learn==0.11.0


import pandas as pd
import numpy as np
import warnings
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


df.isnull().sum()


df.describe(include='all')


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")
import joblib
from sklearn.preprocessing import LabelEncoder
# --- 1. Define columns
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
target_col = 'Personality'

# --- 2. Train-test split
X = df[numerical_cols + categorical_cols]
y = df[target_col]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

joblib.dump(label_encoder, 'label_encoder.pkl')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# --- 3. Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


# --- 4. Models + Parameter grids
model_params = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'classifier__C': [0.1, 1, 10]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, None]
        }
    },
    # 'SVM': {
    #     'model': SVC(probability=True),
    #     'params': {
    #         'classifier__C': [0.1, 1, 10],
    #         'classifier__kernel': ['linear', 'rbf']
    #     }
    # },
    # 'KNN': {
    #     'model': KNeighborsClassifier(),
    #     'params': {
    #         'classifier__n_neighbors': [3, 5, 7]
    #     }
    # },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6]
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1]
        }
    }
}

# --- 5. Loop through models and run GridSearchCV
for name, mp in model_params.items():
    print(f"\nğŸ”¹ Model: {name}")
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', mp['model'])
    ])
    
    grid = GridSearchCV(
        pipeline,
        mp['params'],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Evaluation
    print(f"Best Params: {grid.best_params_}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")


# Same model training loop as before
best_model_name = None
best_score = -np.inf
final_best_model = None

for name, mp in model_params.items():
    print(f"\nğŸ”¹ Training Model: {name}")
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', mp['model'])
    ])
    
    grid = GridSearchCV(
        pipeline,
        mp['params'],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    
    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model_name = name
        final_best_model = grid.best_estimator_

    print(f"  âœ… Best CV Score (F1-macro): {grid.best_score_:.4f}")
    print(f"  âœ… Best Params: {grid.best_params_}")

# Save the best model to disk
joblib.dump(final_best_model, 'best_model.pkl')
print(f"\nâœ… Best model '{best_model_name}' saved as 'best_model.pkl'")


# Load best model
loaded_model = joblib.load('best_model.pkl')

# Load your test.csv
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Make sure columns match (numerical_cols + categorical_cols)
X_test_csv = test_df[numerical_cols + categorical_cols]

# Predict
y_test_pred = loaded_model.predict(X_test_csv)
y_test_proba = loaded_model.predict_proba(X_test_csv)[:, 1]

# If ground truth is available in test.csv
if 'target' in test_df.columns:
    print("\nğŸ§ª Classification Report on test.csv:")
    print(classification_report(test_df['target'], y_test_pred))
    print(f"ğŸ�¯ ROC AUC Score: {roc_auc_score(test_df['target'], y_test_proba):.4f}")
else:
    print("\nğŸ”� Predictions on test.csv (first 10 rows):")
    print(pd.DataFrame({'Predicted': y_test_pred, 'Probability': y_test_proba}).head(10))


test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
X_test = test_df[numerical_cols + categorical_cols]

# Load saved model + label encoder
model = joblib.load('best_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Predict (returns 0s and 1s)
y_pred = model.predict(X_test)

# Convert 0s and 1s â†’ 'Introvert', 'Extrovert'
y_labels = label_encoder.inverse_transform(y_pred)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'predicted_personality': y_labels
})
submission.to_csv('submission.csv', index=False)

