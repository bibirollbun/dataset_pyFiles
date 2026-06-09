import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


test_ids = test_df['id']
df.drop('id', axis=1, inplace=True)


binary_cols = ['Stage_fear', 'Drained_after_socializing']
for col in binary_cols:
    df[col] = df[col].replace({'Yes': 1, 'No': 0})
    test_df[col] = test_df[col].replace({'Yes': 1, 'No': 0})

df['Personality'] = df['Personality'].replace({
    'Extrovert': 0,
    'Introvert': 1
})


# Identify numerical features for imputation and scaling
numerical_features = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]


# Separate features (X) and target (y) from the training data
X = df.drop('Personality', axis=1)
y = df['Personality']

outlier_mask = pd.Series(False, index=X.index)
for col in numerical_features: # Only check numerical columns for outliers
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_mask = outlier_mask | (X[col] < lower) | (X[col] > upper)

n_outliers = outlier_mask.sum()
print('Outliers detected (in training data):', n_outliers)

# Remove outliers from X and y
X_clean = X[~outlier_mask].reset_index(drop=True)
y_clean = y[~outlier_mask].reset_index(drop=True)
print('Remaining rows after outlier removal (training data):', X_clean.shape[0])
print()


mean_impute_cols = ['Time_spent_Alone', 'Friends_circle_size', 'Post_frequency']
median_impute_cols = ['Social_event_attendance', 'Going_outside']
# 'Stage_fear' and 'Drained_after_socializing' are now 0/1, so mode imputation is appropriate for missing values
mode_impute_cols = ['Stage_fear', 'Drained_after_socializing']

imputation_transformer = ColumnTransformer(
    transformers=[
        ('mean_imputer', SimpleImputer(strategy='mean'), mean_impute_cols),
        ('median_imputer', SimpleImputer(strategy='median'), median_impute_cols),
        ('mode_imputer', SimpleImputer(strategy='most_frequent'), mode_impute_cols)
    ],
    remainder='passthrough' # This is crucial to keep other columns (if any) and maintain order
)


preprocessor_pipeline = Pipeline(steps=[
    ('imputer', imputation_transformer),
    ('scaler', StandardScaler())       
])

X_processed = preprocessor_pipeline.fit_transform(X_clean)

test_df_for_processing = test_df.drop('id', axis=1)
test_processed = preprocessor_pipeline.transform(test_df_for_processing)


model = xgb.XGBClassifier(
    objective='binary:logistic', 
    eval_metric='logloss',       
    use_label_encoder=False,     
    n_estimators=500,            
    learning_rate=0.05,          
    max_depth=5,                 
    subsample=0.7,           
    colsample_bytree=0.7,        
    random_state=42,             
    n_jobs=-1                    
)

# Train
model.fit(X_processed, y_clean)
predictions_numeric = model.predict(test_processed)


predictions_labels = pd.Series(predictions_numeric).replace({
    0: 'Extrovert',
    1: 'Introvert'
})


submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': predictions_labels
})

submission_df.to_csv('submission.csv', index=False)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_processed, y_clean, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation Accuracy Scores: {scores}")
print(f"Mean CV Accuracy: {scores.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {scores.std():.4f}")




