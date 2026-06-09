import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


train_path = '/kaggle/input/playground-series-s4e11/train.csv'
test_path = '/kaggle/input/playground-series-s4e11/test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

test_ids = test_data['id'].copy()

train_data.drop(['Degree'], axis=1, inplace=True, errors='ignore')
test_data.drop(['Degree'], axis=1, inplace=True, errors='ignore')


def preprocess_data(df, is_train=True, imputers=None, encoders=None):

    df = df.copy()
    
    cols_to_drop = ['id', 'Name', 'City']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    if is_train and 'Depression' in df.columns:
        y = df['Depression'].astype(int)
        df = df.drop('Depression', axis=1)
    else:
        y = None
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if 'Profession' in categorical_cols:
        categorical_cols.remove('Profession')
        df['Profession'] = df['Profession'].fillna('Not Applicable')
    
    if is_train:
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        if categorical_cols:
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        
        imputers = {'numerical': num_imputer, 'categorical': cat_imputer}
    else:
        df[numerical_cols] = imputers['numerical'].transform(df[numerical_cols])
        if categorical_cols:
            df[categorical_cols] = imputers['categorical'].transform(df[categorical_cols])
    
    if 'Work Pressure' in df.columns and 'Financial Stress' in df.columns:
        df['Pressure_Stress_Interaction'] = df['Work Pressure'] * df['Financial Stress']
    
    if 'Work/Study Hours' in df.columns and 'Financial Stress' in df.columns:
        df['Hours_Stress_Interaction'] = df['Work/Study Hours'] * df['Financial Stress']
    
    risk_components = []
    if 'Have you ever had suicidal thoughts ?' in df.columns:
        risk_components.append((df['Have you ever had suicidal thoughts ?'] == 'Yes').astype(int))
    if 'Family History of Mental Illness' in df.columns:
        risk_components.append((df['Family History of Mental Illness'] == 'Yes').astype(int))
    if 'Financial Stress' in df.columns:
        risk_components.append(df['Financial Stress'] / 5.0)
    
    if risk_components:
        df['Mental_Health_Risk_Score'] = sum(risk_components) / len(risk_components)
    
    if 'Work/Study Hours' in df.columns:
        df['High_Work_Hours'] = (df['Work/Study Hours'] > 8).astype(int)
    
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 100], 
                                  labels=['Young', 'Adult', 'Middle', 'Senior'])
    
    label_cols = [
        'Gender',
        'Working Professional or Student',
        'Dietary Habits',
        'Have you ever had suicidal thoughts ?',
        'Family History of Mental Illness'
    ]
    
    if is_train:
        encoders = {}
        for col in label_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        for col in label_cols:
            if col in df.columns and col in encoders:
                # Handle unseen labels
                le = encoders[col]
                df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
    
    ohe_cols = ['Profession', 'Sleep Duration', 'Age_Group']
    ohe_cols = [c for c in ohe_cols if c in df.columns]
    
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    
    return df, y, imputers, encoders


X_train_full, y_train_full, imputers, encoders = preprocess_data(
    train_data, is_train=True
)

print(f"Training features shape: {X_train_full.shape}")
print(f" Feature columns: {X_train_full.columns.tolist()[:10]}...")


X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# Validation
y_val_pred = rf_model.predict(X_val)
y_val_proba = rf_model.predict_proba(X_val)[:, 1]

val_accuracy = accuracy_score(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, y_val_proba)

print(f"\n Validation Accuracy: {val_accuracy:.4f}")
print(f" Validation ROC-AUC: {val_roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))


rf_model_full = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model_full.fit(X_train_full, y_train_full)
print("Model trained on full training data")


X_test, _, _, _ = preprocess_data(
    test_data, is_train=False, imputers=imputers, encoders=encoders
)

print(f"Test features shape: {X_test.shape}")

missing_cols = set(X_train_full.columns) - set(X_test.columns)
if missing_cols:
    print(f"Adding {len(missing_cols)} missing columns")
    for col in missing_cols:
        X_test[col] = 0

extra_cols = set(X_test.columns) - set(X_train_full.columns)
if extra_cols:
    print(f"Removing {len(extra_cols)} extra columns")
    X_test = X_test.drop(columns=list(extra_cols))

X_test = X_test[X_train_full.columns]
print(f"Test data aligned: {X_test.shape}")


test_predictions = rf_model_full.predict(X_test)

submission = pd.DataFrame({
    'id': test_ids,
    'Depression': test_predictions
})

submission.to_csv('submission.csv', index=False)

print(" Predictions completed!")
print("\nSubmission Preview:")
print(submission.head(10))
print(f"\n Statistics:")
print(f"   Total: {len(submission):,}")
print(f"   Depression = 0: {(submission['Depression'] == 0).sum():,} ({(submission['Depression'] == 0).sum() / len(submission) * 100:.1f}%)")
print(f"   Depression = 1: {(submission['Depression'] == 1).sum():,} ({(submission['Depression'] == 1).sum() / len(submission) * 100:.1f}%)")
print("\n Saved as: submission.csv")

