import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')


PALETTE = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#f4a261"]
sns.set_palette(PALETTE)
plt.style.use("seaborn-whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'grid.color': '#e0e0e0',
    'figure.facecolor': '#f5f5f5'
})


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print("Original train columns:", train.columns.tolist())


def clean_column_names(df):
    new_columns = []
    for col in df.columns:
        col_clean = re.sub(r'[^a-zA-Z0-9\s]', '', col.strip()).lower()
        col_clean = re.sub(r'\s+', '_', col_clean)
        new_columns.append(col_clean)
    df.columns = new_columns
    return df

train = clean_column_names(train)
test = clean_column_names(test)

print("Cleaned train columns:", train.columns.tolist())


id_col = 'id'

possible_targets = ['personality', 'type', 'target', 'label', 'class']
TARGET_COL = None

for candidate in possible_targets:
    if candidate in train.columns:
        TARGET_COL = candidate
        print(f"Exact match found: Using target column '{TARGET_COL}'")
        break

if TARGET_COL is None:
    for col in train.columns:
        if any(candidate in col for candidate in possible_targets):
            TARGET_COL = col
            print(f"Partial match found: Using target column '{TARGET_COL}'")
            break

if TARGET_COL is None:
    TARGET_COL = train.columns[-1]
    print(f"⚠️ WARNING: Using fallback target column '{TARGET_COL}'")
    
if TARGET_COL not in train.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame")


def map_personality(value):
    value = str(value).lower()
    if 'intro' in value or value in ['0', 'false', 'no', 'i', 'in']:
        return 'introvert'
    elif 'extro' in value or value in ['1', 'true', 'yes', 'e', 'ex']:
        return 'extrovert'
    else:
        return np.nan

train[TARGET_COL] = train[TARGET_COL].apply(map_personality)
train = train.dropna(subset=[TARGET_COL])

print(f"\nTraining set shape: {train.shape}")
print(f"Testing set shape: {test.shape}")
print(f"Target distribution:\n{train[TARGET_COL].value_counts(normalize=True)}")


plt.figure(figsize=(8, 5))
ax = sns.countplot(x=TARGET_COL, data=train, palette=PALETTE[:2])
plt.title(f'{TARGET_COL.title()} Distribution', fontsize=16)
plt.xlabel(TARGET_COL.title())
plt.ylabel('Count')

total = len(train)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    x = p.get_x() + p.get_width()/2
    y = p.get_height() + 20
    ax.annotate(percentage, (x, y), ha='center')
plt.tight_layout()
plt.show()


num_features = train.select_dtypes(include=['int64', 'float64']).columns
num_features = num_features.drop([id_col], errors='ignore')

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train[col], kde=True, color=PALETTE[0])
    plt.title(f'{col.title()} Distribution')
plt.tight_layout()
plt.suptitle('Numerical Features Distribution', y=1.02, fontsize=16)
plt.show()


cat_features = train.select_dtypes(include=['object']).columns
cat_features = cat_features.drop(TARGET_COL, errors='ignore')

plt.figure(figsize=(15, 8))
for i, col in enumerate(cat_features, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=col, data=train, palette=PALETTE[:2])
    plt.title(f'{col.title()} Distribution')
    plt.xticks(rotation=15)
plt.tight_layout()
plt.suptitle('Categorical Features Distribution', y=1.02, fontsize=16)
plt.show()


encoded_train = train.copy()

if TARGET_COL in encoded_train.columns:
    encoded_train[TARGET_COL] = encoded_train[TARGET_COL].map({'introvert': 0, 'extrovert': 1})

for col in cat_features:
    if col in encoded_train.columns:
        le = LabelEncoder()
        encoded_train[col] = le.fit_transform(encoded_train[col])

corr = encoded_train.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, 
            annot_kws={'size': 8}, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.target_encoders = {}
        self.feature_means = {}
        
    def fit(self, X, y=None):
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        for col in num_features:
            self.feature_means[col] = X[col].mean()
        
        cat_features = X.select_dtypes(include=['object']).columns
        for col in cat_features:
            if col in X.columns:
                target_mean = y.groupby(X[col]).mean()
                self.target_encoders[col] = target_mean
                
        return self
        
    def transform(self, X):
        X = X.copy()
        
        for col, mean_val in self.feature_means.items():
            if col in X.columns:
                X[col] = X[col].fillna(mean_val)
        
        for col, encoder in self.target_encoders.items():
            if col in X.columns:
                X[f'{col}_target_enc'] = X[col].map(encoder).fillna(0.5)
        
        interaction_pairs = [
            ('time_spent_alone', 'social_event_attendance'),
            ('going_out', 'friends'),
            ('posting', 'friends')
        ]
        
        for col1, col2 in interaction_pairs:
            clean_col1 = re.sub(r'[^a-z0-9]', '', col1)
            clean_col2 = re.sub(r'[^a-z0-9]', '', col2)
            if clean_col1 in X.columns and clean_col2 in X.columns:
                X[f'{clean_col1}_{clean_col2}_interaction'] = X[clean_col1] * X[clean_col2]
                
        if 'social_event_attendance' in X.columns and 'time_spent_alone' in X.columns:
            X['social_alone_ratio'] = X['social_event_attendance'] / (X['time_spent_alone'] + 1e-8)
            
        if 'friends' in X.columns and 'posting' in X.columns:
            X['friend_post_ratio'] = X['friends'] / (X['posting'] + 1e-8)
            
        if 'avoids_interaction' in X.columns and 'energetic' in X.columns:
            X['social_energy_score'] = X['energetic'] - X['avoids_interaction']
            
        return X


X = train.drop(columns=[id_col, TARGET_COL], errors='ignore')
y = train[TARGET_COL].map({'introvert': 0, 'extrovert': 1})


feature_engineer = FeatureEngineer()
feature_engineer.fit(X, y)
X_fe = feature_engineer.transform(X)
test_fe = feature_engineer.transform(test.drop(columns=[id_col], errors='ignore'))

print(f"\nTotal features after engineering: {len(X_fe.columns)}")
print(f"New features: {list(set(X_fe.columns) - set(train.columns))}")


num_features = X_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X_fe.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical features: {len(num_features)}")
print(f"Categorical features: {len(cat_features)}")


preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]), cat_features)
])

X_preprocessed = preprocessor.fit_transform(X_fe)
X_preprocessed = pd.DataFrame(X_preprocessed, 
                             columns=num_features + cat_features)



X_train, X_val, y_train, y_val = train_test_split(
    X_preprocessed, y, test_size=0.2, stratify=y, random_state=42
)


models = {
    'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, scale_pos_weight=(len(y) - sum(y)) / sum(y)),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced')
}


results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_pred)
    results[name] = accuracy
    print(f"{name} Validation Accuracy: {accuracy:.4f}")


class EnsembleModel:
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        for name, model in self.models.items():
            model.fit(X, y)
            
    def predict_proba(self, X):
        probas = [model.predict_proba(X) for model in self.models.values()]
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba > threshold).astype(int)


ensemble = EnsembleModel(models)
ensemble.fit(X_train, y_train)
val_pred = ensemble.predict(X_val)
accuracy = accuracy_score(y_val, val_pred)
print(f"\nEnsemble Validation Accuracy: {accuracy:.4f}")

ensemble.fit(X_preprocessed, y)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_fe, y)):
    X_train_fold, X_val_fold = X_fe.iloc[train_idx], X_fe.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    X_train_fold_processed = preprocessor.transform(X_train_fold)
    X_val_fold_processed = preprocessor.transform(X_val_fold)
    
    fold_ensemble = EnsembleModel(models)
    fold_ensemble.fit(X_train_fold_processed, y_train_fold)
    
    val_pred = fold_ensemble.predict(X_val_fold_processed)
    fold_acc = accuracy_score(y_val_fold, val_pred)
    fold_f1 = f1_score(y_val_fold, val_pred)
    accuracies.append(fold_acc)
    f1_scores.append(fold_f1)
    
    print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}, F1 Score: {fold_f1:.4f}")

print(f"\nAverage CV Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Average CV F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")



y_pred = ensemble.predict(X_preprocessed)
print("\nClassification Report:")
print(classification_report(y, y_pred))


cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Introvert', 'Extrovert'], 
            yticklabels=['Introvert', 'Extrovert'])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


y_proba = ensemble.predict_proba(X_preprocessed)[:, 1]
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color=PALETTE[3], lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=16)
plt.legend(loc="lower right")
plt.show()


lgb_importance = pd.DataFrame({
    'Feature': num_features + cat_features,
    'Importance': models['LightGBM'].feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=lgb_importance, palette=PALETTE)
plt.title('Top 15 Feature Importances (LightGBM)', fontsize=16)
plt.tight_layout()
plt.show()


thresholds = np.linspace(0.3, 0.7, 41)
accuracies = []

for threshold in thresholds:
    val_pred = ensemble.predict(X_val, threshold=threshold)
    accuracy = accuracy_score(y_val, val_pred)
    accuracies.append(accuracy)


best_idx = np.argmax(accuracies)
best_threshold = thresholds[best_idx]
best_accuracy = accuracies[best_idx]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, color=PALETTE[0], lw=2)
plt.axvline(x=best_threshold, color=PALETTE[1], linestyle='--', 
            label=f'Best Threshold: {best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Threshold Optimization', fontsize=16)
plt.legend()
plt.show()

print(f"Best Threshold: {best_threshold:.4f}, Accuracy: {best_accuracy:.4f}")


def predict_personality(input_data, artifacts_path='personality_classifier.pkl', threshold=best_threshold):
    """
    Predict personality from input data
    
    Args:
        input_data: dict of feature values
        artifacts_path: path to saved artifacts
        threshold: probability threshold for classification
        
    Returns:
        tuple: (predicted_personality, confidence)
    """
    artifacts = joblib.load(artifacts_path)
    feature_engineer = artifacts['feature_engineer']
    preprocessor = artifacts['preprocessor']
    ensemble = artifacts['ensemble']
    
    clean_input = {}
    for key, value in input_data.items():
        clean_key = re.sub(r'[^a-z0-9]', '', key.lower().replace(' ', '_'))
        clean_input[clean_key] = value
    
    input_df = pd.DataFrame([clean_input])
    
    input_fe = feature_engineer.transform(input_df)
    
    input_preprocessed = preprocessor.transform(input_fe)
    
    proba = ensemble.predict_proba(input_preprocessed)[0]
    prediction = ensemble.predict(input_preprocessed, threshold=threshold)[0]
    
    personality = "extrovert" if prediction == 1 else "introvert"
    
    confidence = proba[1] if personality == "extrovert" else proba[0]
    
    return personality, confidence


artifacts = {
    'feature_engineer': feature_engineer,
    'preprocessor': preprocessor,
    'models': models,
    'ensemble': ensemble,
    'id_col': id_col,
    'target_col': TARGET_COL,
    'feature_names': num_features + cat_features
}


joblib.dump(artifacts, 'personality_classifier.pkl')
print("Model artifacts saved as 'personality_classifier.pkl'")


test_predictions = []
test_confidences = []

for i, row in test.iterrows():
    input_data = row.to_dict()
    personality, confidence = predict_personality(input_data)
    test_predictions.append(personality)
    test_confidences.append(confidence)


submission = pd.DataFrame({
    'id': test[id_col],
    TARGET_COL: test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file created: 'submission.csv'")

results = pd.DataFrame({
    'id': test[id_col],
    'predicted_personality': test_predictions,
    'confidence': test_confidences
})
results.to_csv('prediction_results.csv', index=False)
print("Prediction results with confidence saved: 'prediction_results.csv'")


sample_input = {
    'id': 99999,
    'Time spent alone': 8.5,
    'Social event attendance': 1,
    'Going outside': 2,
    'Friends circle size': 5,
    'Post frequency': 1,
    'Stage fear': 'Yes',
    'Drained after socializing': 'Yes'
}

personality, confidence = predict_personality(sample_input)
print(f"\nTest Prediction Successful!")
print(f"Predicted Personality: {personality}")
print(f"Confidence: {confidence:.2%}")
print(f"Input Features: {sample_input}")




