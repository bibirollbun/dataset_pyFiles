!pip install xgboost catboost shap optuna lightgbm category_encoders


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score 
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter
import torch
import psutil
import warnings 
import os
import json
import joblib
from tqdm import tqdm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import optuna
from lightgbm import LGBMClassifier
from category_encoders import TargetEncoder
warnings.filterwarnings('ignore')

base_path = "/kaggle/input/equity-post-HCT-survival-predictions"
df_train =pd.read_csv(os.path.join(base_path,"train.csv"))
df_test = pd.read_csv(os.path.join(base_path, "test.csv"))
df_submission = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))
df_dictionary = pd.read_csv(os.path.join(base_path, "data_dictionary.csv"))

print("Veri setleri yüklendi:")
print(f"Eğitim veri seti: {df_train.shape}")
print(f"Test veri seti: {df_test.shape}")
print(f"submission veri seti: {df_submission.shape}")

def print_resource_usage():
    process = psutil.Process()
    print(f"Bellek kullanımı: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU kullanımı: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df
    
def feature_selection(X, y, threshold=0.01):
    print("Özellik seçimi yapılıyor...")
    selector = SelectFromModel(
        estimator=XGBClassifier(random_state=42),
        threshold=threshold
    )
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Seçilen özellik sayısı: {len(selected_features)}")
    return selected_features
def preprocess_data(data, fit=False, transformers=None):
    try:
        print("Veri ön işleme başlıyor...")
        data_processed = data.copy()
        columns_to_drop = ['efs', 'efs_time', 'ID'] if fit else ['ID']
        data_processed = data_processed.drop(columns_to_drop, axis=1, errors='ignore')
        print("Veri tipi optimizasyonu yapılıyor...")
        data_processed = optimize_dtypes(data_processed)
       
        numeric_columns = data_processed.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
        categorical_columns = data_processed.select_dtypes(include=['object']).columns
        print(f"Sayısal sütunlar: {len(numeric_columns)}")
        print(f"Kategorik sütunlar: {len(categorical_columns)}")
        
        if len(numeric_columns) > 0:
            print("Eksik veriler dolduruluyor...")
            missing_cols = []
            extra_cols = []
            if fit:
                imputer = KNNImputer(n_neighbors=5)
                data_processed[numeric_columns] = imputer.fit_transform(data_processed[numeric_columns])
                transformers = {
                    'imputer': imputer,
                    'qt': {col: QuantileTransformer(output_distribution='normal') for col in numeric_columns},
                    'pt' : {col: PowerTransformer() for col in numeric_columns},
                    'numeric_columns': list(numeric_columns)
                }
                for col in numeric_columns:
                    transformers['qt'][col].fit(data_processed[[col]])
                    transformers['pt'][col].fit(data_processed[[col]])
            else:
                if transformers is None:
                    raise ValueError("Test verisi için transformers parametresi gerekli!")
                fit_numeric_columns = transformers['numeric_columns']
                missing_cols = [col for col in fit_numeric_columns if col not in numeric_columns]
                extra_cols = [col for col in numeric_columns if col not in fit_numeric_columns]
                if missing_cols:
                    raise ValueError(f"Test verisinde eksik sütunlar: {missing_cols}")
                if extra_cols:
                    data_processed = data_processed.drop(extra_cols, axis=1)
                    numeric_columns = [col for col in numeric_columns if col in fit_numeric_columns]
                data_processed[numeric_columns] = transformers['imputer'].transform(data_processed[numeric_columns])
                
            for i in range(len(numeric_columns)):
                for j in range(i+1, len(numeric_columns)):
                    col1, col2 = numeric_columns[i], numeric_columns[j]
                    data_processed[f'{col1}_times_{col2}'] = data_processed[col1] * data_processed[col2]
                    data_processed[f'{col1}_div_{col2}'] = data_processed[col1] / (data_processed[col2] + 1e-6)
            for col in numeric_columns:
                if fit:
                    data_processed[f'{col}_quantile'] = transformers['qt'][col].transform(data_processed[[col]])
                    data_processed[f'{col}_power'] = transformers['pt'][col].transform(data_processed[[col]])
                else:
                    data_processed[f'{col}_quantile'] = transformers['qt'][col].transform(data_processed[[col]])
                    data_processed[f'{col}_power'] = transformers['pt'][col].transform(data_processed[[col]])
                
            print("Aykırı değerler işleniyor...")
            for col in tqdm(numeric_columns, desc="Aykırı değerler tespit ediliyor"):
                Q1 = data_processed[col].quantile(0.25)      
                Q3 = data_processed[col].quantile(0.75)          
                IQR = Q3 - Q1         
                lower_bound = Q1 - 1.5 * IQR           
                upper_bound = Q3 + 1.5 * IQR
                data_processed[col] = data_processed[col].clip(lower_bound, upper_bound)          
            
            print("Özellik mühendisliği yapılıyor...")
            for col in tqdm(numeric_columns, desc="Özellikler oluşturuluyor"):
                data_processed[f'{col}_squared'] = data_processed[col] ** 2
                data_processed[f'{col}_cubed'] = data_processed[col] ** 3
                data_processed[f'{col}_log'] = np.log1p(data_processed[col].clip(lower=0))
                data_processed[f'{col}_exp'] = np.exp(data_processed[col].clip(upper=10))
             
            print("Polinom özellikleri oluşturuluyor...")
            poly = PolynomialFeatures(degree=2, include_bias=False)
            if fit:
                poly_features = poly.fit_transform(data_processed[numeric_columns]) if fit else poly.transform(data_processed[numeric_columns])
                transformers['poly'] = poly
            else:
                poly_features = transformers['poly'].transform(data_processed[numeric_columns])
            poly_feature_names = [f'poly_{i}' for  i in range(poly_features.shape[1])]
            data_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
            data_processed = pd.concat([data_processed, data_poly], axis=1)
        if len(categorical_columns) > 0:
            print("Kategorik değişkenler işleniyor...")
            for col in tqdm(categorical_columns, desc="Kategorik değişkenler işleniyor"):
                data_processed[col] = data_processed[col].fillna(data_processed[col].mode()[0])
                
            print("Kategorik değişkenler sayısallaştırılıyor...")
            for col in tqdm(categorical_columns, desc="Kategorik değişkenler dönüştürülüyor"):
                if fit:
                    le = LabelEncoder()
                    data_processed[col] = le.fit_transform(data_processed[col])
                    transformers['le_' + col] = le
                else:
                    data_processed[col] = transformers['le_' + col].transform(data_processed[col])
                    
        print("Veri ön işleme tamamlandı.")
        if fit:
            return data_processed, transformers
        else:
            return data_processed
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        raise
print("\nEğitim veri seti ön işleniyor...")
df_train_processed, transformers = preprocess_data(df_train, fit=True)

print("\nTest veri seti ön işleniyor...")
df_test_processed = preprocess_data(df_test, fit=False, transformers=transformers)

print("\nİşlenmiş eğitim veri seti boyutu:", df_train_processed.shape)
print("İşlenmiş test veri seti boyutu:", df_test_processed.shape)

print("\nÖzellikler ve hedef değişken ayrılıyor...")
y_train = df_train['efs']
X_train = df_train_processed
X_test = df_test_processed
print(f"X_train şekli: {X_train.shape}")
print(f"y_train şekli: {y_train.shape}")
print(f"X_test şekli: {X_test.shape}")

X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()

print("X_train_imputed'daki NaN sayısı:", X_train_imputed.isna().sum().sum())
if X_train_imputed.isna().sum().sum() > 0:
    raise ValueError("X_train_imputed hala NaN içeriyor, preprocess_data fonksiyonunu kontrol et!")

print("\nVeri dengeleme başlıyor...")
print("Dengeleme öncesi hedef değişken dağılımı:")
print(Counter(y_train))
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)
selected_features = feature_selection(X_train_resampled, y_train_resampled)
X_train_selected = X_train_resampled[selected_features]
X_test_selected = X_test_imputed[selected_features]
print("\nDengeleme sonrası hedef değişken dağılımı:")
print(Counter(y_train_resampled))

def create_ensemble_models():
    models = {
        'xgb': XGBClassifier(random_state=42),
        'lgbm': LGBMClassifier(random_state=42),
        'catboost': CatBoostClassifier(random_state=42, verbose=False),
        'rf': RandomForestClassifier(random_state=42),
        'gb': GradientBoostingClassifier(random_state=42),
    }
    return models
    
def create_stacking_model(base_models):
    meta_model = LogisticRegression()
    stacking = StackingClassifier(
        estimators=[(name, model) for name, model in base_models.items()],
        final_estimator=meta_model,
        cv=5
    )
    return stacking
    
def evaluate_model(model, X, y, X_val=None, y_val=None):
    if X_val is None or y_val is None:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        return {
            'roc_auc_mean': cv_scores.mean(),
            'roc_auc_std': cv_scores.std()
        }
    else:
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return {
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'accuracy': accuracy_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred)
        }
        
def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators',100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42
    }     
    model = XGBClassifier(**xgb_params)
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring="roc_auc")
    return scores.mean()
    
print("\nHiperparametre optimizasyonu başlıyor...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("\nEn iyi parametreler:", study.best_params)
print("En iyi skor:", study.best_value)

best_params = study.best_params
final_model = XGBClassifier(**best_params)
final_model.fit(X_train_resampled, y_train_resampled)
print("\nEnsemble modeller oluşturuluyor...")
base_models = create_ensemble_models()
stacking_model = create_stacking_model(base_models)

print("\nEnsemble modeller eğitiliyor...")
model_scores = {}
for name, model in base_models.items():
    print(f"\n{name} modeli eğitiliyor...")
    model.fit(X_train_selected, y_train_resampled)
    score = evaluate_model(model, X_train_selected, y_train_resampled)
    model_scores[name] = score
    print(f"{name} model performansı:")
    print(score)
    
print("\nStacking model eğitiliyor...")
stacking_model.fit(X_train_selected, y_train_resampled)
stacking_scored = evaluate_model(stacking_model, X_train_selected, y_train_resampled)
print("Stacking model performansı:")
print(stacking_scored)

print("\nStacking modelin çapraz doğrulama ROC AUC skoru hesaplanıyor...")
cv_scores_stacking = cross_val_score(stacking_model, X_train_selected, y_train_resampled)
print(f"Stacking Model - Ortalama ROC AUC: {cv_scores_stacking.mean():.4f}, Std: {cv_scores_stacking.std():.4f}")
print("\nBireysel modellerin çapraz doğrulama ROC AUC skorları hesaplanıyor...")
for name, model in base_models.items():
    cv_scores = cross_val_score(model, X_train_selected, y_train_resampled, cv=5, scoring='roc_auc')
    print(f"{name} - Ortalama ROC AUC: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")

print("\nFinal tahminler oluşturuluyor...")
predictions = []
for name, model in base_models.items():
    pred = model.predict_proba(X_test_selected)[:, 1]
    predictions.append(pred)
final_predictions = np.mean(predictions, axis=0)

results = {
    'base_models': model_scores,
    'stacking_model': stacking_scored,
    'selected_features': selected_features
}
with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=4)
    
submission = pd.DataFrame({
    'ID': df_test['ID'],
    'efs': final_predictions
})
submission.to_csv('submission.csv', index=False)

joblib.dump(stacking_model, 'stacking_model.joblib')
for name, model in base_models.items():
    joblib.dump(model, f'{name}_model.joblib')
    
print("\nTüm modeller ve sonuçlar kaydedildi!")
print_resource_usage()


base_path = "/kaggle/input/equity-post-HCT-survival-predictions"
df_train =pd.read_csv(os.path.join(base_path,"train.csv"))
df_test = pd.read_csv(os.path.join(base_path, "test.csv"))
df_submission = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))
df_dictionary = pd.read_csv(os.path.join(base_path, "data_dictionary.csv"))
print("Veri setleri yüklendi:")




