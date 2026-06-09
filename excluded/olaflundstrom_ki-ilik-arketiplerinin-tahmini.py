# --- KÃ¼tÃ¼phanelerin Kurulumu ve Ä°Ã§e AktarÄ±lmasÄ± ---
!pip install -q optuna catboost xgboost lightgbm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import lightgbm as lgb
import xgboost as xgb
import catboost as ctb

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# GÃ¶rselleÅŸtirme ayarlarÄ±
sns.set_style('whitegrid')
plt.rc('figure', figsize=(12, 8))
plt.rc('font', size=12)


# --- Veri Setlerini YÃ¼kleme ---
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    sample_submission_df = pd.read_csv("sample_submission.csv")
except FileNotFoundError:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
    sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# Korelasyon Analizi iÃ§in veriyi hazÄ±rla
corr_df = train_df.copy()
le_corr = LabelEncoder()
corr_df['Personality'] = le_corr.fit_transform(corr_df['Personality'])
for col in ['Stage_fear', 'Drained_after_socializing']:
    if col in corr_df.columns:
        corr_df[col] = le_corr.fit_transform(corr_df[col])

plt.figure(figsize=(14, 12))
sns.heatmap(corr_df.drop('id', axis=1).corr(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Ã–zelliklerin Korelasyon Matrisi', fontsize=18, weight='bold')
plt.show()


class FeatureEngineer:
    def __init__(self):
        self.le = LabelEncoder()
        self.imputer = KNNImputer(n_neighbors=5)
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        self.poly_feature_names_ = None
        self.original_feature_names_ = None
        self.final_feature_names_ = None

    def fit_transform(self, df_train, df_test):
        y_train_series = df_train['Personality'].copy()
        df_train = df_train.drop('Personality', axis=1)
        
        train_ids = df_train['id']
        test_ids = df_test['id']
        
        combined_df = pd.concat([df_train.drop('id', axis=1), df_test.drop('id', axis=1)], ignore_index=True)
        
        combined_df = self._create_semantic_features(combined_df)
        
        self.original_feature_names_ = combined_df.columns.tolist()
        df_imputed = pd.DataFrame(self.imputer.fit_transform(combined_df), columns=self.original_feature_names_)
        
        poly_features = self.poly.fit_transform(df_imputed)
        self.poly_feature_names_ = self.poly.get_feature_names_out(self.original_feature_names_)
        df_poly = pd.DataFrame(poly_features, columns=self.poly_feature_names_)
        
        df_scaled = pd.DataFrame(self.scaler.fit_transform(df_poly), columns=self.poly_feature_names_)
        self.final_feature_names_ = df_scaled.columns.tolist()
        
        X_train = df_scaled.iloc[:len(df_train)]
        X_test = df_scaled.iloc[len(df_train):]
        
        y_train_encoded = self.le.fit_transform(y_train_series)
        
        return X_train, y_train_encoded, X_test, test_ids

    @staticmethod
    def _create_semantic_features(df):
        df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
        df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
        df['Social_Engagement'] = df['Social_event_attendance'] + df['Friends_circle_size']
        df['Digital_Footprint'] = df['Post_frequency'] / (df['Time_spent_Alone'] + 1)
        return df

print("Ã–zellik mÃ¼hendisliÄŸi ve Ã¶n iÅŸleme baÅŸlatÄ±lÄ±yor...")
fe = FeatureEngineer()
X, y, X_test, test_ids = fe.fit_transform(train_df, test_df)

print(f"Ä°ÅŸlem tamamlandÄ±. Yeni Ã¶zellik sayÄ±sÄ±: {X.shape[1]}")

# EÄŸitim ve doÄŸrulama setlerine ayÄ±r
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 4. Modelleme Stratejisi ---
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Optuna logging'i sustur
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 60)
print("4. MODELLEMÄ° STRATEJÄ°SÄ° BAÅ�LATILIYOR")
print("=" * 60)

# --- 4.1. Temel Modeller (Baseline Models) ---
print("\n4.1. Temel Modeller DeÄŸerlendiriliyor...")

# Ã‡apraz doÄŸrulama stratejisi
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Temel modeller
baseline_models = {
    'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision_Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'KNN': KNeighborsClassifier(n_neighbors=7)
}

baseline_results = {}
for name, model in baseline_models.items():
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    baseline_results[name] = np.mean(scores)
    print(f"{name}: {np.mean(scores):.5f} (Â±{np.std(scores):.4f})")

# --- 4.2. GeliÅŸmiÅŸ Gradyan ArtÄ±rma Makineleri ---
print("\n4.2. GeliÅŸmiÅŸ Gradyan ArtÄ±rma Makineleri...")

# Random Forest (temel topluluk modeli)
print("Random Forest eÄŸitiliyor...")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_scores = cross_val_score(rf_model, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
rf_result = np.mean(rf_scores)
print(f"Random Forest: {rf_result:.5f} (Â±{np.std(rf_scores):.4f})")

# Hiperparametre optimizasyonu fonksiyonlarÄ±
def optimize_lightgbm(X, y, cv_strategy, n_trials=100):
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbosity': -1,
            'random_state': 42
        }
        
        scores = []
        for train_idx, val_idx in cv_strategy.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            model = lgb.train(params, train_data, num_boost_round=1000, 
                             valid_sets=[train_data], 
                             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            preds = model.predict(X_val_fold)
            preds_binary = (preds > 0.5).astype(int)
            scores.append(accuracy_score(y_val_fold, preds_binary))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def optimize_catboost(X, y, cv_strategy, n_trials=100):
    def objective(trial):
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 50
        }
        
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
        else:
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1)
        
        scores = []
        for train_idx, val_idx in cv_strategy.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            model = ctb.CatBoostClassifier(**params)
            model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold))
            
            preds = model.predict(X_val_fold)
            scores.append(accuracy_score(y_val_fold, preds))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

# LightGBM optimizasyonu
print("LightGBM hiperparametre optimizasyonu yapÄ±lÄ±yor...")
lgb_best_params, lgb_best_score = optimize_lightgbm(X, y, cv_strategy, n_trials=50)
print(f"LightGBM En Ä°yi Skor: {lgb_best_score:.5f}")

# CatBoost optimizasyonu
print("CatBoost hiperparametre optimizasyonu yapÄ±lÄ±yor...")
ctb_best_params, ctb_best_score = optimize_catboost(X, y, cv_strategy, n_trials=50)
print(f"CatBoost En Ä°yi Skor: {ctb_best_score:.5f}")

# XGBoost (basit parametrelerle)
print("XGBoost eÄŸitiliyor...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)
xgb_scores = cross_val_score(xgb_model, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
xgb_result = np.mean(xgb_scores)
print(f"XGBoost: {xgb_result:.5f} (Â±{np.std(xgb_scores):.4f})")

# SonuÃ§larÄ± topla
advanced_results = {
    'Random_Forest': rf_result,
    'LightGBM_Optimized': lgb_best_score,
    'CatBoost_Optimized': ctb_best_score,
    'XGBoost': xgb_result
}

# --- 4.3. Hibrit ve Topluluk Modelleri ---
print("\n4.3. Hibrit ve Topluluk Modelleri...")

# --- 4.3.1. Oylama SÄ±nÄ±flandÄ±rÄ±cÄ±sÄ± ---
print("4.3.1. Oylama SÄ±nÄ±flandÄ±rÄ±cÄ±sÄ± eÄŸitiliyor...")

# En iyi modelleri seÃ§
best_lgb = lgb.LGBMClassifier(**lgb_best_params, n_estimators=1000, random_state=42, verbose=-1)
best_ctb = ctb.CatBoostClassifier(**ctb_best_params, random_seed=42, verbose=False)
best_xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')

# Oylama modeli
voting_clf = VotingClassifier(
    estimators=[
        ('lgb', best_lgb),
        ('ctb', best_ctb),
        ('xgb', best_xgb)
    ],
    voting='soft'
)

voting_scores = cross_val_score(voting_clf, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
voting_result = np.mean(voting_scores)
print(f"Voting Classifier: {voting_result:.5f} (Â±{np.std(voting_scores):.4f})")

# --- 4.3.2. Derin Sinir AÄŸÄ± (Deep Neural Network) ---
print("4.3.2. Derin Sinir AÄŸÄ± tasarlanÄ±yor ve eÄŸitiliyor...")

class PersonalityDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3):
        super(PersonalityDNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Bilgiye dayalÄ± kayÄ±p fonksiyonu
class KnowledgeInformedLoss(nn.Module):
    def __init__(self, lambda_knowledge=0.1):
        super(KnowledgeInformedLoss, self).__init__()
        self.lambda_knowledge = lambda_knowledge
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions, targets, features):
        # Standart BCE loss
        bce_loss = self.bce_loss(predictions.squeeze(), targets.float())
        
        # Psikolojik bilgiye dayalÄ± ceza terimi
        # Ã–rnek: Sosyal etkinlik katÄ±lÄ±mÄ± yÃ¼ksek olanlarÄ±n extrovert olma eÄŸilimi
        social_engagement_idx = 9  # Social_Engagement feature index (Ã¶zellik mÃ¼hendisliÄŸinden)
        social_scores = features[:, social_engagement_idx]
        
        # YÃ¼ksek sosyal puanÄ± olan ama introvert olarak tahmin edilen durumlar iÃ§in ceza
        social_penalty = torch.mean(torch.relu(social_scores - 0.5) * torch.relu(0.5 - predictions.squeeze()))
        
        # Toplam kayÄ±p
        total_loss = bce_loss + self.lambda_knowledge * social_penalty
        
        return total_loss

def train_dnn_cv(X, y, cv_strategy, epochs=200, batch_size=64, learning_rate=0.001):
    """DNN modelini Ã§apraz doÄŸrulama ile eÄŸit"""
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X, y)):
        X_train_fold = torch.FloatTensor(X.iloc[train_idx].values)
        X_val_fold = torch.FloatTensor(X.iloc[val_idx].values)
        y_train_fold = torch.FloatTensor(y[train_idx])
        y_val_fold = torch.FloatTensor(y[val_idx])
        
        # Model oluÅŸtur
        model = PersonalityDNN(input_dim=X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = KnowledgeInformedLoss(lambda_knowledge=0.1)
        
        # EÄŸitim
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_fold)
            loss = criterion(outputs, y_train_fold, X_train_fold)
            loss.backward()
            optimizer.step()
        
        # DeÄŸerlendirme
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            val_preds = (val_outputs.squeeze() > 0.5).float()
            accuracy = (val_preds == y_val_fold).float().mean().item()
            cv_scores.append(accuracy)
    
    return np.mean(cv_scores), np.std(cv_scores)

# DNN eÄŸitimi
dnn_mean_score, dnn_std_score = train_dnn_cv(X, y, cv_strategy)
print(f"Deep Neural Network: {dnn_mean_score:.5f} (Â±{dnn_std_score:.4f})")

# --- 4.3.3. YÄ±ÄŸÄ±nlama (Stacking) Modeli ---
print("4.3.3. YÄ±ÄŸÄ±nlama (Stacking) Modeli oluÅŸturuluyor...")

def get_oof_predictions(model, X, y, cv_strategy):
    """Out-of-fold tahminleri al"""
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in cv_strategy.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold = y[train_idx]
        
        # Model tipine gÃ¶re eÄŸitim
        if hasattr(model, 'predict_proba'):
            model.fit(X_train_fold, y_train_fold)
            oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
        else:
            # DNN iÃ§in Ã¶zel iÅŸlem
            X_train_tensor = torch.FloatTensor(X_train_fold.values)
            X_val_tensor = torch.FloatTensor(X_val_fold.values)
            y_train_tensor = torch.FloatTensor(y_train_fold)
            
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = KnowledgeInformedLoss(lambda_knowledge=0.1)
            
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor, X_train_tensor)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                oof_preds[val_idx] = val_outputs.squeeze().numpy()
    
    return oof_preds

# Base learner'larÄ± tanÄ±mla
base_learners = {
    'lgb': lgb.LGBMClassifier(**lgb_best_params, n_estimators=1000, random_state=42, verbose=-1),
    'ctb': ctb.CatBoostClassifier(**ctb_best_params, random_seed=42, verbose=False),
    'dnn': PersonalityDNN(input_dim=X.shape[1])
}

# OOF tahminlerini al
print("Base learner'lar iÃ§in OOF tahminleri alÄ±nÄ±yor...")
oof_predictions = {}

for name, model in base_learners.items():
    print(f"  {name} OOF tahminleri alÄ±nÄ±yor...")
    oof_predictions[name] = get_oof_predictions(model, X, y, cv_strategy)

# Meta-features oluÅŸtur
meta_features = np.column_stack([oof_predictions[name] for name in base_learners.keys()])
meta_X = pd.DataFrame(meta_features, columns=list(base_learners.keys()))

# Meta-learner eÄŸit (Lojistik Regresyon)
meta_model = LogisticRegression(random_state=42)
meta_scores = cross_val_score(meta_model, meta_X, y, cv=cv_strategy, scoring='accuracy')
stacking_result = np.mean(meta_scores)
print(f"Stacking Model: {stacking_result:.5f} (Â±{np.std(meta_scores):.4f})")

# --- Test seti iÃ§in tahminler ---
print("\nTest seti iÃ§in base learner tahminleri hazÄ±rlanÄ±yor...")

# Base learner'larÄ± tÃ¼m veri Ã¼zerinde eÄŸit
test_preds = {}
for name, model in base_learners.items():
    if name == 'dnn':
        # DNN iÃ§in Ã¶zel eÄŸitim
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y)
        X_test_tensor = torch.FloatTensor(X_test.values)
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = KnowledgeInformedLoss(lambda_knowledge=0.1)
        
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor, X_tensor)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_preds[name] = test_outputs.squeeze().numpy()
    else:
        model.fit(X, y)
        test_preds[name] = model.predict_proba(X_test)[:, 1]

# Test meta-features
test_meta_features = np.column_stack([test_preds[name] for name in base_learners.keys()])
test_preds_stack = pd.DataFrame(test_meta_features, columns=list(base_learners.keys()))

# Meta-model'i tÃ¼m veri Ã¼zerinde eÄŸit
meta_model.fit(meta_X, y)

# SonuÃ§larÄ± topla
ensemble_results = {
    'Voting_Classifier': voting_result,
    'Deep_Neural_Network': dnn_mean_score,
    'Stacking_Model': stacking_result
}

print("\n" + "="*60)
print("4. BÃ–LÃœM TAMAMLANDI - MODEL PERFORMANSLARI:")
print("="*60)
print("\nTemel Modeller:")
for model, score in baseline_results.items():
    print(f"  {model}: {score:.5f}")

print("\nGeliÅŸmiÅŸ Modeller:")
for model, score in advanced_results.items():
    print(f"  {model}: {score:.5f}")

print("\nTopluluk Modelleri:")
for model, score in ensemble_results.items():
    print(f"  {model}: {score:.5f}")

# En iyi modeli belirle
all_results = {**baseline_results, **advanced_results, **ensemble_results}
best_model_name = max(all_results, key=all_results.get)
best_score = all_results[best_model_name]

print(f"\nğŸ�† EN Ä°YÄ° MODEL: {best_model_name} - Skor: {best_score:.5f}")


# --- SonuÃ§larÄ± BirleÅŸtirme ve GÃ¶rselleÅŸtirme ---
all_results = {**baseline_results, **advanced_results, **ensemble_results}
results_df = pd.DataFrame.from_dict(all_results, orient='index', columns=['Accuracy']).sort_values('Accuracy', ascending=False)

plt.figure(figsize=(12, 10))
ax = sns.barplot(x=results_df['Accuracy'], y=results_df.index, palette='rocket')
plt.title('Modellerin Performans KarÅŸÄ±laÅŸtÄ±rmasÄ± (DoÄŸruluk)', fontsize=18, weight='bold')
plt.xlabel('DoÄŸruluk Skoru', fontsize=12)
plt.xlim(min(results_df['Accuracy']) * 0.99, max(results_df['Accuracy']) * 1.001)
for i, (p, val) in enumerate(zip(ax.patches, results_df['Accuracy'])):
    ax.text(val + 0.0005, p.get_y() + p.get_height() / 2, f'{val:.5f}', ha='left', va='center')
plt.show()

# --- Nihai GÃ¶nderim ---
print("En iyi model olan Stacking modeli ile nihai tahminler yapÄ±lÄ±yor...")
final_predictions_encoded = meta_model.predict(test_preds_stack)
final_predictions = fe.le.inverse_transform(final_predictions_encoded)

submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nGÃ¶nderim dosyasÄ± 'submission.csv' baÅŸarÄ±yla oluÅŸturuldu.")
display(submission_df.head())

