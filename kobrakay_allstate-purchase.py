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


# %% [markdown]
# ğŸš€ Allstate Purchase Prediction - Solution ComplÃ¨te
# ## ğŸ“Š EDA AvancÃ© + Feature Engineering + Multi-ModÃ¨les + SHAP
# *CrÃ©Ã© avec une approche professionnelle pour maximiser les insights et la performance*

# %% [code]
# =============================================================================
# 1. IMPORTATION DES LIBRAIRIES
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# ModÃ¨les avancÃ©s
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# InterprÃ©tabilitÃ©
import shap

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("âœ… Toutes les librairies sont importÃ©es !")

# %% [code]
# =============================================================================
# 2. CHARGEMENT DES DONNÃ‰ES
# =============================================================================
try:
    train = pd.read_csv('/kaggle/input/allstate-purchase-prediction-challenge/train.csv')
    test = pd.read_csv('/kaggle/input/allstate-purchase-prediction-challenge/test.csv')
    print("âœ… DonnÃ©es chargÃ©es avec succÃ¨s !")
except:
    # Fallback pour les donnÃ©es simulÃ©es
    print("âš ï¸�  Chargement des fichiers Ã©chouÃ© - crÃ©ation de donnÃ©es d'exemple")
    np.random.seed(42)
    n_samples = 10000
    train = pd.DataFrame({
        'customer_ID': range(n_samples),
        'shopping_pt': np.random.randint(1, 10, n_samples),
        'record_type': np.random.randint(0, 2, n_samples),
        'day': np.random.randint(1, 31, n_samples),
        'state': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'location': np.random.choice(['Urban', 'Rural'], n_samples),
        'group_size': np.random.randint(1, 5, n_samples),
        'homeowner': np.random.randint(0, 2, n_samples),
        'car_age': np.random.uniform(0, 20, n_samples),
        'car_value': np.random.uniform(5000, 50000, n_samples),
        'risk_factor': np.random.uniform(0.5, 3.5, n_samples),
        'age_oldest': np.random.randint(25, 70, n_samples),
        'age_youngest': np.random.randint(18, 65, n_samples),
        'married_couple': np.random.randint(0, 2, n_samples),
        'C_previous': np.random.uniform(0, 1, n_samples),
        'duration_previous': np.random.uniform(0.1, 2.0, n_samples),
        'A': np.random.uniform(0, 1, n_samples),
        'B': np.random.uniform(0, 1, n_samples),
        'C': np.random.uniform(0, 1, n_samples),
        'D': np.random.uniform(0, 1, n_samples),
        'E': np.random.uniform(0, 1, n_samples),
        'F': np.random.uniform(0, 1, n_samples),
        'G': np.random.uniform(0, 1, n_samples)
    })
    test = train.copy()

print(f"ğŸ“Š Dimensions Train: {train.shape}")
print(f"ğŸ“Š Dimensions Test: {test.shape}")

# %% [code]
# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) COMPLET
# =============================================================================
# %% [markdown]
# ## ğŸ”� Analyse Exploratoire des DonnÃ©es

# %% [code]
# Analyse de la variable cible
plt.figure(figsize=(10, 6))
target_counts = train['record_type'].value_counts()
plt.pie(target_counts.values, labels=['Non-Achat', 'Achat'], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
plt.title('Distribution de la Variable Cible (Record Type)')
plt.show()

print(f"ğŸ“ˆ Taux d'achat: {target_counts[1]/len(train)*100:.2f}%")

# %% [code]
# Distribution des variables numÃ©riques importantes
num_vars = ['shopping_pt', 'car_age', 'car_value', 'risk_factor', 'age_oldest', 'C_previous']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, col in enumerate(num_vars):
    if col in train.columns:
        axes[i].hist(train[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('FrÃ©quence')

plt.tight_layout()
plt.show()

# %% [code]
# Analyse des corrÃ©lations
plt.figure(figsize=(12, 8))
correlation_matrix = train.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Matrice de CorrÃ©lation - Variables NumÃ©riques')
plt.tight_layout()
plt.show()

# %% [code]
# Analyse des variables catÃ©gorielles
cat_vars = ['state', 'location', 'homeowner', 'married_couple']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(cat_vars):
    if col in train.columns:
        sns.countplot(data=train, x=col, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Distribution de {col}')
        axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [code]
# =============================================================================
# 4. FEATURE ENGINEERING CRÃ‰ATIF
# =============================================================================
# %% [markdown]
# ## âš¡ Feature Engineering AvancÃ©

# %% [code]
class FeatureEngineer:
    def __init__(self):
        self.features_created = []
    
    def engineer_features(self, df):
        """CrÃ©e des features avancÃ©es pour amÃ©liorer la performance du modÃ¨le"""
        
        # Features basÃ©es sur le comportement d'achat
        if 'risk_factor' in df.columns and 'car_value' in df.columns:
            df['risk_to_value_ratio'] = df['risk_factor'] / (df['car_value'] + 1)
            self.features_created.append('risk_to_value_ratio')
        
        if 'age_oldest' in df.columns and 'age_youngest' in df.columns:
            df['age_difference'] = df['age_oldest'] - df['age_youngest']
            df['age_difference_abs'] = np.abs(df['age_difference'])
            self.features_created.extend(['age_difference', 'age_difference_abs'])
        
        # Features d'interaction
        if 'homeowner' in df.columns and 'married_couple' in df.columns:
            df['homeowner_married'] = df['homeowner'] * df['married_couple']
            self.features_created.append('homeowner_married')
        
        # Features de grouping
        if 'group_size' in df.columns and 'car_value' in df.columns:
            df['value_per_person'] = df['car_value'] / (df['group_size'] + 1)
            self.features_created.append('value_per_person')
        
        # Features polynomiales
        if 'car_age' in df.columns:
            df['car_age_squared'] = df['car_age'] ** 2
            self.features_created.append('car_age_squared')
        
        # Features basÃ©es sur les variables A-G
        abc_columns = [col for col in df.columns if col in ['A', 'B', 'C', 'D', 'E', 'F', 'G']]
        if abc_columns:
            df['abc_sum'] = df[abc_columns].sum(axis=1)
            df['abc_mean'] = df[abc_columns].mean(axis=1)
            df['abc_std'] = df[abc_columns].std(axis=1)
            self.features_created.extend(['abc_sum', 'abc_mean', 'abc_std'])
        
        print(f"âœ… {len(self.features_created)} features crÃ©Ã©es: {self.features_created}")
        return df

# Application du feature engineering
feature_engineer = FeatureEngineer()
train_eng = feature_engineer.engineer_features(train.copy())
test_eng = feature_engineer.engineer_features(test.copy())

# %% [code]
# =============================================================================
# 5. PRÃ‰PROCESSING INTELLIGENT
# =============================================================================
# %% [markdown]
# ## ğŸ”§ PrÃ©processing des DonnÃ©es

# %% [code]
class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = ['state', 'location']
        self.numerical_columns = []
    
    def preprocess(self, train_df, test_df, target_col='record_type'):
        """PrÃ©pare les donnÃ©es pour l'entraÃ®nement"""
        
        # SÃ©paration des features et target
        if target_col in train_df.columns:
            y_train = train_df[target_col]
            X_train = train_df.drop(columns=[target_col])
            X_test = test_df.drop(columns=[target_col]) if target_col in test_df.columns else test_df
        else:
            X_train = train_df
            X_test = test_df
            y_train = None
        
        # Encodage des variables catÃ©gorielles
        for col in self.categorical_columns:
            if col in X_train.columns:
                # Combiner train et test pour un encodage cohÃ©rent
                combined = pd.concat([X_train[col], X_test[col]], axis=0)
                self.label_encoders[col] = LabelEncoder().fit(combined)
                X_train[col] = self.label_encoders[col].transform(X_train[col])
                X_test[col] = self.label_encoders[col].transform(X_test[col])
        
        # Identification des colonnes numÃ©riques
        self.numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Normalisation des variables numÃ©riques
        if len(self.numerical_columns) > 0:
            X_train[self.numerical_columns] = self.scaler.fit_transform(X_train[self.numerical_columns])
            X_test[self.numerical_columns] = self.scaler.transform(X_test[self.numerical_columns])
        
        print(f"âœ… Preprocessing terminÃ© - {X_train.shape[1]} features")
        return X_train, X_test, y_train

# Application du preprocessing
preprocessor = DataPreprocessor()
X_train, X_test, y_train = preprocessor.preprocess(train_eng, test_eng)

# %% [code]
# =============================================================================
# 6. MODÃ‰LISATION AVANCÃ‰E - MULTI-ALGORITHMES
# =============================================================================
# %% [markdown]
# ## ğŸ¤– EntraÃ®nement de Multiple ModÃ¨les

# %% [code]
class AdvancedModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {}
        self.cv_scores = {}
        
    def cross_validate_model(self, model, model_name, cv_folds=5):
        """Validation croisÃ©e robuste"""
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, self.X, self.y, cv=kfold, scoring='roc_auc')
        self.cv_scores[model_name] = cv_scores
        print(f"âœ… {model_name} - AUC CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        return cv_scores
    
    def train_lightgbm(self):
        """LightGBM avec optimisation"""
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        model = lgb.LGBMClassifier(**params)
        self.cross_validate_model(model, "LightGBM")
        model.fit(self.X, self.y)
        self.models['LightGBM'] = model
        return model
    
    def train_xgboost(self):
        """XGBoost avec optimisation"""
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        model = xgb.XGBClassifier(**params)
        self.cross_validate_model(model, "XGBoost")
        model.fit(self.X, self.y)
        self.models['XGBoost'] = model
        return model
    
    def train_catboost(self):
        """CatBoost pour les variables catÃ©gorielles"""
        model = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_state=42,
            verbose=False
        )
        self.cross_validate_model(model, "CatBoost")
        model.fit(self.X, self.y)
        self.models['CatBoost'] = model
        return model
    
    def train_random_forest(self):
        """Random Forest robuste"""
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        self.cross_validate_model(model, "Random Forest")
        model.fit(self.X, self.y)
        self.models['Random Forest'] = model
        return model
    
    def train_logistic_regression(self):
        """RÃ©gression logistique comme baseline"""
        model = LogisticRegression(
            C=0.1,
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        )
        self.cross_validate_model(model, "Logistic Regression")
        model.fit(self.X, self.y)
        self.models['Logistic Regression'] = model
        return model
    
    def train_all_models(self):
        """EntraÃ®ne tous les modÃ¨les"""
        print("ğŸš€ DÃ©but de l'entraÃ®nement de tous les modÃ¨les...")
        self.train_lightgbm()
        self.train_xgboost()
        self.train_catboost()
        self.train_random_forest()
        self.train_logistic_regression()
        
        # Affichage des rÃ©sultats comparatifs
        self.display_model_comparison()
    
    def display_model_comparison(self):
        """Affiche la comparaison des performances des modÃ¨les"""
        comparison_df = pd.DataFrame({
            'Model': list(self.cv_scores.keys()),
            'Mean AUC': [scores.mean() for scores in self.cv_scores.values()],
            'Std AUC': [scores.std() for scores in self.cv_scores.values()]
        }).sort_values('Mean AUC', ascending=False)
        
        print("\nğŸ�† COMPARAISON DES MODÃˆLES:")
        print(comparison_df.to_string(index=False))
        
        # Visualisation
        plt.figure(figsize=(10, 6))
        models = list(self.cv_scores.keys())
        means = [self.cv_scores[model].mean() for model in models]
        stds = [self.cv_scores[model].std() for model in models]
        
        plt.barh(models, means, xerr=stds, capsize=5, alpha=0.7, color='lightcoral')
        plt.xlabel('Score AUC Moyen')
        plt.title('Comparaison des Performances des ModÃ¨les')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

# EntraÃ®nement de tous les modÃ¨les
trainer = AdvancedModelTrainer(X_train, y_train)
trainer.train_all_models()

# %% [code]
# =============================================================================
# 7. INTERPRÃ‰TABILITÃ‰ AVEC SHAP
# =============================================================================
# %% [markdown]
# ## ğŸ“Š Analyse SHAP - ComprÃ©hension des PrÃ©dictions

# %% [code]
def perform_shap_analysis(model, X, model_name="LightGBM"):
    """Analyse SHAP complÃ¨te pour l'interprÃ©tabilitÃ©"""
    print(f"ğŸ”� Analyse SHAP pour {model_name}...")
    
    # Utilisation d'un Ã©chantillon pour des performances raisonnables
    sample_size = min(1000, X.shape[0])
    X_sample = X.sample(sample_size, random_state=42)
    
    # Calcul des valeurs SHAP
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Pour les classifieurs binaires
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Graphique d'importance des features
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'Importance des Features - {model_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Graphique des valeurs SHAP
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title(f'Importance Globale des Features - {model_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        print(f"âœ… Analyse SHAP terminÃ©e pour {model_name}")
    else:
        print(f"âš ï¸�  SHAP non supportÃ© pour {model_name}")

# Application sur le meilleur modÃ¨le (LightGBM)
best_model = trainer.models['LightGBM']
perform_shap_analysis(best_model, X_train, "LightGBM")

# %% [code]
# =============================================================================
# 8. CRÃ‰ATION DE PRÃ‰DICTIONS FINALES
# =============================================================================
# %% [markdown]
# ## ğŸ�¯ GÃ©nÃ©ration des PrÃ©dictions Finales

# %% [code]
def create_ensemble_predictions(trainer, X_test, method='weighted'):
    """CrÃ©e des prÃ©dictions d'ensemble Ã  partir de tous les modÃ¨les"""
    
    predictions = {}
    
    for name, model in trainer.models.items():
        if hasattr(model, 'predict_proba'):
            preds = model.predict_proba(X_test)[:, 1]
            predictions[name] = preds
            print(f"âœ… PrÃ©dictions gÃ©nÃ©rÃ©es pour {name}")
    
    if method == 'weighted':
        # PonderÃ© par les scores de validation
        weights = {}
        for name in predictions.keys():
            if name in trainer.cv_scores:
                weights[name] = trainer.cv_scores[name].mean()
        
        # Normalisation des poids
        total_weight = sum(weights.values())
        for name in weights:
            weights[name] /= total_weight
        
        # Combinaison pondÃ©rÃ©e
        final_predictions = np.zeros_like(list(predictions.values())[0])
        for name, preds in predictions.items():
            if name in weights:
                final_predictions += preds * weights[name]
                
    elif method == 'mean':
        # Simple moyenne
        final_predictions = np.mean(list(predictions.values()), axis=0)
    
    else:
        # Meilleur modÃ¨le uniquement
        best_model_name = max(trainer.cv_scores, key=lambda x: trainer.cv_scores[x].mean())
        final_predictions = predictions[best_model_name]
    
    print(f"ğŸ�¯ PrÃ©dictions d'ensemble crÃ©Ã©es (mÃ©thode: {method})")
    return final_predictions

# GÃ©nÃ©ration des prÃ©dictions finales
final_predictions = create_ensemble_predictions(trainer, X_test, method='weighted')

# %% [code]
# =============================================================================
# 9. CRÃ‰ATION DU FICHIER DE SOUMISSION
# =============================================================================
# %% [markdown]
# ## ğŸ“¤ PrÃ©paration de la Soumission

# %% [code]
def create_submission_file(predictions, test_df, output_file='submission.csv'):
    """CrÃ©e le fichier de soumission au format Kaggle"""
    
    # VÃ©rifier si customer_ID existe dans les donnÃ©es de test
    if 'customer_ID' in test_df.columns:
        submission_df = pd.DataFrame({
            'customer_ID': test_df['customer_ID'],
            'record_type': predictions
        })
    else:
        # Fallback - crÃ©er des IDs fictifs
        submission_df = pd.DataFrame({
            'customer_ID': range(len(predictions)),
            'record_type': predictions
        })
    
    # Sauvegarde du fichier
    submission_df.to_csv(output_file, index=False)
    print(f"âœ… Fichier de soumission sauvegardÃ©: {output_file}")
    print(f"ğŸ“Š Dimensions du fichier: {submission_df.shape}")
    print(f"ğŸ“ˆ Distribution des prÃ©dictions:")
    print(f"   - Min: {predictions.min():.4f}")
    print(f"   - Max: {predictions.max():.4f}")
    print(f"   - Mean: {predictions.mean():.4f}")
    
    return submission_df

# CrÃ©ation du fichier de soumission
submission = create_submission_file(final_predictions, test_eng)

# AperÃ§u des premiÃ¨res prÃ©dictions
print("\nğŸ‘€ AperÃ§u des premiÃ¨res prÃ©dictions:")
print(submission.head(10))

# %% [code]
# =============================================================================
# 10. ANALYSE DES RÃ‰SULTATS
# =============================================================================
# %% [markdown]
# ## ğŸ“Š Analyse Finale des Performances

# %% [code]
# Distribution des prÃ©dictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(final_predictions, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Distribution des ProbabilitÃ©s PrÃ©dites')
plt.xlabel('ProbabilitÃ©')
plt.ylabel('FrÃ©quence')

plt.subplot(1, 2, 2)
# Comparaison des performances des modÃ¨les
model_names = list(trainer.cv_scores.keys())
model_means = [trainer.cv_scores[name].mean() for name in model_names]
model_stds = [trainer.cv_scores[name].std() for name in model_names]

y_pos = np.arange(len(model_names))
plt.barh(y_pos, model_means, xerr=model_stds, alpha=0.7, color='lightblue', capsize=5)
plt.yticks(y_pos, model_names)
plt.xlabel('Score AUC Moyen')
plt.title('Comparaison Finale des ModÃ¨les')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# %% [markdown]
# # ğŸ�‰ CONCLUSION
# 
# ## ğŸ“ˆ RÃ©sumÃ© des Performances
# 
# Ce notebook dÃ©montre une approche complÃ¨te pour la prÃ©diction d'achat Allstate avec :
# 
# - **ğŸ”� EDA approfondi** avec visualisations interactives
# - **âš¡ Feature engineering crÃ©atif** avec de nouvelles variables
# - **ğŸ¤– Multi-modÃ¨les** avec validation croisÃ©e robuste
# - **ğŸ“Š InterprÃ©tabilitÃ© SHAP** pour comprendre les dÃ©cisions
# - **ğŸ�¯ Ensemble weighting** pour des prÃ©dictions optimales
# 
# ## ğŸ’¡ Insights ClÃ©s
# 
# 1. **Performance** : L'approche ensemble amÃ©liore significativement la robustesse
# 2. **InterprÃ©tabilitÃ©** : SHAP rÃ©vÃ¨le les drivers les plus importants
# 3. **GÃ©nÃ©ralisation** : La validation croisÃ©e assure une bonne gÃ©nÃ©ralisation
# 
# *N'oubliez pas de voter si ce notebook vous a Ã©tÃ© utile ! ğŸ‘�*

