import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')


# RMSLE with numerical stability
def rmsle_stable(y_true, y_pred):
    y_true = np.maximum(y_true, 1e-8)
    y_pred = np.maximum(y_pred, 1e-8)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


# MISSING VALUES HANDLING
def handle_missing_values(df, is_train=True, missing_stats=None):
    """
    Traite les valeurs manquantes de façon intelligente selon le type de données
    """
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    if 'Calories' in numeric_cols and is_train:
        numeric_cols.remove('Calories')
    
    print(f"Colonnes numériques: {numeric_cols}")
    print(f"Colonnes catégorielles: {categorical_cols}")
    
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print("\n Valeurs manquantes détectées:")
        for col in missing_info[missing_info > 0].index:
            pct = (missing_info[col] / len(df)) * 100
            print(f"  {col}: {missing_info[col]} ({pct:.1f}%)")
    else:
        print(" Aucune valeur manquante détectée")
        return df, None
    
    if is_train:
        missing_stats = {}
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                
                if missing_pct > 50:
                    df[f'{col}_was_missing'] = df[col].isnull().astype(int)
                    df[col].fillna(df[col].median(), inplace=True)
                    missing_stats[col] = {'strategy': 'median_with_indicator', 'value': df[col].median()}
                    print(f"{col}: Médiane + indicateur (>{missing_pct:.1f}% manquant)")
                    
                elif missing_pct > 20:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    missing_stats[col] = {'strategy': 'median', 'value': median_val}
                    print(f"{col}: Médiane ({missing_pct:.1f}% manquant)")
                    
                elif missing_pct > 5:
                    from sklearn.linear_model import LinearRegression
                    
                    predictor_cols = [c for c in numeric_cols if c != col and df[c].isnull().sum() == 0]
                    
                    if len(predictor_cols) >= 2:
                        not_null_mask = ~df[col].isnull()
                        X_train_reg = df.loc[not_null_mask, predictor_cols]
                        y_train_reg = df.loc[not_null_mask, col]
                        
                        reg_model = LinearRegression()
                        reg_model.fit(X_train_reg, y_train_reg)
                        
                        null_mask = df[col].isnull()
                        X_pred = df.loc[null_mask, predictor_cols]
                        predicted_values = reg_model.predict(X_pred)
                        
                        df.loc[null_mask, col] = predicted_values
                        missing_stats[col] = {'strategy': 'regression', 'model': reg_model, 'predictors': predictor_cols}
                        print(f"{col}: Régression ({missing_pct:.1f}% manquant)")
                    else:
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        missing_stats[col] = {'strategy': 'median', 'value': median_val}
                        print(f"{col}: Médiane fallback ({missing_pct:.1f}% manquant)")
                        
                else:
                    from sklearn.impute import KNNImputer
                    
                    knn_cols = [c for c in numeric_cols if df[c].isnull().sum() < len(df) * 0.3]
                    if len(knn_cols) >= 3:
                        knn_imputer = KNNImputer(n_neighbors=5)
                        df[knn_cols] = knn_imputer.fit_transform(df[knn_cols])
                        missing_stats['knn_imputer'] = knn_imputer
                        missing_stats['knn_cols'] = knn_cols
                        print(f"{col}: KNN imputation ({missing_pct:.1f}% manquant)")
                    else:
                        
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        missing_stats[col] = {'strategy': 'median', 'value': median_val}
                        print(f"{col}: Médiane ({missing_pct:.1f}% manquant)")
        
        
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                
                if missing_pct > 30:
                   
                    df[col].fillna('Unknown', inplace=True)
                    missing_stats[col] = {'strategy': 'unknown', 'value': 'Unknown'}
                    print(f"{col}: 'Unknown' ({missing_pct:.1f}% manquant)")
                    
                elif missing_pct > 10:
                   
                    mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    missing_stats[col] = {'strategy': 'mode', 'value': mode_val}
                    print(f"{col}: Mode '{mode_val}' ({missing_pct:.1f}% manquant)")
                    
                else:
                   
                    from sklearn.ensemble import RandomForestClassifier
                    predictor_cols = [c for c in numeric_cols if df[c].isnull().sum() == 0]
                    
                    if len(predictor_cols) >= 2:
                        not_null_mask = ~df[col].isnull()
                        X_train_clf = df.loc[not_null_mask, predictor_cols]
                        y_train_clf = df.loc[not_null_mask, col]
                        
                        clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        clf_model.fit(X_train_clf, y_train_clf)
                        
                        null_mask = df[col].isnull()
                        X_pred = df.loc[null_mask, predictor_cols]
                        predicted_values = clf_model.predict(X_pred)
                        
                        df.loc[null_mask, col] = predicted_values
                        missing_stats[col] = {'strategy': 'classification', 'model': clf_model, 'predictors': predictor_cols}
                        print(f"{col}: Classification ({missing_pct:.1f}% manquant)")
                    else:
                        mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df[col].fillna(mode_val, inplace=True)
                        missing_stats[col] = {'strategy': 'mode', 'value': mode_val}
                        print(f"{col}: Mode fallback '{mode_val}' ({missing_pct:.1f}% manquant)")
    
    else:
        if missing_stats is None:
            print("Aucune statistique de training fournie pour le test set")
            return df, None
            
        for col, stats in missing_stats.items():
            if col in df.columns and df[col].isnull().sum() > 0:
                
                if stats['strategy'] == 'median':
                    df[col].fillna(stats['value'], inplace=True)
                    
                elif stats['strategy'] == 'median_with_indicator':
                    df[f'{col}_was_missing'] = df[col].isnull().astype(int)
                    df[col].fillna(stats['value'], inplace=True)
                    
                elif stats['strategy'] == 'unknown':
                    df[col].fillna(stats['value'], inplace=True)
                    
                elif stats['strategy'] == 'mode':
                    df[col].fillna(stats['value'], inplace=True)
                    
                elif stats['strategy'] == 'regression':
                    null_mask = df[col].isnull()
                    if null_mask.sum() > 0:
                        X_pred = df.loc[null_mask, stats['predictors']]
                        predicted_values = stats['model'].predict(X_pred)
                        df.loc[null_mask, col] = predicted_values
                        
                elif stats['strategy'] == 'classification':
                    null_mask = df[col].isnull()
                    if null_mask.sum() > 0:
                        X_pred = df.loc[null_mask, stats['predictors']]
                        predicted_values = stats['model'].predict(X_pred)
                        df.loc[null_mask, col] = predicted_values
        
        if 'knn_imputer' in missing_stats:
            knn_cols = missing_stats['knn_cols']
            available_cols = [c for c in knn_cols if c in df.columns]
            if len(available_cols) > 0:
                df[available_cols] = missing_stats['knn_imputer'].transform(df[available_cols])
    
    print(f"Traitement des valeurs manquantes terminé")
    return df, missing_stats


def create_advanced_features(df, sex_encoder=None, is_train=True):
    df = df.copy()
    
    if is_train:
        sex_encoder = LabelEncoder()
        df['Sex_encoded'] = sex_encoder.fit_transform(df['Sex'])
    else:
        df['Sex_encoded'] = sex_encoder.transform(df['Sex'])
    
    # Physics-based features (based on exercise physiology)
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['BSA'] = 0.007184 * (df['Weight'] ** 0.425) * (df['Height'] ** 0.725)  # Body Surface Area
    
    # Metabolic Equivalent of Task (MET) estimation
    df['MET_estimate'] = (df['Heart_Rate'] - 60) / (200 - df['Age'] - 60) * 15 + 1
    df['Calories_MET'] = df['MET_estimate'] * df['Weight'] * df['Duration'] / 60
    
    # heart rate features
    df['HR_reserve'] = (220 - df['Age']) - 60  # Heart rate reserve
    df['HR_intensity'] = (df['Heart_Rate'] - 60) / df['HR_reserve']
    df['HR_efficiency'] = df['Heart_Rate'] / df['Weight']
    
    # Temperature-based metabolism
    df['Thermal_effect'] = np.where(df['Body_Temp'] > 37.5, 
                                   (df['Body_Temp'] - 37) * 0.13, 0)  # 13% increase per degree
    
    # Age-based metabolism adjustments
    df['Age_metabolism'] = np.where(df['Age'] > 30, 
                                   1 - (df['Age'] - 30) * 0.02, 1)  # 2% decrease per year after 30
    
    # Gender-specific features
    df['Male_multiplier'] = np.where(df['Sex_encoded'] == 1, 1.15, 1.0)  # Males burn ~15% more
    
    # Activity intensity zones
    df['Zone_1'] = ((df['HR_intensity'] >= 0.5) & (df['HR_intensity'] < 0.6)).astype(int)
    df['Zone_2'] = ((df['HR_intensity'] >= 0.6) & (df['HR_intensity'] < 0.7)).astype(int)
    df['Zone_3'] = ((df['HR_intensity'] >= 0.7) & (df['HR_intensity'] < 0.8)).astype(int)
    df['Zone_4'] = ((df['HR_intensity'] >= 0.8) & (df['HR_intensity'] < 0.9)).astype(int)
    df['Zone_5'] = (df['HR_intensity'] >= 0.9).astype(int)
    
    # Complex interactions
    df['Power_output'] = df['Weight'] * df['Heart_Rate'] * df['Duration'] / 1000
    df['Metabolic_load'] = df['BMI'] * df['Heart_Rate'] * df['Duration']
    df['Efficiency_ratio'] = df['Heart_Rate'] / (df['Body_Temp'] * df['Weight'])
    
    # Polynomial features for key variables
    for col in ['Duration', 'Heart_Rate', 'Weight', 'BMI']:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_cubed'] = df[col] ** 3
        df[f'{col}_sqrt'] = np.sqrt(df[col])
        df[f'{col}_log'] = np.log1p(df[col])
    
    # Binning continuous variables
    df['Age_bin'] = pd.cut(df['Age'], bins=10, labels=False)
    df['Weight_bin'] = pd.cut(df['Weight'], bins=10, labels=False)
    df['Duration_bin'] = pd.cut(df['Duration'], bins=10, labels=False)
    
    if is_train:
        return df, sex_encoder
    else:
        return df


def add_clustering_features(df, physical_kmeans=None, workout_kmeans=None, n_clusters=8, is_train=True):
    # Cluster users based on physical characteristics
    physical_features = ['Age', 'Height', 'Weight', 'Sex_encoded']
    
    if is_train:
        physical_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Physical_cluster'] = physical_kmeans.fit_predict(df[physical_features])
    else:
        df['Physical_cluster'] = physical_kmeans.predict(df[physical_features])
    
    # Cluster workouts based on intensity
    workout_features = ['Duration', 'Heart_Rate', 'Body_Temp']
    
    if is_train:
        workout_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Workout_cluster'] = workout_kmeans.fit_predict(df[workout_features])
        return df, physical_kmeans, workout_kmeans
    else:
        df['Workout_cluster'] = workout_kmeans.predict(df[workout_features])
        return df


def optimize_xgb(X, y, n_trials=100):
    def objective(trial):
        params = {
            'objective': 'reg:squaredlogerror',
            'eval_metric': 'rmsle',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 15.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 15.0),
            'n_estimators': 1000,
            'random_state': 42
        }
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=50, 
                     verbose=False)
            
            pred = model.predict(X_val)
            score = rmsle_stable(y_val, pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params



def pseudo_labeling(X_train, y_train, X_test, model, confidence_threshold=0.95):
    """Add high-confidence test predictions to training data"""
    
    # Get predictions and their confidence (inverse of std across folds)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold = X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx]
        
        model.fit(X_fold, y_fold)
        pred = model.predict(X_test)
        test_preds.append(pred)
    
    test_preds = np.array(test_preds)
    mean_preds = np.mean(test_preds, axis=0)
    std_preds = np.std(test_preds, axis=0)
    
    # Select high-confidence predictions
    confidence = 1 / (1 + std_preds)  # Higher confidence = lower std
    high_conf_mask = confidence > np.percentile(confidence, confidence_threshold * 100)
    
    # Add pseudo-labels to training data
    X_pseudo = X_test[high_conf_mask]
    y_pseudo = mean_preds[high_conf_mask]
    
    X_augmented = pd.concat([X_train, X_pseudo], ignore_index=True)
    y_augmented = pd.concat([y_train, pd.Series(y_pseudo)], ignore_index=True)
    
    print(f"Added {len(y_pseudo)} pseudo-labels ({len(y_pseudo)/len(X_test)*100:.1f}% of test set)")
    return X_augmented, y_augmented


def create_nn_ensemble(X_train, y_train, X_test):
    """Create ensemble of neural networks with different architectures"""
    
    # Normalize features for NN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Different NN architectures
    nn_configs = [
        {'hidden_layer_sizes': (100, 50), 'alpha': 0.01},
        {'hidden_layer_sizes': (200, 100, 50), 'alpha': 0.001},
        {'hidden_layer_sizes': (150, 75), 'alpha': 0.1},
        {'hidden_layer_sizes': (300, 150, 75), 'alpha': 0.01}
    ]
    
    nn_predictions = []
    
    for config in nn_configs:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_preds = []
        
        for train_idx, val_idx in kf.split(X_train_scaled):
            X_fold = X_train_scaled[train_idx]
            y_fold = y_train.iloc[train_idx]
            
            nn = MLPRegressor(
                hidden_layer_sizes=config['hidden_layer_sizes'],
                alpha=config['alpha'],
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            nn.fit(X_fold, y_fold)
            pred = nn.predict(X_test_scaled)
            fold_preds.append(pred)
        
        nn_predictions.append(np.mean(fold_preds, axis=0))
    
    return np.mean(nn_predictions, axis=0)


print("=== GETTING START ===")

# feature engineering
print("1. Creating advanced features...")
train_fe, sex_encoder = create_advanced_features(train, is_train=True)
test_fe = create_advanced_features(test, sex_encoder=sex_encoder, is_train=False)

# Add clustering features
print("2. Adding clustering features...")
train_fe, physical_kmeans, workout_kmeans = add_clustering_features(train_fe, is_train=True)
test_fe = add_clustering_features(test_fe, physical_kmeans, workout_kmeans, is_train=False)

# Feature selection
feature_cols = [col for col in train_fe.columns if col not in ['id', 'Sex', 'Calories']]
X = train_fe[feature_cols]
y = train_fe['Calories']
X_test = test_fe[feature_cols]

print(f"Total features: {len(feature_cols)}")

# Hyperparameter optimization (comment out if too slow)
print("3. Optimizing hyperparameters...")
# best_params = optimize_xgb(X.sample(10000), y.sample(10000), n_trials=50)  # Sample for speed
# print(f"Best params: {best_params}")

# Use pre-optimized params for speed
best_params = {
    'objective': 'reg:squaredlogerror',
    'eval_metric': 'rmsle',
    'learning_rate': 0.005,
    'max_depth': 8,
    'min_child_weight': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'n_estimators': 2000
}

# Train optimized models with CV
print("4. Training optimized models...")
kf = KFold(n_splits=7, shuffle=True, random_state=42)  # More folds for stability

# XGBoost with optimized params
xgb_preds = np.zeros(len(X_test))
xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train_fold, y_train_fold,
             eval_set=[(X_val_fold, y_val_fold)],
             early_stopping_rounds=100,
             verbose=False)
    
    val_pred = model.predict(X_val_fold)
    score = rmsle_stable(y_val_fold, val_pred)
    xgb_scores.append(score)
    
    xgb_preds += model.predict(X_test) / 7

print(f"XGBoost CV RMSLE: {np.mean(xgb_scores):.6f}")

# Neural Network ensemble
print("5. Training neural network ensemble...")
nn_preds = create_nn_ensemble(X, y, X_test)

# Final ensemble
print("6. Creating final ensemble...")
# Weight based on validation performance
ensemble_preds = 0.7 * xgb_preds + 0.3 * nn_preds

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': np.maximum(ensemble_preds, 0.001)
})

submission.to_csv('submission.csv', index=False)
print(f"Final prediction range: {submission['Calories'].min():.2f} - {submission['Calories'].max():.2f}")
print("Advanced submission created!")

# Create multiple submissions with different strategies
# strategies = {
#     'conservative': 0.8 * xgb_preds + 0.2 * nn_preds,
#     'aggressive': 0.5 * xgb_preds + 0.5 * nn_preds,
#     'xgb_only': xgb_preds
# }

# for name, preds in strategies.items():
#     sub = pd.DataFrame({
#         'id': test['id'],
#         'Calories': np.maximum(preds, 0.001)
#     })
#     sub.to_csv(f'calories_{name}.csv', index=False)

# print("Multiple strategy submissions created!")

