import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

# 1. Chargement des données
def load_data():
    train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
    sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
    
    # Chargement des données supplémentaires
    extra_data_paths = [
        "/kaggle/input/hex-s5e6/FertilizerPrediction1.csv",
        "/kaggle/input/hex-s5e6/FertilizerPrediction2.csv"
    ]
    
    for path in extra_data_paths:
        if os.path.exists(path):
            extra_data = pd.read_csv(path)
            train = pd.concat([train, extra_data], ignore_index=True)
    
    return train, test, sub

# 2. Préparation des données
def prepare_data(train, test):
    # Encodage des variables catégorielles
    cat_cols = ["Soil Type", "Crop Type"]
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train[cat_cols] = enc.fit_transform(train[cat_cols])
    test[cat_cols] = enc.transform(test[cat_cols])
    
    # Encodage de la target
    le = LabelEncoder()
    y = le.fit_transform(train["Fertilizer Name"])
    X = train.drop(["id", "Fertilizer Name"], axis=1)
    X_test = test.drop("id", axis=1)
    
    return X, y, X_test, le

# 3. Fonction d'évaluation MAP@3
def map_at_3(y_true, y_pred_proba):
    ap_scores = []
    for true, proba in zip(y_true, y_pred_proba):
        top3 = np.argsort(proba)[-3:][::-1]
        precision = 0.0
        correct = 0
        for i, pred in enumerate(top3):
            if pred == true:
                correct += 1
                precision += correct / (i + 1)
        ap = precision / min(3, correct) if correct > 0 else 0.0
        ap_scores.append(ap)
    return np.mean(ap_scores)

# 4. Paramètres optimisés pour XGBoost
params = {
    'objective': 'multi:softprob',
    'num_class': 7,
    'tree_method': 'hist',  # Plus rapide pour les grands datasets (si j'ai bien compris)
    'learning_rate': 0.1,   # Plus agressif
    'max_depth': 8,         # Un peu plus profond
    'min_child_weight': 3,  # Plus permissif
    'gamma': 0.2,           # Regularisation supplémentaire
    'subsample': 0.9,       # Stochastic boosting
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,      # L2 regularization plus forte
    'n_estimators': 2000,   # Plus d'arbres (early stopping s'occupera du reste)
    'early_stopping_rounds': 100,
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'enable_categorical': False  # Important avec OrdinalEncoder
}

# Pipeline principal
def main():
    # Chargement des données
    print("Chargement des données...")
    train, test, sub = load_data()
    
    # Préparation des données
    print("Préparation des données...")
    X, y, X_test, le = prepare_data(train, test)
    
    # Split des données
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    # Entraînement du modèle XGBoost
    print("Entraînement du modèle...")
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Évaluation
    y_proba_val = model.predict_proba(X_val)
    val_map = map_at_3(y_val, y_proba_val)
    print(f"\nValidation MAP@3: {val_map:.4f}")
    
    # Prédiction finale avec légère diversification
    y_proba_test = model.predict_proba(X_test)
    # Ajout d'un petit bruit pour diversifier les prédictions
    y_proba_test = y_proba_test * (1 + 0.01 * np.random.randn(*y_proba_test.shape))
    
    top3_preds = [
        " ".join(le.inverse_transform(np.argsort(proba)[-3:][::-1])) 
        for proba in y_proba_test
    ]
    sub["Fertilizer Name"] = top3_preds
    sub.to_csv("submission.csv", index=False)
    
    print("\nTop 5 prédictions:")
    print(sub.head())
    print(f"\nScore final MAP@3: {val_map:.4f}")

if __name__ == "__main__":
    main()

