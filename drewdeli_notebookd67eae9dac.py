import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

import optuna
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
sample = pd.read_csv('/kaggle/input/forest-cover-type-prediction/sampleSubmission.csv')
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


train.head()


train.info()


train.isnull().sum(), test.isnull().sum()


train['Cover_Type'].value_counts()


X = train.drop(['Id', 'Cover_Type'], axis=1)
y = train['Cover_Type']
test_data = test.drop('Id', axis=1)


soil_cols = [col for col in X.columns if 'Soil_Type' in col]
wilderness_cols = [col for col in X.columns if 'Wilderness_Area' in col]


X['Soil_Count'] = X[soil_cols].sum(axis=1)
X['Wilderness_Count'] = X[wilderness_cols].sum(axis=1)

test_data['Soil_Count'] = test_data[soil_cols].sum(axis=1)
test_data['Wilderness_Count'] = test_data[wilderness_cols].sum(axis=1)


X['Elevation_diff'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']
X['Hydro_Fire'] = X['Horizontal_Distance_To_Hydrology'] + X['Horizontal_Distance_To_Fire_Points']
X['Hydro_Road'] = X['Horizontal_Distance_To_Hydrology'] + X['Horizontal_Distance_To_Roadways']
X['Mean_Distance'] = (X['Horizontal_Distance_To_Hydrology'] + 
                      X['Horizontal_Distance_To_Roadways'] + 
                      X['Horizontal_Distance_To_Fire_Points']) / 3

test_data['Elevation_diff'] = test_data['Elevation'] - test_data['Vertical_Distance_To_Hydrology']
test_data['Hydro_Fire'] = test_data['Horizontal_Distance_To_Hydrology'] + test_data['Horizontal_Distance_To_Fire_Points']
test_data['Hydro_Road'] = test_data['Horizontal_Distance_To_Hydrology'] + test_data['Horizontal_Distance_To_Roadways']
test_data['Mean_Distance'] = (test_data['Horizontal_Distance_To_Hydrology'] + 
                              test_data['Horizontal_Distance_To_Roadways'] + 
                              test_data['Horizontal_Distance_To_Fire_Points']) / 3


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_data)


X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = XGBClassifier(**params)
    
    # Преобразуем метки для XGBoost
    y_train_adj = y_train - 1
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train_adj, cv=cv, scoring='accuracy', n_jobs=-1)
    return score.mean()
    
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)
best_params = study.best_params
print(f"Best params: {best_params}")
print(f"Best score: {study.best_value}")


xgb_model = XGBClassifier(**best_params, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train - 1)

y_pred_xgb = xgb_model.predict(X_val) + 1
accuracy_xgb = accuracy_score(y_val, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")


lgb_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'num_leaves': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train, y_train - 1)

y_pred_lgb = lgb_model.predict(X_val) + 1
accuracy_lgb = accuracy_score(y_val, y_pred_lgb)
print(f"LightGBM Accuracy: {accuracy_lgb:.4f}")


rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_val)
accuracy_rf = accuracy_score(y_val, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")


cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    verbose=0,
    random_state=42
)
cat_model.fit(X_train, y_train - 1)

y_pred_cat = cat_model.predict(X_val) + 1
accuracy_cat = accuracy_score(y_val, y_pred_cat)
print(f"CatBoost Accuracy: {accuracy_cat:.4f}")


from sklearn.ensemble import VotingClassifier

voting_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model),  # RandomForest не требует преобразования
        ('cat', cat_model)
    ],
    voting='soft',
    n_jobs=-1
)

class AdjustedVotingClassifier:
    def __init__(self, estimators, voting='soft'):
        self.estimators = estimators
        self.voting = voting
        
    def fit(self, X, y):
        # Преобразуем метки для моделей, которые требуют этого
        y_adj = y - 1
        for name, model in self.estimators:
            if name in ['xgb', 'lgb', 'cat']:
                model.fit(X, y_adj)
            else:
                model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = []
        for name, model in self.estimators:
            pred = model.predict(X)
            if name in ['xgb', 'lgb', 'cat']:
                pred = pred + 1
            predictions.append(pred)
        
        # Простое голосование большинством
        predictions = np.array(predictions)
        final_pred = []
        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            final_pred.append(np.bincount(votes.astype(int)).argmax())
        
        return np.array(final_pred)

xgb_wrapped = XGBClassifier(**best_params, random_state=42)
lgb_wrapped = lgb.LGBMClassifier(**lgb_params)
rf_wrapped = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)
cat_wrapped = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_state=42)

xgb_wrapped.fit(X_train, y_train - 1)
lgb_wrapped.fit(X_train, y_train - 1)
rf_wrapped.fit(X_train, y_train)
cat_wrapped.fit(X_train, y_train - 1)

y_pred_xgb = xgb_wrapped.predict(X_val).ravel() + 1
y_pred_lgb = lgb_wrapped.predict(X_val).ravel() + 1
y_pred_rf = rf_wrapped.predict(X_val).ravel()
y_pred_cat = cat_wrapped.predict(X_val).ravel() + 1

predictions = np.array([y_pred_xgb, y_pred_lgb, y_pred_rf, y_pred_cat])
final_pred = []
for i in range(predictions.shape[1]):
    votes = predictions[:, i]
    final_pred.append(np.bincount(votes.astype(int)).argmax())

final_pred = np.array(final_pred)
accuracy_vote = accuracy_score(y_val, final_pred)
print(f"Voting Classifier Accuracy: {accuracy_vote:.4f}")


xgb_final = XGBClassifier(**best_params, random_state=42)
lgb_final = lgb.LGBMClassifier(**lgb_params)
rf_final = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)
cat_final = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_state=42)

xgb_final.fit(X_scaled, y - 1)
lgb_final.fit(X_scaled, y - 1)
rf_final.fit(X_scaled, y)
cat_final.fit(X_scaled, y - 1)


test_pred_xgb = xgb_final.predict(test_scaled).reshape(-1) + 1
test_pred_lgb = lgb_final.predict(test_scaled).reshape(-1) + 1
test_pred_rf = rf_final.predict(test_scaled).reshape(-1)
test_pred_cat = cat_final.predict(test_scaled).reshape(-1) + 1

# Проверка shapes
print(f"XGB shape: {test_pred_xgb.shape}")
print(f"LGB shape: {test_pred_lgb.shape}")
print(f"RF shape: {test_pred_rf.shape}")
print(f"Cat shape: {test_pred_cat.shape}")

assert test_pred_xgb.shape == test_pred_lgb.shape == test_pred_rf.shape == test_pred_cat.shape


test_predictions = np.vstack([test_pred_xgb, test_pred_lgb, test_pred_rf, test_pred_cat])
print(f"Test predictions shape: {test_predictions.shape}")

final_test_pred = []
for i in range(test_predictions.shape[1]):
    votes = test_predictions[:, i]
    final_test_pred.append(np.bincount(votes.astype(int)).argmax())

final_test_pred = np.array(final_test_pred)
print(f"Final predictions shape: {final_test_pred.shape}")
print(f"Unique values: {np.unique(final_test_pred)}")


submission = pd.DataFrame({
    'Id': test['Id'],
    'Cover_Type': final_test_pred
})

submission.to_csv('submission.csv', index=False)
print("saved")
print(submission.head())

