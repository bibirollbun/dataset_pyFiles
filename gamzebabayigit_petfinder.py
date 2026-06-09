import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        (os.path.join(dirname, filename))
 


train = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/test.csv')
submission = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/sample_submission.csv')
 


train.shape


test.shape


submission.shape


train.head()


train.tail()


test.head()


submission.head()


train.info()


train.describe().T


train.isnull().sum()


def feature_engineering(df):
    df = df.copy()
    # Sağlık özellikleri - çarpan olarak
    df['Health_Score'] = (df['Vaccinated'] == 1).astype(int) + \
                        (df['Dewormed'] == 1).astype(int) + \
                        (df['Sterilized'] == 1).astype(int)

    # Yaş
    df['Age_In_Years'] = df['Age'] / 12
    df['Is_Baby'] = (df['Age'] <= 3).astype(int)
    df['Is_Young'] = ((df['Age'] > 3) & (df['Age'] <= 12)).astype(int)
    df['Is_Adult'] = ((df['Age'] > 12) & (df['Age'] <= 60)).astype(int)
    df['Is_Senior'] = (df['Age'] > 60).astype(int)


    df['Is_Free'] = (df['Fee'] == 0).astype(int)
    df['Is_Cheap'] = ((df['Fee'] > 0) & (df['Fee'] <= 100)).astype(int)
    df['Is_Expensive'] = (df['Fee'] > 100).astype(int)
    df['Fee_Per_Month'] = df['Fee'] / (df['Age'] + 1)


    df['Type_Gender'] = df['Type'].astype(str) + '_' + df['Gender'].astype(str)



    for col in ['Breed1', 'Breed2', 'Color1', 'State']:
      if col in df.columns:
        grouped = train.groupby(col)['AdoptionSpeed'].agg(['mean', 'std', 'count']) if 'AdoptionSpeed' in train.columns else None
        if grouped is not None:
            df[f'{col}_AdoptSpeed_Mean'] = df[col].map(grouped['mean'])
            df[f'{col}_AdoptSpeed_Std'] = df[col].map(grouped['std']).fillna(0)
    return df



train = feature_engineering(train)
test = feature_engineering(test)


train.head()


test.head()


# Kategorik ve sayısal özellikleri ayır
categorical_features = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
                        'Color3', 'MaturitySize', 'FurLength', 'Vaccinated',
                        'Dewormed', 'Sterilized', 'Health', 'State']



# ID ve target dışındaki tüm sütunlar feature olacak
text_features = ['Name', 'Description', 'RescuerID']
feature_cols = [col for col in train.columns
                if col not in ['PetID', 'AdoptionSpeed', 'Name', 'Description', 'RescuerID']]



X = train[feature_cols]
y = train['AdoptionSpeed']
X_test = test[feature_cols]


# Toplam özellik sayısı
len(feature_cols)



# LightGBM parametreleri
params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1
}


# Stratified K-Fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X))
test_predictions = np.zeros((len(X_test), n_splits))
feature_importance_df = pd.DataFrame()
kappa_scores = []


X["Type_Gender"] = X["Type_Gender"].astype("category")
X_test["Type_Gender"] = X_test["Type_Gender"].astype("category")


for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{n_splits}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Dataset oluştur
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)

    # Model eğit
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )

    # OOF tahminleri
    oof_preds = model.predict(X_valid, num_iteration=model.best_iteration)
    oof_predictions[valid_idx] = np.argmax(oof_preds, axis=1)

    # Test tahminleri
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    test_predictions[:, fold-1] = np.argmax(test_preds, axis=1)

    # Kappa skoru hesapla
    kappa = cohen_kappa_score(y_valid, np.argmax(oof_preds, axis=1), weights='quadratic')
    kappa_scores.append(kappa)
    print(f"Fold {fold} Quadratic Kappa: {kappa:.4f}\n")

    # Feature importance
    fold_importance = pd.DataFrame()
    fold_importance['feature'] = feature_cols
    fold_importance['importance'] = model.feature_importance(importance_type='gain')
    fold_importance['fold'] = fold
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)



overall_kappa = cohen_kappa_score(y, oof_predictions, weights='quadratic')
print(f"GENEL SONUÇLAR")
print(f"OOF Quadratic Kappa: {overall_kappa:.4f}")
print(f"CV Kappa (ortalama): {np.mean(kappa_scores):.4f} (+/- {np.std(kappa_scores):.4f})")



# Ortalama feature importance
mean_importance = feature_importance_df.groupby('feature')['importance'].mean().reset_index()
mean_importance = mean_importance.sort_values('importance', ascending=False)

print("\nEn Önemli 20 Özellik:")
print(mean_importance.head(20).to_string(index=False))



plt.figure(figsize=(10, 12))
top_features = mean_importance.head(30)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 30 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()



# Test tahminleri
final_predictions = np.round(test_predictions.mean(axis=1)).astype(int)

# Submission dosyası oluştur
submission['AdoptionSpeed'] = final_predictions
submission.to_csv('submission.csv', index=False)


