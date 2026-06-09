import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')
np.random.seed(42)


def comprehensive_eda(df, name="Train"):
    print(f"\n{'='*50}")
    print(f"EDA PARA {name.upper()} (Shape: {df.shape})")
    print('='*50)
    
    # AnÃ¡lisis bÃ¡sico
    print("\nğŸ”� InformaciÃ³n bÃ¡sica:")
    print(df.info())
    
    print("\nğŸ“Š EstadÃ­sticas descriptivas:")
    print(df.describe(include='all').T)
    
    # AnÃ¡lisis de valores faltantes
    print("\nâ�“ Valores faltantes:")
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.concat([missing, missing_pct], axis=1, keys=['Total', '%'])
    print(missing_df[missing_df['Total'] > 0])
    
    # VisualizaciÃ³n de valores faltantes
    if missing.sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title(f'DistribuciÃ³n de Valores Faltantes - {name}')
        plt.show()
    
    # DistribuciÃ³n de variables numÃ©ricas
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        print("\nğŸ“ˆ DistribuciÃ³n de variables numÃ©ricas:")
        fig, axes = plt.subplots(nrows=len(num_cols), ncols=2, figsize=(14, 5*len(num_cols)))
        for i, col in enumerate(num_cols):
            # Histograma y KDE
            sns.histplot(df[col], kde=True, ax=axes[i,0], color='skyblue')
            axes[i,0].set_title(f'DistribuciÃ³n de {col}')
            
            # Boxplot
            sns.boxplot(x=df[col], ax=axes[i,1], color='lightgreen')
            axes[i,1].set_title(f'Boxplot de {col}')
            
        plt.tight_layout()
        plt.show()
    
    # DistribuciÃ³n de variables categÃ³ricas
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print("\nğŸ“Š DistribuciÃ³n de variables categÃ³ricas:")
        fig, axes = plt.subplots(nrows=len(cat_cols), figsize=(10, 4*len(cat_cols)))
        for i, col in enumerate(cat_cols):
            # GrÃ¡fico de barras
            sns.countplot(x=df[col], ax=axes[i], palette='viridis')
            axes[i].set_title(f'DistribuciÃ³n de {col}')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Anotar porcentajes
            total = len(df[col])
            for p in axes[i].patches:
                percentage = f'{100 * p.get_height()/total:.1f}%'
                x = p.get_x() + p.get_width() / 2
                y = p.get_height() + 0.02
                axes[i].annotate(percentage, (x, y), ha='center')
                
        plt.tight_layout()
        plt.show()
    
    # Correlaciones
    if len(num_cols) > 1:
        print("\nğŸ§© Matriz de correlaciÃ³n:")
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                   mask=np.triu(np.ones_like(corr, dtype=bool)))
        plt.title(f'Matriz de CorrelaciÃ³n - {name}')
        plt.show()


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')


comprehensive_eda(train, "Train")


comprehensive_eda(test, "Test")


comprehensive_eda(original, "Original")


original_copy = original.copy()
for k in range(7):
    original = pd.concat([original, original_copy], axis=0, ignore_index=True)


X_train = train.drop(['id', 'Personality'], axis=1, errors='ignore')
y_train = train['Personality']
X_test = test.drop(['id'], axis=1, errors='ignore')
X_original = original.drop(['id', 'Personality'], axis=1, errors='ignore')
y_original = original['Personality']


cat_cols = X_train.select_dtypes(include=['object']).columns
imputer = SimpleImputer(strategy='most_frequent')
X_train[cat_cols] = imputer.fit_transform(X_train[cat_cols])
X_test[cat_cols] = imputer.transform(X_test[cat_cols])
X_original[cat_cols] = imputer.transform(X_original[cat_cols])


label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    X_original[col] = le.transform(X_original[col])
    label_encoders[col] = le


target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)
y_original_encoded = target_encoder.transform(y_original)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_original_scaled = scaler.transform(X_original)


X_combined = np.vstack([X_train_scaled, X_original_scaled])
y_combined = np.concatenate([y_train_encoded, y_original_encoded])


xgb_model = XGBClassifier(
    objective='binary:logistic',
    max_leaves=25,
    min_child_weight=0.0034,
    learning_rate=0.0947,
    n_estimators=1000,
    subsample=0.8025,
    colsample_bylevel=0.8360,
    colsample_bytree=0.8733,
    reg_alpha=0.0029,
    reg_lambda=27.1263,
    random_state=42,
    tree_method='hist',
    enable_categorical=False
)

lgbm_model = LGBMClassifier(
    objective='binary',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    random_state=42,
    verbose=0
)


stacking_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('cat', cat_model)
    ],
    final_estimator=xgb.XGBClassifier(
        learning_rate=0.02,
        n_estimators=200,
        max_depth=4,
        random_state=42
    ),
    cv=5,
    stack_method='predict_proba'
)


print("\nğŸš€ Entrenando modelo ensamblado...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
    print(f"\n=== Fold {fold+1}/5 ===")
    
    # Datos de entrenamiento y validaciÃ³n
    X_train_fold, X_val_fold = X_combined[train_idx], X_combined[val_idx]
    y_train_fold, y_val_fold = y_combined[train_idx], y_combined[val_idx]
    
    # Entrenamiento
    stacking_model.fit(X_train_fold, y_train_fold)
    
    # PredicciÃ³n
    val_pred = stacking_model.predict(X_val_fold)
    val_acc = accuracy_score(y_val_fold, val_pred)
    cv_scores.append(val_acc)
    
    print(f"Accuracy Fold {fold+1}: {val_acc:.4f}")
    print(classification_report(y_val_fold, val_pred))
    print(confusion_matrix(y_val_fold, val_pred))

print("\nğŸ“Š Resultados de validaciÃ³n cruzada:")
print(f"Scores: {cv_scores}")
print(f"Media: {np.mean(cv_scores):.4f} | DesviaciÃ³n: {np.std(cv_scores):.4f}")

# Entrenamiento final con todos los datos
print("\nğŸ”¥ Entrenando modelo final con todos los datos...")
stacking_model.fit(X_combined, y_combined)


xgb_model.fit(X_combined, y_combined)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Importancia de CaracterÃ­sticas (XGBoost)')
plt.tight_layout()
plt.show()


test_pred = stacking_model.predict(X_test_scaled)
test_pred_labels = target_encoder.inverse_transform(test_pred)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_pred_labels
})

submission.to_csv('submission_ensemble.csv', index=False)
print("\nâœ… Â¡Submission creada con Ã©xito!")

