# Importación de Librerías
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from lightgbm import LGBMClassifier, early_stopping

def metric(y_true, y_pred):
    res = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return "kappa", res, True


# Cargar los datos
train = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/train/train.csv').set_index("PetID")
test = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/test.csv').set_index("PetID")


!ls /kaggle/input



!ls /kaggle/input/examen-lab-ii-text-franco-fontan
!ls /kaggle/input/d/agustinafranco/examen-lab-ii-img-franco-fontan



!ls -R /kaggle/input/examen-lab-ii-text-franco-fontan
!ls -R /kaggle/input/d/agustinafranco/examen-lab-ii-img-franco-fontan



# Unir texto preprocesado
train = train.join(pd.read_parquet("/kaggle/input/examen-lab-ii-text-franco-fontan/train.parquet"))
test  = test.join(pd.read_parquet("/kaggle/input/examen-lab-ii-text-franco-fontan/test.parquet"))

# Unir features de imágenes preprocesadas
train = train.join(pd.read_parquet("/kaggle/input/d/agustinafranco/examen-lab-ii-img-franco-fontan/train.parquet"))
test  = test.join(pd.read_parquet("/kaggle/input/d/agustinafranco/examen-lab-ii-img-franco-fontan/test.parquet"))




# Convertir a categorías
categorical_features = [
    'Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
    'FurLength', 'MaturitySize', 'Vaccinated', 'Dewormed',
    'Sterilized', 'Health', 'State', 'RescuerID'
]
for c in categorical_features:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")


from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from lightgbm import LGBMClassifier
import numpy as np

# =============================
# Definir y entrenar el modelo
# =============================

NUM_FOLDS = 5
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# predicciones OOF (una clase por fila)
oof_preds = np.zeros(len(train), dtype=np.int8)

# probabilidades para el set de test (5 clases)
test_probas = np.zeros((len(test), 5), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print(f"Fold {fold + 1}/{NUM_FOLDS}")

    # separar features y target
    X_train = train.iloc[train_idx].drop('AdoptionSpeed', axis=1).select_dtypes(exclude='object')
    y_train = train.iloc[train_idx]['AdoptionSpeed'].values

    X_valid = train.iloc[val_idx].drop('AdoptionSpeed', axis=1).select_dtypes(exclude='object')
    y_valid = train.iloc[val_idx]['AdoptionSpeed'].values

    X_test = test.select_dtypes(exclude='object')

    # asegurar mismas columnas en train/valid/test
    shared_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[shared_cols]
    X_valid = X_valid[shared_cols]
    X_test  = X_test[shared_cols]

    model = LGBMClassifier(
        objective='multiclass',
        num_class=5,
        metric='multi_logloss',   # métrica interna de LightGBM
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )

    # ¡OJO! NO usamos cohen_kappa_score aquí dentro
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='multi_logloss',
        callbacks=[early_stopping(stopping_rounds=10, verbose=False)]
    )

    # probabilidades en valid y test
    valid_proba = model.predict_proba(X_valid)      # (n_valid, 5)
    oof_preds[val_idx] = np.argmax(valid_proba, 1)  # clases 0..4

    test_probas += model.predict_proba(X_test) / NUM_FOLDS

# =============================
# Calcular kappa
# =============================
y_true = train['AdoptionSpeed'].values
kappa = cohen_kappa_score(y_true, oof_preds, weights='quadratic')
print(f"Kappa: {kappa:.4f}")



print("Kappa score:", kappa)



# =============================
# Generar archivo submission
# =============================
test_labels = np.argmax(test_probas, axis=1)

submission = pd.DataFrame({
    'PetID': test.index,
    'AdoptionSpeed': test_labels
})
submission.to_csv('submission.csv', index=False)
submission.head()




# Asegurarse de que los valores de AdoptionSpeed sean válidos
valid_adoption_speeds = [0, 1, 2, 3, 4]
submission['AdoptionSpeed'] = submission['AdoptionSpeed'].map(lambda x: x if x in valid_adoption_speeds else 0)


# Imprimir información de debug
print("Submission DataFrame:")
print(submission.head())
print("\nValue counts for AdoptionSpeed:")
print(submission['AdoptionSpeed'].value_counts())

submission.to_csv('submission.csv', index=False)

