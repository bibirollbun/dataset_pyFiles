# Importación de Librerías
import pandas as pd 

from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from lightgbm import LGBMClassifier, early_stopping

def metric(y_true, y_pred):
    res = cohen_kappa_score(y_true, y_pred.reshape((y_true.shape[0], 5), order="F").argmax(axis=1), weights= 'quadratic')
    return "kappa", res, True


train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv').set_index("PetID")
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv').set_index("PetID")
folds = [train.index[test_idx] for train_idx, test_idx in KFold(5).split(train)]


for c in ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'FurLength', 'MaturitySize', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State', 'RescuerID']:
    train[c] = train[c].astype("category")    
    test[c] = test[c].astype("category")


train = train\
    .join(pd.read_parquet("../input/examen-lab-ii-preprocesado-texto/train.parquet"))\
    .join(pd.read_parquet("../input/examen-lab-ii-preprocesado-img/train.parquet"))\
    .select_dtypes(exclude="O")
train


test = test\
    .join(pd.read_parquet("../input/examen-lab-ii-preprocesado-texto/test.parquet"))\
    .join(pd.read_parquet("../input/examen-lab-ii-preprocesado-img/test.parquet"))\
    .select_dtypes(exclude="O")
test


fold_preds = []
test_preds = []

for i, fold in enumerate(folds):
    # Datos de entrenamiento 
    X_train = train.drop(fold).drop("AdoptionSpeed", axis=1).select_dtypes(exclude="O")
    y_train = train.drop(fold)["AdoptionSpeed"]

    # Datos de validación (igual que el original)
    X_valid = train.loc[fold].drop("AdoptionSpeed", axis=1).select_dtypes(exclude="O")
    y_valid = train.loc[fold, "AdoptionSpeed"]

    # Modelo LGBM con hiperparámetros ajustados
    learner = LGBMClassifier(
        objective="multiclass",
        num_class=5,
        learning_rate=0.03,
        n_estimators=800,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,         # antes: por defecto
        colsample_bytree=0.8,  # antes: por defecto
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    # Entrenamos 
    learner.fit(X_train, y_train)

    learner.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric=metric,
        callbacks=[early_stopping(50)]  # antes 5 → hiperparámetro mejorado
    )

    # Predicciones 
    fold_preds.append(pd.Series(learner.predict(X_valid), index=fold, name="AdoptionSpeed"))
    test_preds.append(pd.DataFrame(learner.predict_proba(test[X_train.columns]), index=test.index))

# Agregamos las predicciones de todos los folds 
fold_preds = pd.concat(fold_preds)
test_preds = sum(test_preds).idxmax(axis=1).rename("AdoptionSpeed")

    


test_preds.to_csv("submission.csv")
!head submission.csv

