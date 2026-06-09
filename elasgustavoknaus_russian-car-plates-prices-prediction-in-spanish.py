import numpy as np
import pandas as pd
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_pred - y_true) / denominator
    diff[denominator == 0] = 0  # Evita divisiÃ³n por cero
    return np.mean(diff) * 100
    
y_true = [100, 200, 300]
y_pred = [110, 190, 290]

print(f"sMAPE: {smape(y_true, y_pred):.2f}%")



train = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv",index_col = "id") 
test = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv", index_col = "id")

train


test.drop("price", axis = 1, inplace = True)


display(train.describe().T)
# Contar cuÃ¡ntos precios tiene cada "plate"
precios_por_plate = train.groupby("plate")["price"].count()
print(precios_por_plate)



train.price.hist(bins = 20, color = "black")


def descomponer_placa(plate):
    plate = plate.strip()  # eliminar espacios
    
    # Validar que la longitud sea correcta
    if len(plate) >= 8:
        return {
            "serie_inicio": plate[0],            # Letra inicial
            "numero": plate[1:4],                # Los tres nÃºmeros
            "serie_final": plate[4:6],           # Dos letras
            "region": plate[6:]                  # El cÃ³digo de regiÃ³n
        }
    else:
        return {"serie_inicio": None, "numero": None, "serie_final": None, "region": None}

X = train.copy()
Y  = X.pop("price")

def preprocesamiento(X = X,test = test):
    
    df = pd.concat([X,test], axis = 0) # concatename en el eje de las filas
    
    # transformamos date a fomrmato fecha de pandas
    
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # DESCOMPONEMOS Y JUNTAMOS
    
    df_descompuesto = pd.DataFrame(
        df["plate"]
          .apply(descomponer_placa)  # Serie de dicts, Ã­ndice = el del df original
          .to_list(),
        index=df.index               # reusar exactamente el mismo Ã­ndice
    )
    
    df_final = pd.concat([df, df_descompuesto], axis=1)
    df_final.drop("plate", axis = 1, inplace = True)
    
    # SEPARAMOS:
    
    t_rows = train.shape[0]
    
    X_new_train = df_final.iloc[:t_rows]
    X_new_test = df_final.iloc[t_rows:]
    X_new_train[["numero","region"]]=X_new_train[["numero","region"]].astype("int32")
    X_new_test[["numero","region"]]=X_new_test[["numero","region"]].astype("int32")
    return X_new_train, X_new_test


X_new_train, X_new_test = preprocesamiento(X,test)

X_new_test.info()


X_new_train.info()



# from sklearn.dummy import DummyRegressor # El modelo predictivo realmente aporta valor ? con esto lo averiguaremos
# from sklearn.model_selection import train_test_split, cross_val_predict
# from sklearn.metrics import make_scorer, r2_score
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OrdinalEncoder, RobustScaler
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# import pandas as pd

# # ====================
# # 1. Definir sMAPE
# # ====================
# def smape(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
#     diff = np.abs(y_pred - y_true) / denominator
#     diff[denominator == 0] = 0
#     return np.mean(diff) * 100

# # Menos es mejor
# smape_scorer = make_scorer(smape, greater_is_better=False)


# X_train, X_test, y_train, y_test = train_test_split(X_new_train, Y, test_size=0.135, random_state=42)

# # ====================
# # 3. Preprocesamiento
# # ====================
# numeric_features = X_new_train.select_dtypes(include=['float', 'int']).columns.tolist()
# categorical_features = ['region']

# preprocessor = ColumnTransformer([
#     ('num', RobustScaler(), numeric_features),                # Escalar numÃ©ricas
#     ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)  # Codificar categÃ³ricas
# ])

# # ====================
# # 4. Modelos
# # ====================
# models = {
#     "WithMedian": DummyRegressor(strategy='median'),
#     "Random Forest": RandomForestRegressor(),
#     "Gradient Boosting": GradientBoostingRegressor()
# }

# # ====================
# # 5. Entrenamiento + PredicciÃ³n cruzada
# # ====================
# predictions = {}
# scores = {}

# for name, model in models.items():
#     pipe = Pipeline([
#         ('preprocess', preprocessor),
#         ('regressor', model)
#     ])
    
#     y_pred = cross_val_predict(pipe, X_train, y_train, cv=5)
#     smape_score = smape(y_train, y_pred)
#     r2 = r2_score(y_train, y_pred)
    
#     predictions[name] = y_pred
#     scores[name] = {'sMAPE': smape_score, 'R2': r2}
    
#     print(f"\nğŸ”� Modelo: {name}")
#     print(f"  sMAPE: {smape_score:.2f}%")
#     print(f"  RÂ²: {r2:.2f}")

# # ====================
# # 6. Promedio ponderado
# # ====================

# # FILTRAR modelos que no queremos en el promedio
# models_to_combine = [m for m in models if not isinstance(models[m], DummyRegressor)]

# # Obtener errores sMAPE solo para esos modelos
# smape_errors = np.array([scores[m]['sMAPE'] for m in models_to_combine])
# inv_errors = 1 / (smape_errors + 1e-8)
# weights = inv_errors / inv_errors.sum()

# print("\nğŸ“Š Pesos del promedio ponderado segÃºn sMAPE (excluyendo DummyRegressor):")
# for i, model in enumerate(models_to_combine):
#     print(f"  {model}: {weights[i]:.2f}")

# # Promedio ponderado solo con los modelos seleccionados
# final_pred = sum(predictions[m] * weights[i] for i, m in enumerate(models_to_combine))

# # Evaluar
# final_smape = smape(y_train, final_pred)
# final_r2 = r2_score(y_train, final_pred)

# print(f"\nğŸ“ˆ Promedio Ponderado (sin Dummy):")
# print(f"  sMAPE: {final_smape:.2f}%")
# print(f"  RÂ²: {final_r2:.2f}")



from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.metrics import make_scorer, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# 1. MÃ©trica sMAPE
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_pred - y_true) / denom
    diff[denom == 0] = 0
    return np.mean(diff) * 100

smape_scorer = make_scorer(smape, greater_is_better=False)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_new_train, Y, test_size=0.135, random_state=42
)

# 3. Preprocesamiento
numeric_features  = X_train.select_dtypes(include=['float', 'int']).columns.tolist()
categorical_feats = ['region']

preprocessor = ColumnTransformer([
    ('num', RobustScaler(), numeric_features),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_feats)
])

# 4. Pipelines + ParÃ¡metros para GridSearch
pipelines = {
    'rf': Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(
            random_state=42,
            n_jobs=-1              # paraleliza la construcciÃ³n de los Ã¡rboles
        ))
    ]),
    'hgb': Pipeline([
        ('pre', preprocessor),
        ('reg', HistGradientBoostingRegressor(
            random_state=42,
            max_iter=100           # no soporta n_jobs
        ))
    ]),
}

param_grids = {
    'rf': {
        'reg__n_estimators': [100, 200, 300],
        'reg__max_depth': [None, 10, 20],
        'reg__max_features': ['sqrt', 0.5]
    },
    'hgb': {
        'reg__max_iter': [100, 200],
        'reg__learning_rate': [0.01, 0.1],
        'reg__max_depth': [3, 5]
    }
}

# 5. GridSearchCV (paralelizado sobre folds y combos)
best_models = {}
for name in pipelines:
    print(f"\n--- Buscando best params para {name} ---")
    gs = GridSearchCV(
        pipelines[name],
        param_grids[name],
        cv=5,
        scoring=smape_scorer,
        n_jobs=-1,             # usa todos los nÃºcleos
        verbose=1
    )
    gs.fit(X_train, y_train)
    print(f"Mejores params ({name}): {gs.best_params_}")
    print(f"sMAPE (cv): {-gs.best_score_:.2f}%")
    best_models[name] = gs.best_estimator_

# 6. EvaluaciÃ³n final en train con cross_val_predict (paralelizado)
scores = {}
preds  = {}
for name, model in best_models.items():
    y_pred = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=5,
        n_jobs=-1,            # paraleliza folds
        verbose=1
    )
    scores[name] = {
        'sMAPE': smape(y_train, y_pred),
        'R2':    r2_score(y_train, y_pred)
    }
    preds[name] = y_pred

# 7. Baseline con DummyRegressor (cross_val_predict tambiÃ©n en paralelo)
dummy = Pipeline([
    ('pre', preprocessor),
    ('reg', DummyRegressor(strategy='median'))
])
y_dummy = cross_val_predict(
    dummy,
    X_train,
    y_train,
    cv=5,
    n_jobs=-1
)
scores['baseline'] = {
    'sMAPE': smape(y_train, y_dummy),
    'R2':    r2_score(y_train, y_dummy)
}

# 8. Mostrar resultados
print("\n===== Resumen de MÃ©tricas (5-fold CV) =====")
for name, sc in scores.items():
    print(f"{name:>10} â†’ sMAPE: {sc['sMAPE']:.2f}%   RÂ²: {sc['R2']:.3f}")

# 9. (Opcional) Promedio ponderado con solo los mejores
names = [n for n in best_models]
errs = np.array([scores[n]['sMAPE'] for n in names])
inv  = 1/(errs + 1e-8)
wts  = inv/inv.sum()
print("\nPesos ensemble:", dict(zip(names, np.round(wts,2))))

ensemble_pred = sum(preds[n] * w for n, w in zip(names, wts))
print("Ensemble sMAPE:", smape(y_train, ensemble_pred))
print("Ensemble R2:   ", r2_score(y_train, ensemble_pred))



preds_test = {}
for name, model in best_models.items():
    preds_test[name] = model.predict(X_new_test)
    print("preds_test - name: ", name)
# . Combina en ensemble ponderado (usando los mismos wts que calculaste)

ensemble_pred = sum(preds_test[name] * w for name, w in zip(best_models, wts)) # PROMEDIO de las predicciones - MEJORA de r2 y generalizaciÃ³n 

# . Crea el DataFrame de submission
submission = pd.DataFrame({
    'id':   X_new_test.index,
    'price': ensemble_pred
})

# . Guarda a CSV
submission.to_csv('submission.csv', index=False)

