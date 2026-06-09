import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col="id")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col="id")


# notebook: House Prices--advance_preprocesor_search_parameters.ipynb
def grab_col_names(df, target=None, cat_th=10, car_th=20):
    # Categóricas (objetos, categorías o booleanas)
    cat_cols = [col for col in df.columns if df[col].dtype in ["O", "category", "bool"]]
    
    # Numéricas pero categóricas (discretas con pocas categorías)
    num_but_cat = [col for col in df.columns 
                   if df[col].nunique() < cat_th and df[col].dtype in ["int64", "float64"]]
    
    # Categóricas cardinales (muchas categorías únicas)
    cat_but_car = [col for col in df.columns 
                   if df[col].nunique() > car_th and df[col].dtype in ["O", "category"]]
    
    # Unir categóricas verdaderas + numéricas discretas
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    # Numéricas reales (sin incluir num_but_cat)
    num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    # Excluir target si está en alguna lista
    if target:
        for col_list in [cat_cols, num_cols, cat_but_car, num_but_cat]:
            if target in col_list:
                col_list.remove(target)
                
    cat_cols = [col for col in cat_cols if col not in num_but_cat]
    
    # Resumen
    print("-" * 20)
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print("-" * 20)
    print('Cat_cols:\n',cat_cols)
    print('num_cols:\n',num_cols)
    print('cat_but_car:\n',cat_but_car)
    print('num_but_cat:\n',num_but_cat)
    print("-" * 20)
    
    return cat_cols, num_cols, cat_but_car, num_but_cat
#-----------------------------------------------------------------------------------------------
def add_engineered_features(df):
    df = df.copy()
    bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in bool_cols:
        df[col] = df[col].astype(int)  # True → 1, False → 0
    # road_type * speed_limit
    df['road_type_speed'] = df['road_type'] + "_" + df['speed_limit'].astype(str)
    # curvature * speed_limit
    df['curvature_speed'] = df['curvature'] * df['speed_limit']
    # lighting * time_of_day
    df['lighting_time'] = df['lighting'] + "_" + df['time_of_day']
    # weather * road_type
    df['weather_road'] = df['weather'] + "_" + df['road_type']
    # mean of num_reported_accidents for road_type
    df['avg_accidents_by_road_type'] = df.groupby('road_type')['num_reported_accidents'].transform('mean')
    # curvature_speed * num_lanes
    df['curvature_speed_lane'] = df['curvature_speed'] * df['num_lanes']
    # curvature_speed / num_lanes
    df['curvature_speed_per_lane'] = df['curvature_speed'] / df['num_lanes']
    # num_lanes * speed_limit
    df['lane_speed_risk'] = (5 - df['num_lanes']) * df['speed_limit']
    # stats calculated from the relation of num_reported_accidents and weather
    stats = df.groupby('weather')['num_reported_accidents'].agg(['count', 'mean', 'median', 'std'])
    df['accidents_count_by_weather'] = df['weather'].map(stats['count'])
    df['accidents_mean_by_weather'] = df['weather'].map(stats['mean'])
    df['accidents_median_by_weather'] = df['weather'].map(stats['median'])
    df['accidents_std_by_weather'] = df['weather'].map(stats['std'])
    """    
    # Riesgo relativo: accidente por carril - 'accident_risk' no se puede utilizar proque es el target
    df['risk_per_lane'] = df['accident_risk'] / df['num_lanes']
    """
    # Flag combinada de alta peligrosidad
    df['high_risk_condition'] = ((df['speed_limit'] > 60) &
                                       (df['curvature'] > 0.5) &
                                       (df['weather'] != "clear")).astype(int)
    return df


df_train_encoded = add_engineered_features(df_train)
df_test_encoded = add_engineered_features(df_test)

cols_to_drop=['holiday','num_lanes','road_signs_present',
              'road_type','time_of_day','avg_accidents_by_road_type',
              'public_road','accidents_median_by_weather','school_season']

df_train_encoded = df_train_encoded.drop(columns=cols_to_drop)
df_test_encoded = df_test_encoded.drop(columns=cols_to_drop)

X = df_train_encoded.drop(['accident_risk'],axis=1)
y = df_train_encoded.accident_risk

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(X)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


"""
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("scaler", StandardScaler(), num_cols),
        ("passthrough", "passthrough", num_but_cat)
    ]
)

estimators = [
    ("xgb", XGBRegressor(
        n_estimators=200, 
        learning_rate=0.1, 
        random_state=0,
        enable_categorical=False 
    )),
    ("gbr", GradientBoostingRegressor(
        random_state=0
    )),
    ("Random Forest", RandomForestRegressor(
        n_estimators=100, 
        random_state=0
    ))
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    passthrough=True,
    n_jobs=-1
)

stacking_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", stacking_model)
])

stacking_pipeline.fit(X_train, y_train)

y_pred = stacking_pipeline.predict(X_valid)
predictions = stacking_pipeline.predict(df_test_encoded)

mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print(f"Stacking Pipeline | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
"""


"""
submission = pd.DataFrame({
    'id': df_test.index,
    'accident_risk': np.clip(predictions, 0, 1)
})
submission.to_csv('submission.csv', index=False)
print(submission.head())
"""

