import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mute warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


# Data visualization
print(20*'****')
print('DIMENSION OF THE DATA IMPORTED: \n')
print('df_train = ',df_train.shape)
print('df_test = ',df_test.shape)
print(20*'****')
print('Predicting_Road_Accident_Risk: \n')
print(df_train.head(),'\n')
print(20*'****')
print('Predicting_Road_Accident_Risk COLUMNS: \n\nCOLUMN  |||  DTYPE  |||  UNIQUE VALUES\n')
for i in df_train.columns:
    print(f'"{i}"  |||  dtype:{df_train[i].dtypes}  |||  unique_values:{df_train[i].nunique()}\n')
print(20*'****')
print('Predicting_Road_Accident_Risk DESCRIBE:')
print(df_train.describe())
print(20*'****')


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
from sklearn.feature_selection import mutual_info_regression

# Utility functions from Tutorial
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()
#-----------------------------------------------------------------------------------------------
def corr_matrix(df, figsize=(10,8), cmap="coolwarm", factorize_categorical=True):
    df = df.copy()
    
    if factorize_categorical:
        for col in df.select_dtypes(["object", "category"]):
            df[col], _ = df[col].factorize()
    
    corr_matrix = df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap=cmap)
    plt.title("Correlation Matrix")
    plt.show()
#-----------------------------------------------------------------------------------------------
def sub_plot(col):
    n_cols = 3
    n_rows = len(col) // n_cols
    if len(col) % n_cols != 0:
        n_rows += 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()
    return fig, axes
#-----------------------------------------------------------------------------------------------


df_train.isnull().sum()


df_train.duplicated().sum()


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df_train)
print('Cat_cols:\n',cat_cols)
print('num_cols:\n',num_cols)
print('cat_but_car:\n',cat_but_car)
print('num_but_cat:\n',num_but_cat)


fig, axes = sub_plot(num_cols)
for i, col in enumerate(num_cols):
    df_train[col].hist(bins=100, color="darkred", ax=axes[i])
    axes[i].set_title(f"Histograms Plot - {col}")
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()


cols_to_plot = cat_cols+num_but_cat

fig, axes = sub_plot(cols_to_plot)
for i, col in enumerate(cols_to_plot):
    df_train[col].value_counts().sort_values().plot.barh(
        ax=axes[i],
        color="darkgreen"
    )
    axes[i].set_title(f"Barh Plot - {col}")
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8)) 
sns.scatterplot(
    data=df_train,
    x="accident_risk",   
    y="curvature",        
    hue="holiday",       
    style="road_type",        
    palette="deep"
)
plt.title('Scatter Plot: accident_risk vs curvature')
plt.legend(loc="upper right")
plt.show()


plt.figure(figsize=(12, 8))
sns.violinplot(
    data=df_train,
    x="accident_risk",          
    y="curvature",        
    palette="deep"
)

plt.title("Violin Plot: accident_risk vs curvature")
plt.legend(loc="upper right")
plt.show()


plt.figure(figsize=(12, 8))
sns.regplot(
    data=df_train,
    x="accident_risk",  
    y="curvature",    
    scatter_kws={"alpha":0.5}, 
    line_kws={"color":"red"}   
)

plt.title("RegPlot: accident_risk vs curvature")
plt.show()


fig, axes = sub_plot(cols_to_plot)

boxplot_style = {
    "boxprops": {'facecolor':'lightgrey', 'edgecolor':'black'},
    "medianprops": {'color': 'red', 'linewidth': 2.5},
    "whiskerprops": {'color': 'black'},
    "capprops": {'color': 'black'},
    "flierprops": {'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'black', 'markersize': 6}
}

for i, col in enumerate(cols_to_plot):
    ax = axes[i]
    sns.boxplot(data=df_train, x=col, y='accident_risk', ax=ax, **boxplot_style)
    ax.set_title(f'Accident Risk by {col}', fontsize=14)
    ax.set_xlabel(None)
    ax.set_ylabel('Accident Risk', fontsize=12)
    ax.tick_params(axis='x', rotation=20)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Categorical Analysis of Accident Risk', fontsize=20, weight='bold')
fig.text(0.99, 0.01, 'Created By Ozan M.',
         ha='right', va='bottom', fontsize=10, color='dimgray')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


X = df_train.drop(['accident_risk'],axis=1)
y = df_train.accident_risk

scores = make_mi_scores(X, y)
plot_mi_scores(scores)


corr_matrix(X)


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


X = df_train_encoded.drop(['accident_risk'],axis=1)
y = df_train_encoded.accident_risk

scores = make_mi_scores(X, y)
plot_mi_scores(scores)


"""
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(X)
print('Cat_cols:\n',cat_cols)
print('num_cols:\n',num_cols)
print('cat_but_car:\n',cat_but_car)
print('num_but_cat:\n',num_but_cat)

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("scaler", StandardScaler(), num_cols),
        ("passthrough", "passthrough", num_but_cat)
    ]
)

modelos = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5),
    "Decision Tree": DecisionTreeRegressor(random_state=0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
    "Gradient Boosting": GradientBoostingRegressor(random_state=0),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=0)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

resultados = []

for nombre, modelo in modelos.items():
    print(f"\n=== {nombre} ===")
    maes, rmses, r2s = [], [], []
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", modelo)
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        
        mae = mean_absolute_error(y_val, y_pred)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        r2 = r2_score(y_val, y_pred)
        
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        
        print(f"Fold {fold+1} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    
    # Promedio final del modelo
    resultados.append((
        nombre,
        np.mean(maes),
        np.mean(rmses),
        np.mean(r2s)
    ))

# Ordenar por R² promedio
resultados = sorted(resultados, key=lambda x: x[3], reverse=True)

print("\n=== Resultados finales promedios por modelo ===")
for nombre, mae, rmse, r2 in resultados:
    print(f"{nombre:20} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
"""

