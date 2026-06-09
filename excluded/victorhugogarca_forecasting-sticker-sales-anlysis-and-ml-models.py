import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz, plot_tree
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


raw_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


raw_df.info()
test_df.info()


raw_df.head(5)


test_df.head(5)


raw_df.describe()


valores_faltantes = raw_df.isnull().sum()
valores_faltantes


raw_df['num_sold'].describe()


raw_df['date']= pd.to_datetime(raw_df['date'])
test_df['date']= pd.to_datetime(test_df['date'])


def add_dateparts(df,col):
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month
    df[col + '_day'] = df[col].dt.day
    df[col + '_nameday'] = df[col].dt.day_name()
add_dateparts(raw_df, "date")
add_dateparts(test_df, "date")


raw_df.head(5)


test_df.head(5)


fig = px.histogram(raw_df, 
             x= 'date_nameday', 
             y = 'num_sold', 
             color='date_nameday', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_layout(
    title="Ventas por día de la semana",
    xaxis_title="Día de la semana",
    yaxis_title="Cantidad vendida",
    bargap=0.2  
)


fig = px.histogram(raw_df, 
             x= 'date_day', 
             y = 'num_sold', 
             color='date_month', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_layout(
    title="Ventas por día de mes",
    xaxis_title="Día del mes",
    yaxis_title="Cantidad vendida",
    bargap=0.2  
)


fig = px.histogram(raw_df, 
             x= 'date_month', 
             y = 'num_sold', 
             color='date_month', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl
                  )
fig.update_layout(
    title="Ventas por mes",
    xaxis_title="Mes",
    yaxis_title="Cantidad vendida",
    bargap=0.2  # Espacio entre las barras
)


fig = px.histogram(raw_df, 
             x= 'date_year', 
             y = 'num_sold', 
             color='product', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_layout(
    title="Ventas por año",
    xaxis_title="Año",
    yaxis_title="Cantidad vendida",
    bargap=0.2 
)


fig = px.histogram(raw_df, 
             x= 'country', 
             y = 'num_sold', 
             color='country', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_layout(
    title="Ventas por país",
    xaxis_title="País",
    yaxis_title="Cantidad vendida",
    bargap=0.2  
)


fig = px.box(raw_df, 
             x= 'country', 
             y = 'num_sold', 
             color='country', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_layout(
    title="Diagrama de caja por país",
    xaxis_title="País",
    yaxis_title="Cantidad vendida",
    bargap=0.2  
)


fig = px.histogram(raw_df, 
             x= 'store', 
             y = 'num_sold', 
             color='product', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_layout(
    title="Ventas por tienda",
    xaxis_title="Tienda",
    yaxis_title="Cantidad vendida",
    bargap=0.2  
)


fig = px.box(raw_df, 
             x= 'store', 
             y = 'num_sold', 
             color='store', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_layout(
    title="Diagrama de caja por tienda",
    xaxis_title="Tienda",
    yaxis_title="Cantidad vendida",
    bargap=0.2  
)


fig = px.histogram(raw_df, 
             x= 'product', 
             y = 'num_sold', 
             color='country', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_layout(
    title="Ventas de productos",
    xaxis_title="Producto",
    yaxis_title="Cantidad vendida",
    bargap=0.2  
)


fig = px.box(raw_df, 
             x= 'product', 
             y = 'num_sold', 
             color='product', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_layout(
    title="Diagrama de caja por producto",
    xaxis_title="Producto",
    yaxis_title="Cantidad vendida",
    bargap=0.2  
)


train_clean_df= raw_df.fillna(raw_df.num_sold.mean())

train_clean_df.isna().sum()


outliers = train_clean_df[
### Outliers de paises
(((train_clean_df['country']=='Canada') & (train_clean_df['num_sold']>=2274))|
((train_clean_df['country']=='Norway') & (train_clean_df['num_sold']>=2047))|
((train_clean_df['country']=='Finland') & (train_clean_df['num_sold']>=2286))|
((train_clean_df['country']=='Singapore') & (train_clean_df['num_sold']>=2610))|
((train_clean_df['country']=='Italy') & (train_clean_df['num_sold']>=1652)))|
###Outliers de productos
(((train_clean_df['product']=='Kaggle') & (train_clean_df['num_sold']>=3250))|
((train_clean_df['product']=='Kaggle Tiers') & (train_clean_df['num_sold']>=2681))|
((train_clean_df['product']=='Kerneler') & (train_clean_df['num_sold']>=1490))|
((train_clean_df['product']=='Kerneler Dark Mode') & (train_clean_df['num_sold']>=1768))|
((train_clean_df['product']=='Holographic Goose') & (train_clean_df['num_sold']>=463)))|
###Outliers de tiendas
(((train_clean_df['store']=='Discount Stickers') & (train_clean_df['num_sold']>=1347))|
((train_clean_df['store']=='Stickers for Less') & (train_clean_df['num_sold']>=2766))|
((train_clean_df['store']=='Premium Sticker Mart') & (train_clean_df['num_sold']>=3265)))
]


indices_a_eliminar = outliers.index
train_clean_df = train_clean_df.drop(indices_a_eliminar)


#1. Dividimos el train_clean_df en 2 conjuntos, uno de entrenamiento y otro de validacion, 
# dejando el orden de las fechas cronologicamente pues deseamos hacer predicciones a futuro
train_df, val_df = train_test_split(train_clean_df, test_size = 0.2, shuffle = False)

#2. Definimos las columnas que seran nuestras variables y nuestra columna objetivo
input_cols = ['country', 'store', 'product', 'date_year', 'date_month', 'date_day', 'date_nameday']
target_col = 'num_sold'

#3. Separamos las columnas de acuerdo al objetivo, si son target o input.
train_inputs, train_targets = train_df[input_cols], train_df[target_col]
val_inputs, val_targets = val_df[input_cols], val_df[target_col]
test_inputs = test_df[input_cols]

#4 Identificamos la columnas numericas y categoricas para ocupar OneHotEncoding adecuadamente
numerical_cols = ['date_day']
categorical_cols = ['country', 'store', 'product', 'date_year', 'date_month', 'date_nameday']

#5. Codificamos las columnas categoricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

#6. Nombramos los conjuntos que ocuparemos para entrenar, validar y probar nuestro modelo
X_train = train_inputs[encoded_cols + numerical_cols]
X_val = val_inputs[encoded_cols + numerical_cols]
X_test = test_inputs[encoded_cols + numerical_cols]


def testing_model(model, name):

    model.fit(X_train, train_targets)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    score_train = model.score(X_train, train_targets)
    score_val = model.score(X_val, val_targets)

    print(f"{name}\n"
      f"Métricas de entrenamiento:\n"
      f"  - Mean Squared Error: {train_rmse}\n"
      f"  - Score R²: {score_train}\n"
      f"Métricas de validación:\n"
      f"  - Mean Squared Error: {val_rmse}\n"
      f"  - Score R²: {score_val}")



def weights_model(model):
    #Ajustamos el modelo
    model.fit(X_train, train_targets)
    #Creamos el DataFrame con los pesos
    weights_df = pd.DataFrame({
         'feature': np.append(X_train.columns,+1),
         'weight': np.append(model.coef_, model.intercept_)})
    #Ordenamos el df y  lo graficamos
    weights_sort=weights_df.sort_values('weight', ascending=True)
    fig = px.histogram(weights_sort, 
             x= 'weight', 
             y = 'feature', 
             #color='feature', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)                       
    fig.update_layout(
        title="Importancia de las caracteristicas",
        bargap=0.2  
        )    
    return fig


def importance_feature(model):
    #Ajustamos el modelo
    model.fit(X_train, train_targets)
    #Creamos el DataFrame con los pesos
    weights_df = pd.DataFrame({
         'feature': X_train.columns,
         'weight': model.feature_importances_})
    #Ordenamos el df y  lo graficamos
    weights_sort=weights_df.sort_values('weight', ascending=True)
    fig = px.bar(weights_sort, 
             x= 'weight', 
             y = 'feature', 
             #color='feature', 
             color_discrete_sequence=px.colors.sequential.Aggrnyl)                       
    fig.update_layout(
        title="Importancia de las caracteristicas",
        bargap=0.2  
        )    
    return fig


%%time
testing_model(LinearRegression(), 'Regresion Lineal')


weights_model(LinearRegression())


%%time
testing_model(Ridge(), 'Ridge')


weights_model(Ridge())


%%time
testing_model(SGDRegressor(), 'SGD Regressor')


weights_model(SGDRegressor())


%%time
testing_model(DecisionTreeRegressor(), 'Arbol de desicion')


def visualize_decision_tree(model, feature_names, max_depth=None):
    plt.figure(figsize=(24, 10))  # Increase figure size
    plot_tree(model, 
              feature_names=feature_names, 
              filled=True, 
              fontsize=10, 
              proportion=True, 
              max_depth=max_depth) 
    plt.title("Decision Tree Visualization", fontsize=16)
    plt.show()

# Entrenamos un modelo de regresion con maximo 4 para su visualizacion
decision_tree_model = DecisionTreeRegressor(max_depth=4)  # Limit depth to 4
decision_tree_model.fit(X_train, train_targets)

# Visualize the improved tree
visualize_decision_tree(decision_tree_model, feature_names=X_train.columns)


importance_feature(DecisionTreeRegressor())


def testing_params(modelo,**params):
    model = modelo(random_state=42, **params)
    model.fit(X_train, train_targets)
    
    train_error = 1 - model.score(X_train, train_targets) 
    val_error = 1 - model.score(X_val, val_targets)
    return { **params,'Training Error': train_error, 'Validation Error': val_error}


testing_params_df = pd.DataFrame([testing_params(DecisionTreeRegressor, max_depth=8*i ) for i in range(1, 5)])
plt.figure()
plt.plot(testing_params_df['max_depth'], testing_params_df['Training Error'])
plt.plot(testing_params_df['max_depth'], testing_params_df['Validation Error'])
plt.title('Training vs. Validation Error')
plt.xlabel('Max. Depth')
plt.ylabel('Prediction Error (1 - R^2)')
plt.legend(['Training', 'Validation'])
plt.show()


testing_params_df.sort_values('Validation Error', ascending=True).head(1)


testing_params_df = pd.DataFrame([testing_params(DecisionTreeRegressor, max_depth = 16, max_leaf_nodes=25*i) for i in range(1, 10)])
plt.figure()
plt.plot(testing_params_df['max_leaf_nodes'], testing_params_df['Training Error'])
plt.plot(testing_params_df['max_leaf_nodes'], testing_params_df['Validation Error'])
plt.title('Training vs. Validation Error')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('Prediction Error (1 - R^2)')
plt.legend(['Training', 'Validation'])
plt.show()


testing_params_df.sort_values('Validation Error', ascending=True).head(1)


%%time
testing_model(DecisionTreeRegressor(max_depth=16, max_leaf_nodes = 175, random_state=42), 'Arbol de desicion')


%%time
testing_model(RandomForestRegressor(), 'Random Forest')


importance_feature(RandomForestRegressor())


testing_params_df = pd.DataFrame([testing_params(RandomForestRegressor, n_estimators=2*i, n_jobs=-1 ) for i in range(12, 25)])
plt.figure()
plt.plot(testing_params_df['n_estimators'], testing_params_df['Training Error'])
plt.plot(testing_params_df['n_estimators'], testing_params_df['Validation Error'])
plt.title('Training vs. Validation Error')
plt.xlabel('N estimators')
plt.ylabel('Prediction Error (1 - R^2)')
plt.legend(['Training', 'Validation'])
plt.show()


testing_params_df.sort_values('Validation Error', ascending=True).head(1)


testing_params_df = pd.DataFrame([testing_params(RandomForestRegressor, n_estimators=40, max_depth=3*i, n_jobs=-1 ) for i in range(1, 10)])
plt.figure()
plt.plot(testing_params_df['max_depth'], testing_params_df['Training Error'])
plt.plot(testing_params_df['max_depth'], testing_params_df['Validation Error'])
plt.title('Training vs. Validation Error')
plt.xlabel('Max Depth')
plt.ylabel('Prediction Error (1 - R^2)')
plt.legend(['Training', 'Validation'])
plt.show()


testing_params_df.sort_values('Validation Error', ascending=True).head(1)


testing_params_df = pd.DataFrame([testing_params(RandomForestRegressor, n_estimators=40, max_depth=18, max_leaf_nodes=50*i, n_jobs=-1 ) for i in range(4, 8)])
plt.figure()
plt.plot(testing_params_df['max_leaf_nodes'], testing_params_df['Training Error'])
plt.plot(testing_params_df['max_leaf_nodes'], testing_params_df['Validation Error'])
plt.title('Training vs. Validation Error')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('Prediction Error (1 - R^2)')
plt.legend(['Training', 'Validation'])
plt.show()


testing_params_df.sort_values('Validation Error', ascending=True).head(1)


%%time
testing_model(RandomForestRegressor(random_state = 42, n_estimators=18, max_depth=18, max_leaf_nodes=340, n_jobs=-1), 'Random Forest')


def predict_and_submit(model, fname):
    model.fit(X_train, train_targets)
    test_preds = model.predict(X_test)
    sub_df = test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
    sub_df['num_sold'] = test_preds
    sub_df.to_csv(fname, index=None)


predict_and_submit(RandomForestRegressor(random_state = 42, n_estimators=18, max_depth=18, max_leaf_nodes=340, n_jobs=-1),'rf0_optimized_submission')




