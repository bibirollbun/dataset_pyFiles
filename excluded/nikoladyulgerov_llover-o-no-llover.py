import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report


# Show the figures once runned the code
pio.renderers.default = "kaggle"


SEED = 45


df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col="id")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col="id")


df_train.sample(5)


df_test.sample(5)


df_train.info()


df_test.info()


df_train.describe().T


df_train['winddirection'].unique()


print(f"Duplicados en conjunto de entrenamiento: {df_train.duplicated().sum()}")
print(f"Duplicados en conjunto de test: {df_test.duplicated().sum()}")


num_cols = df_train.select_dtypes(include='float').columns.to_list()
# cat_cols = []
target = "rainfall"
num_cols


fig = px.histogram(df_train, x=target, color=target, barmode='group')
fig.show()


rainfall_classes = df_train["rainfall"].unique()
color_map = px.colors.qualitative.Plotly[:len(rainfall_classes)]
rainfall_color_dict = dict(zip(rainfall_classes, color_map))  # Map colors to categories

for col in num_cols:
    
    # Create a figure with 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Boxplot of {col}", f"Histogram of {col}"])

    # Boxplot (Grouped by Rainfall)
    for category in rainfall_classes:
        fig.add_trace(
            go.Box(
                y=df_train[df_train["rainfall"] == category][col],
                name=f"Rainfall: {category}",
                marker_color=rainfall_color_dict[category],
                boxpoints="all",
                jitter=0.5,
                pointpos=-2
            ),
            row=1, col=1
        )

    # Overlayed Histogram (Grouped by Rainfall)
    for category in rainfall_classes:
        fig.add_trace(
            go.Histogram(
                x=df_train[df_train["rainfall"] == category][col],
                name=f"Rainfall: {category}",
                marker_color=rainfall_color_dict[category],  # Apply correct color
                opacity=0.6,
                showlegend=False
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        showlegend=True, barmode="overlay"  # Overlapping histograms
    )

    fig.show()


fig = px.imshow(df_train[num_cols].corr().round(2), text_auto=True)
fig.show()


df_test['winddirection'] = df_test['winddirection'].fillna(df_test['winddirection'].median())
df_test['winddirection'].isna().sum()


def feature_engineering(df):
    
    # Nuevas variables numéricas
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_dew_diff'] = df['temparature'] - df['dewpoint']
    df['humid_temp'] = df['humidity'] * df['temparature']
    df['sun_cloud_ratio'] = df['sunshine'] / (df['cloud'] + 1) # valores altos, más sol; valores bajos, mas nubes # "1" evitar divisiones por 0
    df['cloud_windspeed_interaction'] = df['cloud'] * df['windspeed'] # muchas nubes y vientos suelen indicar tormentas
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)
    
    # Nuevas variables categóricas
    df['month'] = pd.to_datetime(df['day'], format='%j').dt.month 
    df['season'] = df['month'].apply(lambda x: 1 if 3 <= x <= 5  # Primavera
                                         else 2 if 6 <= x <= 8  # Verano
                                         else 3 if 9 <= x <= 11  # Otoño
                                         else 0)  # Invierno
    df['wind_cardinal'] = pd.cut(df['winddirection'], bins=[0, 90, 180, 270, 360], labels=['NE', 'SE', 'SW', 'NW'], include_lowest=True)
    df['windspeed_category'] = pd.cut(df['windspeed'], bins=[0, 10, 30, 50, 60], labels=['Calma', 'Brisa', 'Ventoso', 'Muy Ventoso'],include_lowest=True)

    # Borrar variables redundantes
    df = df.drop(columns=['maxtemp', 'mintemp', 'winddirection', 'day','month'])
    
    return df


df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)


df_train.sample(5, random_state=SEED)


cat_cols = ['wind_cardinal', 'windspeed_category', 'season']
num_cols = df_train.select_dtypes(include=np.number).columns.to_list()
num_cols.remove('rainfall') # target
print(f"Características numéricas: {num_cols}")


X = df_train.drop(['rainfall'], axis=1)
y = df_train['rainfall']


num_transformer = Pipeline([
    ('minmax_scaler', MinMaxScaler())
])	
num_transformer


cat_transformer = Pipeline([
        ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])
cat_transformer


preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])
preprocessor


dtree_model = DecisionTreeClassifier(max_depth=5, random_state=SEED)

pipe_dtree = Pipeline([
    ('prep', preprocessor),
    ('clas', dtree_model)
])


parameters = {}
parameters['clas__max_depth'] = [None, 3, 5, 7, 9]
# parameters['clas__class_weight'] = [None, 'balanced']

GS = GridSearchCV(pipe_dtree, parameters, cv=5, scoring='roc_auc', refit=True)

GS.fit(X, y)

best_pipe_tree = GS.best_estimator_

print("Mejor score: ", GS.best_score_)
print("Mejore configuración de parámetros: ", GS.best_params_)


def show_results(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()


y_pred = best_pipe_tree.predict_proba(X)[:, 1]
show_results(y, y_pred)


logr_model = LogisticRegression(max_iter=1000)

pipe_logr = Pipeline([
    ('prep', preprocessor),
    ('clas', logr_model)
])

pipe_logr


parameters = {}
parameters['clas__C'] = [10e-3, 10e-2, 10e-1, 1, 10, 100, 1000]

GS = GridSearchCV(pipe_logr, parameters, cv=5, scoring='roc_auc', refit=True)

GS.fit(X, y)

print("Mejor score: ", GS.best_score_)
print("Mejore configuración de parámetros: ", GS.best_params_)

best_pipe_logr = GS.best_estimator_


y_pred = best_pipe_logr.predict_proba(X)[:, 1]
show_results(y, y_pred)


from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier()

pipe_gb = Pipeline([
    ('prep', preprocessor),
    ('clas', gb_model)
])

parameters = {}
parameters['clas__learning_rate'] = [0.001, 0.01, 0.1, 1]
parameters['clas__n_estimators'] = [50, 100, 150, 200]

GS = GridSearchCV(pipe_gb, parameters, cv=5, scoring='roc_auc', refit=True)

GS.fit(X, y)

print("Mejor score: ", GS.best_score_)
print("Mejore configuración de parámetros: ", GS.best_params_)

best_pipe_gb = GS.best_estimator_


y_pred = best_pipe_gb.predict_proba(X)[:, 1]
show_results(y, y_pred)


if hasattr(best_pipe_tree.named_steps['clas'], 'feature_importances_'):
    importances = best_pipe_tree.named_steps['clas'].feature_importances_
else:
    print("Model doesn't have feature_importances_")


encoded_features = best_pipe_tree.named_steps['prep'].transformers_[1][1].named_steps['one_hot'].get_feature_names_out(cat_cols)
encoded_features


all_features = np.concatenate([num_cols, encoded_features])

feature_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
})

# Step 5: Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Step 6: Plot with Plotly
fig = px.bar(feature_importance_df, x='Feature', y='Importance', title='Feature Importances',
             labels={'Feature': 'Feature', 'Importance': 'Importance'},
             color='Importance', color_continuous_scale='Viridis')

fig.update_layout(xaxis_tickangle=-45)
fig.show()


svm_model = SVC(probability=True);

pipe_svc = Pipeline([
    ('prep', preprocessor),
    ('clas', svm_model)
])

parameters = {}
parameters['clas__C'] = [10e-2, 1, 100]
parameters['clas__kernel'] = ['linear', 'rbf']

GS = GridSearchCV(pipe_svc, parameters, cv=5, scoring='roc_auc', refit=True)

GS.fit(X, y)

print("Mejor score: ", GS.best_score_)
print("Mejore configuración de parámetros: ", GS.best_params_)

best_pipe_svc = GS.best_estimator_


y_pred = best_pipe_svc.predict_proba(X)[:, 1]
show_results(y, y_pred)


test_preds = best_pipe_gb.predict_proba(df_test)[:, 1] # mejor modelo

submission = pd.DataFrame({'id': df_test.index, 'rainfall': test_preds})
submission.to_csv("submission.csv", index=False)
print("\nFichero a enviar guardado como 'submission.csv'.")




