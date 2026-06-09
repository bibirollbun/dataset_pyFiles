pip install "optuna-integration[xgboost]"


import pandas as pd
import numpy as np
import os 
import time 
import math
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from category_encoders import TargetEncoder
import plotly.subplots as sp
import plotly.figure_factory as ff  

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import KFold, cross_val_score
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import optuna
pio.renderers.default = 'iframe_connected'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head()


train.info()


train.describe().round(2)


print("Duplicated Rows:",train.duplicated().sum())
print("-"*30)
print("Number of Rows:",train.shape[0])
print("-"*30)
print("Number of Columns:",train.shape[1])


train.isnull().sum()


print("Numeric Col Names",train.select_dtypes(include=['number']).columns)


print("Categorical Col Names",train.select_dtypes(include=['object']).columns)


num_col= ['num_lanes', 'curvature', 'speed_limit',
          'num_reported_accidents']
target_col = 'accident_risk'
cat_col = ['road_type', 'lighting', 'weather', 'time_of_day']
bin_col = ['road_signs_present', 'public_road', 'holiday', 'school_season']


for col in cat_col:
    print(f"Unique categories in '{col}' column: {train[col].unique()}")
    print("<--- --- --- --- --- --- --- --- --- --->\n")


weather_risk = train.groupby('weather')['accident_risk'].mean().sort_values(ascending=False)
print(f"\nHighest Risk Weather: {weather_risk.index[0]} (Avg Risk: {weather_risk.iloc[0]:.3f})")
print(f"Lowest Risk Weather: {weather_risk.index[-1]} (Avg Risk: {weather_risk.iloc[-1]:.3f})")


road_risk = train.groupby('road_type')['accident_risk'].mean().sort_values(ascending=False)
print(f"\nHighest Risk Road Type: {road_risk.index[0]} (Avg Risk: {road_risk.iloc[0]:.3f})")
print(f"Lowest Risk Road Type: {road_risk.index[-1]} (Avg Risk: {road_risk.iloc[-1]:.3f})")


time_risk = train.groupby('time_of_day')['accident_risk'].mean().sort_values(ascending=False)
print(f"\nHighest Risk Time: {time_risk.index[0]} (Avg Risk: {time_risk.iloc[0]:.3f})")
print(f"Lowest Risk Time: {time_risk.index[-1]} (Avg Risk: {time_risk.iloc[-1]:.3f})")


print(f"\nBINARY VARIABLES IMPACT:")
for col in bin_col:
    risk_diff = train.groupby(col)['accident_risk'].mean().diff().iloc[-1]
    mean_true = train[train[col] == True]['accident_risk'].mean()
    mean_false = train[train[col] == False]['accident_risk'].mean()
    print(f"  {col}: True={mean_true:.3f}, False={mean_false:.3f}, Diff={risk_diff:.3f}")


road_type_counts = train['road_type'].value_counts().reset_index()
road_type_counts.columns = ['road_type', 'count']

fig = px.bar(
    road_type_counts,
    x='road_type',
    y='count',
    title='Road Type Distribution',
    color='road_type',  
    color_discrete_sequence=px.colors.qualitative.T10  
)

fig.update_layout(
    width=500,
    height=400,
    plot_bgcolor='black',  
    paper_bgcolor='black',  
    font_color='white'
)

fig.show()


lighting_counts = train['lighting'].value_counts().reset_index()
lighting_counts.columns = ['lighting', 'count']

fig = px.bar(
    lighting_counts,
    x='lighting',
    y='count',
    title='Lighting Distribution',
    color='lighting',  
    color_discrete_sequence=px.colors.qualitative.T10  
)

fig.update_layout(
    width=500,
    height=400,
    plot_bgcolor='black',  
    paper_bgcolor='black',  
    font_color='white'
)

fig.show()


weather_counts = train['weather'].value_counts().reset_index()
weather_counts.columns = ['weather', 'count']

fig_weather = px.bar(
    weather_counts,
    x='weather',
    y='count',
    title='Weather Condition Distribution',
    color='weather',  
    color_discrete_sequence=px.colors.qualitative.T10  
)

fig_weather.update_layout(
    width=500,
    height=400,
    plot_bgcolor='black',  
    paper_bgcolor='black',  
    font_color='white'
)

fig_weather.show()


time_of_day_counts = train['time_of_day'].value_counts().reset_index()
time_of_day_counts.columns = ['time_of_day', 'count']

fig_time = px.bar(
    time_of_day_counts,
    x='time_of_day',
    y='count',
    title='Time of Day Distribution',
    color='time_of_day',  
    color_discrete_sequence=px.colors.qualitative.T10  
)

fig_time.update_layout(
    width=500,
    height=400,
    plot_bgcolor='black',  
    paper_bgcolor='black',  
    font_color='white'
)

fig_time.show()


cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']


df_melted = train[cols].melt(var_name="Category", value_name="Value")
counts = df_melted.groupby(["Category", "Value"]).size().reset_index(name="Count")

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type':'domain'}, {'type':'domain'}],
           [{'type':'domain'}, {'type':'domain'}]],
    subplot_titles=cols
)

for i, col in enumerate(cols):
    row = i // 2 + 1
    col_pos = i % 2 + 1
    df_col = counts[counts['Category'] == col]
    fig.add_trace(
        go.Pie(
            labels=df_col['Value'],
            values=df_col['Count'],
            marker_colors=px.colors.qualitative.T10,
            showlegend=(i==0)  
        ),
        row=row,
        col=col_pos
    )

fig.update_layout(
    title_text='Distribution of Binary Values',
    width=600,
    height=600,
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white'
)

fig.show()


sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

plt.figure(figsize=(6, 4))
sns.histplot(train[target_col], bins=50, kde=True, color='steelblue')
plt.title('Distribution of Accident Risk')
plt.xlabel(target_col)
plt.ylabel('Count')
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(9, 8))  

for i, col in enumerate(num_col):
    row = i // 2
    col_pos = i % 2
    sns.histplot(train[col], bins=50, kde=True, color='steelblue', ax=axes[row, col_pos])
    axes[row, col_pos].set_title(f'Distribution of {col}')
    axes[row, col_pos].set_xlabel(col)
    axes[row, col_pos].set_ylabel('Count')

plt.tight_layout()
plt.show()


road_accidents = train.groupby('road_type')['num_reported_accidents'].sum().reset_index()

fig_road_pie = px.pie(
    road_accidents,
    names='road_type',               
    values='num_reported_accidents', 
    title='Accidents by Road Type',
    color='road_type',
    color_discrete_sequence=px.colors.qualitative.T10
)

fig_road_pie.update_layout(
    width=500,
    height=400,
    plot_bgcolor='black',  
    paper_bgcolor='black',  
    font_color='white'
)

fig_road_pie.show()


categorical_cols = train.select_dtypes(include=["bool"]).columns.tolist()

base_colors = px.colors.qualitative.T10

n_cols = 2
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
plt.figure(figsize=(14, 5 * n_rows))

for i, col in enumerate(categorical_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    
    unique_vals = train[col].unique()
    palette = {
        val: base_colors[j % len(base_colors)] 
        for j, val in enumerate(unique_vals)
    }
    
    sns.boxplot(
        x=col, 
        y="accident_risk", 
        data=train, 
        palette=palette
    )
    
    plt.xticks(rotation=30)
    plt.title(f"Accident Risk vs {col}")

plt.tight_layout()
plt.show()


numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ["accident_risk", "id"]]

base_colors = px.colors.qualitative.T10
n_cols = 2
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
plt.figure(figsize=(14, 5 * n_rows))
plt.suptitle("Accident Risk vs Numerical Features", fontsize=16, fontweight="bold", y=1.02)

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    unique_vals = train["road_type"].unique()
    palette = {val: base_colors[j % len(base_colors)] for j, val in enumerate(unique_vals)}
    sns.scatterplot(x=col, y="accident_risk", hue="road_type", data=train, alpha=0.5, palette=palette)
    plt.title(f"{col}")
    plt.xlabel(col)
    plt.ylabel("Accident Risk")
    plt.legend(title="road_type", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

road_type_risk = train.groupby('road_type')['accident_risk'].mean().sort_values()
weather_risk = train.groupby('weather')['accident_risk'].mean().sort_values()
time_risk = train.groupby('time_of_day')['accident_risk'].mean().sort_values()
lighting_risk = train.groupby('lighting')['accident_risk'].mean().sort_values()

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Road Type', 'Weather', 'Time of Day', 'Lighting'),
    horizontal_spacing=0.15,
    vertical_spacing=0.15
)

fig.add_trace(
    go.Bar(x=road_type_risk.values, y=road_type_risk.index, orientation='h', marker_color=colors[0]),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=weather_risk.values, y=weather_risk.index, orientation='h', marker_color=colors[1]),
    row=1, col=2
)

time_risk_sorted = time_risk.sort_values(ascending=False)
fig.add_trace(
    go.Bar(x=time_risk_sorted.values, y=time_risk_sorted.index, orientation='h', marker_color=colors[2]),
    row=2, col=1
)

fig.add_trace(
    go.Bar(x=lighting_risk.values, y=lighting_risk.index, orientation='h', marker_color=colors[3]),
    row=2, col=2
)

fig.update_layout(
    height=700,
    width=700,
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    showlegend=False,
    title_text="Accident Risk Analysis With Category Columns",
    title_x=0.5
)

fig.show()


time_risk = train.groupby('time_of_day')['accident_risk'].agg(['mean', 'std', 'count']).reset_index()

time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
if set(time_risk['time_of_day']).issubset(set(time_order)):
    time_risk['time_of_day'] = pd.Categorical(time_risk['time_of_day'], categories=time_order, ordered=True)
    time_risk = time_risk.sort_values('time_of_day')

fig = px.bar(
    time_risk,
    x='time_of_day',
    y='mean',
    color='time_of_day',
    color_discrete_sequence=px.colors.qualitative.T10,
    text=time_risk['mean'].round(3),
    title='Average Accident Risk by Time of Day'
)

fig.update_traces(textposition='outside')
fig.update_layout(
    width=500,
    height=400,
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(color='white'),
    xaxis=dict(title='Time of Day', showgrid=False, color='white'),
    yaxis=dict(title='Average Accident Risk', color='white'),
)

fig.show()


numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
corr_matrix = train[numerical_cols + ['accident_risk']].corr()

plt.figure(figsize=(5,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
target_col = 'accident_risk'


for c in bool_cols:
    if c in train.columns or test.columns:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)


def feature_engineering(df):

    df = df.copy()

    # Basic interactions
    df['speed_x_curvature'] = df['speed_limit'] * df['curvature']
    df['curv_speed'] = df['curvature'] * df['speed_limit']  
    df['curvature_squared'] = df['curvature'] ** 2
    df['speed_sq'] = df['speed_limit'] ** 2

    # Nonlinear transforms & logs (safe)
    df['accidents_log'] = np.log1p(df['num_reported_accidents'])
    df['curv_log'] = np.log1p(df['curvature'])
    df['speed_log'] = np.log1p(df['speed_limit'])
    df['inv_speed'] = 1.0 / (df['speed_limit'] + 1.0)

    # Ratios / density per lane
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['curv_per_lane'] = df['curvature'] / (df['num_lanes'] + 1)
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)

    # Combined risk indices
    df['danger_score'] = (df['speed_limit'] / 100.0) * (df['curvature'] ** 2)
    df['risk_density'] = df['curv_speed'] / (df['num_lanes'] + 1.0)
    df['accident_density'] = df['accidents_per_lane'] * df['speed_x_curvature']

    # Polynomial / smoother mixes
    # Use np.where to protect from negative inside sqrt though curvature and speed_limit are non-negative in domain
    df['poly_mix1'] = np.sqrt(np.maximum(df['curvature'] * df['speed_limit'], 0))
    df['poly_mix2'] = (df['num_reported_accidents'] ** 0.3) * df['speed_limit']

    # Statistical combos
    df['risk_index'] = (df['curv_speed'] * df['accidents_per_lane']) / (df['speed_limit'] + 1.0)
    df['stability_score'] = (df['num_lanes'] / (1.0 + df['curvature'])) * df['speed_limit']

    # Binary derived flags
    df['tight_lane'] = (df['num_lanes'] <= 2).astype(int)
    df['sharp_curve'] = (df['curvature'] > 0.6).astype(int)
    df['high_speed_zone'] = (df['speed_limit'] > 80).astype(int)
    df['critical_zone'] = ((df['sharp_curve'] == 1) & (df['high_speed_zone'] == 1)).astype(int)

    # Clean up infinite / extremely large values (if any)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df



train = feature_engineering(train)
test = feature_engineering(test)


train.head()


train.info()


X = train.drop(columns=['id', 'accident_risk'])
y = train['accident_risk']
X_test = test.drop(columns=['id'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial, X_train_raw, y_train, cat_cols):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'verbosity': 0,
        'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 5e-2),
        'subsample': trial.suggest_uniform('subsample', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.3),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 10),
    }

    # Single validation split to avoid repeated step reporting
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42
    )

    # Target encoding for categorical features
    te = TargetEncoder(cols=cat_cols, smoothing=10.0)
    X_tr[cat_cols] = te.fit_transform(X_tr[cat_cols], y_tr)
    X_val[cat_cols] = te.transform(X_val[cat_cols])

    model = XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
        callbacks=[XGBoostPruningCallback(trial, "validation_0-rmse")]
    )

    y_pred = model.predict(X_val)
    cv_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Trial {trial.number} | CV RMSE: {cv_rmse:.6f} | Params: {trial.params}")
    return cv_rmse


#study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1))
#study.optimize(lambda trial: objective(trial, X_train, y_train, cat_cols), n_trials=100)


#print("Best RMSE:", study.best_value)
#print("Best hyperparameters:", study.best_params)


best_params = {
    'n_estimators': 2529,
    'max_depth': 11,
    'learning_rate': 0.04411481270144032,
    'subsample': 0.9327095844859239,
    'gamma': 0.02054596944973483,
    'colsample_bytree': 0.8858667578016076,
    'min_child_weight': 4,
    'reg_alpha': 3.9416720899763574e-05,
    'reg_lambda': 0.04064930109333212,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'verbosity': 0
}


kf = KFold(n_splits=10, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
rmse_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Target encoding
    te = TargetEncoder(cols=cat_cols, smoothing=10.0)
    X_tr[cat_cols] = te.fit_transform(X_tr[cat_cols], y_tr).astype(float)
    X_val[cat_cols] = te.transform(X_val[cat_cols]).astype(float)
    
    # Model training
    model = XGBRegressor(**best_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    
    # Validation prediction and RMSE
    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_list.append(rmse)
    
    # Test prediction
    X_test_encoded = X_test.copy()
    X_test_encoded[cat_cols] = te.transform(X_test_encoded[cat_cols]).astype(float)
    test_preds += model.predict(X_test_encoded) / kf.n_splits
    
    print(f"Fold {fold+1} RMSE: {rmse:.6f}")

print(f"Mean CV RMSE: {np.mean(rmse_list):.6f}")


plt.figure(figsize=(12, 8))
plot_importance(model, max_num_features=30, importance_type='weight')
plt.title("Top 30 Feature Importances")
plt.show()


test.head()


submission['accident_risk'] = test_preds


submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved!")


submission.head()


print("\nðŸ“Š Prediction Summary:")
print(submission["accident_risk"].describe().round(2))


plt.figure(figsize=(8,5))
plt.hist(submission["accident_risk"], bins=50, edgecolor="black")
plt.title("Distribution of Predictions")
plt.xlabel("Predicted Target")
plt.ylabel("Frequency")
plt.show()

