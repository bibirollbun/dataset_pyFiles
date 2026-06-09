import pandas as pd
import numpy as np
import warnings
import plotly.io as pio

pio.renderers.default = 'iframe'
warnings.filterwarnings('ignore')
SEED = 42
TRAIN_PATH = '/kaggle/input/playground-series-s5e4/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e4/test.csv'
SUB_PATH = '/kaggle/input/playground-series-s5e4/sample_submission.csv'


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sub_df = pd.read_csv(SUB_PATH)
train_df.head()


train_df = train_df.drop(columns=['id'])
test_id = test_df['id']
test_df = test_df.drop(columns=['id'])
train_df.info()


test_df.info()


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED)
train_df.shape, val_df.shape, test_df.shape


# normal preprocessing pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder

train_y = train_df['Listening_Time_minutes']
train_X = train_df.drop(columns=['Listening_Time_minutes'])
val_X = val_df.drop(columns=['Listening_Time_minutes'])
val_y = val_df['Listening_Time_minutes']
numeric_features = train_X.select_dtypes(include=['int64', 'float64']).columns.tolist()
catgorical_features = train_X.select_dtypes(include=['object']).columns.tolist()

pipeline = Pipeline(steps=[
    ('encoder', CatBoostEncoder(cols=catgorical_features,handle_unknown='value')),
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', 'passthrough', catgorical_features)
    ])),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])



train_X_transformed = pipeline.fit_transform(train_X, train_y)
train_X_transformed = pd.DataFrame(train_X_transformed, columns=train_X.columns)
val_X_transformed = pd.DataFrame(pipeline.transform(val_X), columns=train_X.columns)
pipeline.transform(test_df)
test_df_transformed = pd.DataFrame(pipeline.transform(test_df), 
                       columns=test_df.columns)
train_X_transformed.head()


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


plot_train_df = train_X_transformed.copy()
plot_train_df['Listening_Time_minutes'] = train_y
plot_val_df = val_X_transformed.copy()
plot_val_df['Listening_Time_minutes'] = val_y

mean_train = train_y.mean()
std_train = train_y.std()

mean_val = val_y.mean()
std_val = val_y.std()


fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Train Data Distribution', 'Validation Data Distribution')
)

fig.add_trace(
    go.Histogram(
        x=plot_train_df['Listening_Time_minutes'],
        nbinsx=20,
        opacity=0.5,
        name='Train'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Histogram(
        x=plot_val_df['Listening_Time_minutes'],
        nbinsx=20,
        opacity=0.5,
        name='Validation'
    ),
    row=1, col=2
)


fig.add_vline(x=mean_train, line_color='red', line_dash='dash', line_width=2, row=1, col=1)
fig.add_vline(x=mean_train + std_train, line_color='blue', line_dash='dot', line_width=2, row=1, col=1)
fig.add_vline(x=mean_train - std_train, line_color='blue', line_dash='dot', line_width=2, row=1, col=1)


fig.add_vline(x=mean_val, line_color='red', line_dash='dash', line_width=2, row=1, col=2)
fig.add_vline(x=mean_val + std_val, line_color='blue', line_dash='dot', line_width=2, row=1, col=2)
fig.add_vline(x=mean_val - std_val, line_color='blue', line_dash='dot', line_width=2, row=1, col=2)


fig.add_shape(
    type="rect",
    xref="x1",
    yref="paper",
    x0=mean_train - std_train,
    x1=mean_train + std_train,
    y0=0,
    y1=1,
    fillcolor="rgba(150, 0, 255, 0.3)",  # Lighter purple for better visibility
    line=dict(color="rgba(150, 0, 255, 0.8)", width=1),  # Adding border for definition
    layer="below"
)


fig.add_shape(
    type="rect",
    xref="x2",
    yref="paper",
    x0=mean_val - std_val,
    x1=mean_val + std_val,
    y0=0,
    y1=1,
    fillcolor="rgba(150, 0, 255, 0.3)",  
    line=dict(color="rgba(150, 0, 255, 0.8)", width=1),  
    layer="below"
)

fig.update_layout(
    title='Distribution of Listening Time with Bollinger Bands',
    xaxis_title='Listening Time (minutes)',
    xaxis2_title='Listening Time (minutes)',
    yaxis_title='Frequency',
    yaxis2_title='Frequency',
    bargap=0.1,
    showlegend=False,
    height=500,
    width=1000
)

fig.show()


corr_matrix = plot_train_df.corr().round(3)

fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu",
    range_color=[-1, 1]
)
fig.update_layout(
    title='Correlation Matrix',
    width= 900,
    height=700,
    xaxis=dict(tickangle=45),
    yaxis=dict(autorange='reversed')
)
fig.show()


plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(
    data=plot_train_df,
    x='Publication_Time',
    y='Host_Popularity_percentage',
    hue='Listening_Time_minutes',
    palette='RdBu_r', 
    alpha=0.8,
    s=100  
)


sns.regplot(
    data=plot_train_df,
    x='Publication_Time', 
    y='Host_Popularity_percentage',
    scatter=False,  
    line_kws={'color': 'black', 'linestyle': '--'}
)

norm = plt.Normalize(plot_train_df['Listening_Time_minutes'].min(), 
                     plot_train_df['Listening_Time_minutes'].max())
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Listening Time (minutes)')


plt.title('Scatter Plot of Publication Time vs Host Popularity', fontsize=14)
plt.xlabel('Publication Time', fontsize=12)
plt.ylabel('Host Popularity Percentage', fontsize=12)


plt.tight_layout()
plt.show()


fig = make_subplots(
    rows=len(numeric_features)//2 + len(numeric_features)%2, 
    cols=2,
    subplot_titles=numeric_features,
    vertical_spacing=0.08,
    horizontal_spacing=0.05
)

row, col = 1, 1
train_color = px.colors.qualitative.Plotly[0]  # Blue
test_color = px.colors.qualitative.Plotly[1]   # Red

for i, feature in enumerate(numeric_features):
    fig.add_trace(
        go.Box(
            x=train_X[feature],
            name="Train",
            marker=dict(color=train_color),
            boxmean=True,  
            notched=True,  
            boxpoints='outliers',  
            line=dict(width=2),
            offsetgroup=0
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Box(
            x=test_df[feature],
            name="Test",
            marker=dict(color=test_color),
            boxmean=True,
            notched=True,
            boxpoints='outliers',
            line=dict(width=2),
            offsetgroup=1
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text=feature, row=row, col=col)
    

    col += 1
    if col > 2:
        col = 1
        row += 1


fig.update_layout(
    title_text='Distribution of Numeric Features (Train vs Test)',
    title_x=0.5,
    height=500*((len(numeric_features)//2) + (len(numeric_features)%2)),
    width=1000,
    showlegend=True,
    template="plotly_white",  # Use a cleaner template
    margin=dict(t=80, b=40, l=40, r=40),
    plot_bgcolor="rgba(250, 250, 250, 0.9)",
    paper_bgcolor="white"
)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.5)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.5)')

fig.show()


cat_cols = train_X.select_dtypes(include=['object']).columns.tolist()


fig = make_subplots(
    rows=len(cat_cols), 
    cols=1,
    subplot_titles=[f"{col} vs Listening Time" for col in cat_cols],
    vertical_spacing=0.05
)

for i, col in enumerate(cat_cols):
    top_cats = train_X[col].value_counts().nlargest(10).index.tolist()
    filtered_df = train_X[train_X[col].isin(top_cats)]
    
    fig.add_trace(
        go.Box(
            x=filtered_df[col],
            y=plot_train_df['Listening_Time_minutes'],
            name=col
        ),
        row=i+1, col=1
    )

fig.update_layout(
    title_text='Listening Time by Categorical Features (Top Categories)',
    height=400*len(cat_cols),
    width=1000,
    showlegend=False
)
fig.show()


corr_with_target = corr_matrix['Listening_Time_minutes'].abs().sort_values(ascending=False)
top_features = corr_with_target[1:5].index.tolist()  # Skip the target itself

# Create scatter plot matrix
fig = px.scatter_matrix(
    plot_train_df,
    dimensions=top_features,
    color='Listening_Time_minutes',
    color_continuous_scale='Plotly3',
    opacity=0.7
)
fig.update_traces(marker=dict(size=5))
fig.update_layout(
    title='Scatter Matrix of Top Correlated Features',
    width=1000,
    height=1000
)
fig.show()


import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm
import optuna
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)
FOLD = 5
EARLY_STOPPING = 50
TRIALS = 1 # for efficiency

def objective(trial):

    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 500, 2000, step=100),
        'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
        'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
        'subsample': trial.suggest_float('lgb_subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True),
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': SEED,
        'verbose': -1

    }

    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 500, 2000, step=100),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': SEED,
        'verbosity': 0
    }

    cb_params = {
        'iterations': trial.suggest_int('cb_iterations', 500, 2000, step=100),
        'learning_rate': trial.suggest_float('cb_learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('cb_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('cb_min_data_in_leaf', 1, 20),
        'rsm': trial.suggest_float('cb_rsm', 0.5, 1.0),  # feature subsample ratio
        'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1e-8, 10.0, log=True),
        'loss_function': 'RMSE',
        'random_seed': SEED,
        'verbose': False
    }
    

    lgb_weight = trial.suggest_float('lgb_weight', 0.1, 0.8)
    xgb_weight = trial.suggest_float('xgb_weight', 0.1, 0.8)

    cb_weight = 1.0 - (lgb_weight + xgb_weight)
    
    kf = KFold(n_splits=FOLD, shuffle=True, random_state=SEED)
    cv_scores = []
    fold_count = 0
    for train_index, val_index in kf.split(train_X_transformed):
        X_train, X_val = train_X_transformed.iloc[train_index], train_X_transformed.iloc[val_index]
        y_train, y_val = train_y.iloc[train_index], train_y.iloc[val_index]
        fold_count += 1
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)]
        )
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING,
            verbose=False
        )
        
        cb_model = cb.CatBoostRegressor(**cb_params)
        cb_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=EARLY_STOPPING,
            verbose=False
        )
        
        lgb_pred = lgb_model.predict(X_val)
        xgb_pred = xgb_model.predict(X_val)
        cb_pred = cb_model.predict(X_val)
        
        ensemble_pred = (lgb_pred * lgb_weight) + (xgb_pred * xgb_weight) + (cb_pred * cb_weight)
        
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        cv_scores.append(rmse)
        print('Fold-{}/{} finished'.format(fold_count, FOLD))

    return np.mean(cv_scores)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)

print(f"\nBest ensemble RMSE: {study.best_value:.4f}")
print("Best hyperparameters:")
for param, value in study.best_params.items():
    print(f"    {param}: {value}")

lgb_weight = study.best_params['lgb_weight']
xgb_weight = study.best_params['xgb_weight']
cb_weight = 1.0 - (lgb_weight + xgb_weight)
print(f"\nEnsemble weights:")
print(f"    LightGBM: {lgb_weight:.4f}")
print(f"    XGBoost: {xgb_weight:.4f}")
print(f"    CatBoost: {cb_weight:.4f}")



def train_ensemble_model(params, X_train, y_train):
    
    lgb_params = {k[4:]: v for k, v in params.items() if k.startswith('lgb_') and k != 'lgb_weight'}
    
    xgb_params = {k[4:]: v for k, v in params.items() if k.startswith('xgb_') and k != 'xgb_weight'}

    cb_params = {k[3:]: v for k, v in params.items() if k.startswith('cb_')}

    
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    
    cb_model = cb.CatBoostRegressor(**cb_params)
    cb_model.fit(X_train, y_train, verbose=False)
    
    return lgb_model, xgb_model, cb_model

def predict_ensemble(models, weights, X):
    lgb_model, xgb_model, cb_model = models
    lgb_pred = lgb_model.predict(X)
    xgb_pred = xgb_model.predict(X)
    cb_pred = cb_model.predict(X)
    return (lgb_pred * weights[0]) + (xgb_pred * weights[1]) + (cb_pred * weights[2])


final_models = train_ensemble_model(
    study.best_params, 
    train_X_transformed, 
    train_y
)

ensemble_weights = [lgb_weight, xgb_weight, cb_weight]
val_pred = predict_ensemble(final_models, ensemble_weights, val_X_transformed)

rmse = np.sqrt(mean_squared_error(val_y, val_pred))
mae = mean_absolute_error(val_y, val_pred)
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation MAE: {mae:.4f}")

test_pred = predict_ensemble(final_models, ensemble_weights, test_df_transformed)


sub_df['Listening_Time_minutes'] = test_pred
sub_df.to_csv('submission.csv', index=False)
sub_df.shape

