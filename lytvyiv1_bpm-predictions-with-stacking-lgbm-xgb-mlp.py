!pip install "optuna-integration[lightgbm]"


# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib import gridspec
import seaborn as sns

# Scikit-learn: preprocessing, models, metrics, utilities
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

import xgboost as xgb
import lightgbm as lgb
# Warnings
import warnings
warnings.filterwarnings("ignore")

# optuna
import optuna
from optuna.integration import LightGBMPruningCallback
#Rich
from rich.console import Console
from rich.table import Table

from scipy.stats import chi2_contingency

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy



df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', delimiter = ',')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', delimiter = ',')


df_train.tail()


df_train.shape


df_train = df_train.drop('id', axis = 1)

def print_nan(df):
    result = pd.DataFrame({
        'columns': df.columns,
        'NaN count': df.isna().sum().values,
        'NaN %': (df.isna().sum().values / len(df)) * 100
    })
    display(result)

df_train.info()
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print_nan(df_train)


df_train['TrackDurationMs'].describe()


def transform_ms_to_minutes(df):
    df['TrackDurationMs'] = df['TrackDurationMs'] / 1000
    df['TrackDurationMs'] = df['TrackDurationMs'] / 60
    df = df.rename(columns={'TrackDurationMs': 'TrackDurationMinutes'})
    return df


def split_data(Xdata, Ydata, test_size = 0.2, random_seed = 42):
    # Split into training and test 
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, test_size=test_size, random_state=random_seed)

    print(f"Train shape, X: {Xtrain.shape}, y: {Ytrain.shape}")
    print(f"Test shape, X: {Xtest.shape}, y: {Ytest.shape}")

    return Xtrain, Xtest, Ytrain, Ytest
    
X = df_train.drop('BeatsPerMinute', axis = 1)
y = df_train['BeatsPerMinute']

Xtrain, Xtest, ytrain, ytest = split_data(X, y)


Xtrain = transform_ms_to_minutes(Xtrain)


def visualize_continious_data(continuous, Xdata, Ydata):
    count = len(continuous)
    
    # Create figure with enough rows (two plots per feature)
    fig = plt.figure(figsize=(16, 5 * count), constrained_layout=True)

    # Define grid layout: 2 columns per feature (histogram + violin plot)
    spec = gridspec.GridSpec(nrows=count, ncols=3, figure=fig)

    # Generate a color palette for all continuous features
    palette = sns.color_palette("husl", n_colors=len(continuous))

    for i, feature_name in enumerate(continuous):
        # Left subplot: histogram for distribution of feature values
        hist = fig.add_subplot(spec[i, 0])
        data = Xdata[feature_name]
        
        sns.histplot(data=data, label='Histogram', bins=14, color=palette[i], kde = True)
        hist.set_xlabel('Value')       # X-axis label
        hist.set_ylabel("Count")       # Y-axis label
        hist.set_title(f'Distribution of {feature_name}')  # Title
        hist.legend()                  # Add legend
        hist.grid()                    # Enable grid lines

        # Middle subplot: Boxplot
        box = fig.add_subplot(spec[i, 1])
        sns.boxplot(x=data, ax=box, color=palette[i])
        box.set_ylabel(feature_name)       # Target on Y-axis
        box.set_xlabel('Values')       # Target on Y-axis
        box.set_title(f'Boxplot of {feature_name}')  # Title
        
        # Right subplot: violin plot to compare feature distribution by target class
        scatter = fig.add_subplot(spec[i, 2])
        sns.scatterplot( x = data, y= Ydata, label='RUL', color=palette[i], alpha=0.6)
        scatter.set_xlabel(feature_name)       # Feature on X-axis
        scatter.set_ylabel("Target value")       # Target on Y-axis
        scatter.set_title(f'Scatterplot of {feature_name}')  # Title

    plt.show()



visualize_continious_data(Xtrain.columns, Xtrain, ytrain)


def print_target(target):
    # Create figure with enough rows (two plots per feature)
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    
    # Define grid layout: 2 columns per feature (histogram + violin plot)
    spec = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)
    hist = fig.add_subplot(spec[0, 0])
    data = target
    
    sns.histplot(data=data, label='Histogram', bins=14, kde = True)
    hist.set_xlabel('Value')       # X-axis label
    hist.set_ylabel("Count")       # Y-axis label
    hist.set_title(f'Distribution of target')  # Title
    hist.legend()                  # Add legend
    hist.grid()                    # Enable grid lines
    
    box = fig.add_subplot(spec[0, 1])
    sns.boxplot(x=data, ax=box)
    box.set_ylabel('target')       # Target on Y-axis
    box.set_xlabel('Values')       # Target on Y-axis
    box.set_title(f'Boxplot of target')  # Title
    plt.show()


print_target(ytrain)


display(Xtrain.describe().T)
print(">>>>>>>>>>>>>>>>>>>>>>>>>TARGET<<<<<<<<<<<<<<<<<<<<<<<<")
ytrain.describe()


def plot_heatmap(Xtrain, cols, name, method='pearson'):
    plt.figure(figsize = (12, 10))
    corr_matrix = Xtrain[cols].corr(method=method)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title(f"Matrix of {method} correlations of {name} features")
    plt.show()


plot_heatmap(Xtrain, Xtrain.columns, 'numeric', 'spearman')


def print_correlation_with_target(Xdata, target, numeric, categories, method_for_continuous_data = 'pearson'):
    console = Console()
    table = Table(
        title="Correlations of features with target variable",
        show_header=True,
        header_style="bold magenta",
        highlight=True,
        show_lines=True
    )
    
    table.add_column("Feature Name", style="cyan", justify="left")
    table.add_column("Correlation", style="cyan", justify="left")
    if len(numeric) > 0:
        for col in numeric:
            correlation = Xtrain[col].corr(target, method = method_for_continuous_data)
            table.add_row(col, f"{correlation:.4f}")
    if len(categories) > 0:
        for col in categories:
            correlation = cramers_v(Xtrain[col], target)
            table.add_row(col, f"{correlation:.4f}")

    console.print(table)

print_correlation_with_target(Xtrain, ytrain, Xtrain.columns, [], 'spearman')


prev_cols = np.array(Xtrain.columns)

def compute_bin_edges(train_col, q):
    edges = np.quantile(train_col, q)
    edges = np.unique(edges) 
    edges = np.concatenate(([-np.inf], edges, [np.inf]))
    return edges


def create_features(df, quartile_edges, decile_edges):
    stats = []

    # Rank Features 

    for col in df.columns:
        df[f"{col}_rank"] = df[col].rank(pct=True).astype(np.float32)


    # Quartile Features

    for col in df.columns:
        if quartile_edges is not None and col in quartile_edges:
            edges = quartile_edges[col]
            df[f"{col}_quartile"] = np.digitize(df[col], edges) - 1
            stats.append(f"{col}_quartile")


    # Decile Features

    for col in df.columns:
        if decile_edges is not None and col in decile_edges:
            edges = decile_edges[col]
            df[f"{col}_decile"] = np.digitize(df[col], edges) - 1
            stats.append(f"{col}_decile")


    # Other engineered features


    df['audio_score'] = np.log(df['RhythmScore'] * df['InstrumentalScore'])

    df['vocal_energy'] = np.log(df['Energy'] * df['VocalContent'])

    df['rhythm_energy'] = np.log1p(df['Energy'] * df['RhythmScore'])

    df['loudness_energy'] = np.log1p(np.abs(df['Energy'] * df['AudioLoudness'])) * (-1)

    df['mood'] = np.log1p(df['MoodScore'] * df['RhythmScore'])

    df['energy_per_minute'] = df['Energy'] / df['TrackDurationMinutes']

    df['rhythm_squared'] = df['RhythmScore'] ** 2

    df['energy_squared'] = df['Energy'] ** 2

    return stats



# 1. Calculate the edges on train
quartile_edges = {col: compute_bin_edges(Xtrain[col], [0.25,0.5,0.75]) for col in Xtrain.columns}
decile_edges = {col: compute_bin_edges(Xtrain[col], [i/10 for i in range(1,10)]) for col in Xtrain.columns}

# 2. Feature creating 
stats = create_features(Xtrain, quartile_edges, decile_edges)

new_cols = np.array(list(set(Xtrain.columns) - set(prev_cols) - set(stats))) #For visualization
#new_cols = np.concatenate([new_cols, np.array(['InstrumentalScore', 'AudioLoudness'])])


# local_stats = []
# for col in prev_cols: #For mlp
#     local_stats.append(f"{col}_mean_decile")
#     local_stats.append(f"{col}_std_decile")


visualize_continious_data(new_cols, Xtrain, ytrain)


# stats = [x for x in stats if x not in local_stats]


# Xtrain[local_stats].describe().T


plot_heatmap(Xtrain, new_cols, 'numeric', 'spearman')


print_correlation_with_target(Xtrain, ytrain, new_cols, [], 'spearman')


def objective_lgbm(trial, X, y, n_splits = 5):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1600),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-7, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 120),
        'objective': 'regression',
        'metric': 'l2',
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }


    cv = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    val_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)

        pruning_callback = LightGBMPruningCallback(trial, "l2")
        
        callbacks = [
            #lgb.early_stopping(stopping_rounds = 50, verbose = False),
            #lgb.log_evaluation(0),
            pruning_callback
        ]

        model.fit(X_train, y_train,
                 eval_set = [(X_val, y_val)],
                 eval_metric = 'l2',
                 callbacks = callbacks)
        y_pred = model.predict(X_val)
        val_score = mean_squared_error(y_val, y_pred, squared=False)
        val_scores.append(val_score)

    return np.mean(val_scores)


def objective_xgb(trial, X, y, n_splits = 5):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1600),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1, 20),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'random_state': 42,
        'tree_method': 'hist'  
    }
    
    model = xgb.XGBRegressor(**params)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring = 'neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean()


def print_importances(model, Xtrain):
    importances = model.feature_importances_
    feature_names = Xtrain.columns
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(8, 4))
    plt.barh(feat_df['Feature'].head(20), feat_df['Importance'].head(20))
    plt.gca().invert_yaxis()
    plt.title('Top Feature Importances')
    plt.show()


TRAIN = False
if TRAIN:
    study_lgbm = optuna.create_study(direction='minimize')
    study_lgbm.optimize(lambda trial: objective_lgbm(trial, Xtrain, ytrain, 3), n_trials = 100)
    
    print("The best parameters for LGBM:")
    print(study_lgbm.best_params)


best_params = {
        'n_estimators': 1527,
        'boosting_type': 'gbdt',
        'max_depth': 10,
        'learning_rate': 0.0026869848810714947,
        'num_leaves': 27,
        'subsample': 0.9981473099886489,
        'colsample_bytree': 0.6084821726228951,
        'reg_alpha': 0.00031268097429332654,
        'reg_lambda': 1.315765917692552e-06,
        'min_child_samples': 93,
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'n_jobs': -1,
    }

model_lgbm = lgb.LGBMRegressor(**best_params, random_state = 42)
model_lgbm.fit(Xtrain, ytrain)
print(mean_squared_error(ytrain, model_lgbm.predict(Xtrain), squared=False ))
print_importances(model_lgbm, Xtrain)    


if TRAIN:
    study_xgb = optuna.create_study(direction = 'minimize')
    study_xgb.optimize(lambda trial : objective_xgb(trial, Xtrain, ytrain, n_splits = 3), n_trials=50)
    print("The best paramters for XGBoost:")
    print(study_xgb.best_params)


best_params = {
        'n_estimators': 787,
        'max_depth': 6,
        'learning_rate': 0.001948392016417585,
        'subsample': 0.6703445429015104,
        'colsample_bytree': 0.7361366574150289,
        'gamma': 4.09139105373002,
        'reg_alpha': 12.41230859155185,
        'reg_lambda': 7.618773266241725,
        'min_child_weight': 1
}

model_xgb = xgb.XGBRegressor(**best_params, random_state = 42)
model_xgb.fit(Xtrain, ytrain)
print(mean_squared_error(ytrain, model_xgb.predict(Xtrain), squared=False))
print_importances(model_xgb, Xtrain)


cols_to_scale = [
    #'difficulty_level',
    'loudness_energy',
    'vocal_energy',
    #'AudioLoudness_mean_decile',
    #'perfomance_difficult',
    'audio_score',
    'AudioLoudness',
    'TrackDurationMinutes'
]

scaler = MinMaxScaler()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.MSELoss()


def create_loader(Xdata, ydata, batch_size=64, train=False):
    Xtensor = torch.tensor(Xdata.values, dtype=torch.float32)
    if ydata is not None:
        ytensor = torch.tensor(ydata.values, dtype=torch.float32)
        dataset = TensorDataset(Xtensor, ytensor)
    else:
        dataset = TensorDataset(Xtensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train if ydata is not None else False)



class MyMLP(nn.Module):
    def __init__(self, input_size, output_size, 
                 hidden_layers = [256, 128], 
                activation_function = nn.ReLU,
                dropout_rate = 0.3):
        super(MyMLP, self).__init__()
        self.activation_fc = activation_function
        layers = []
        in_dim = input_size
        for l in hidden_layers:
            layers.append(nn.Linear(in_dim, l))
            layers.append(self.activation_fc())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = l
        layers.append(nn.Linear(in_dim, output_size))
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)
        


def evaluate_mlp(model, loader, criterion, device, has_labels=True):
    model.eval()
    losses, y_true, y_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            if has_labels:
                x, y = batch
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                losses.append(loss.item())
                y_true.extend(y.cpu().numpy().flatten())
            else:
                (x,) = batch
                x = x.to(device)
                preds = model(x)
            y_pred.extend(preds.cpu().numpy().flatten())

    if has_labels:
        error = sum(losses) / len(losses)
        return error, y_true, y_pred
    else:
        return y_pred



def fit_mlp(model, config, want_to_print = False):
    history = {'train_error': [], 'val_error': []}
    train_losses = []
    best_val_error = np.inf
    
    train_loader = config['train_loader']
    val_loader = config['val_loader']
    criterion = config['criterion']
    optim = config['optim']
    device = config['device']
    n_epochs = config['n_epochs']

    epochs_without_improvement = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optim.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        train_error = sum(train_losses) / len(train_losses)
        val_error, _, _ = evaluate_mlp(model, val_loader, criterion, device)
        history['train_error'].append(train_error)
        history['val_error'].append(val_error)
        if val_error < best_val_error:
            best_val_error = val_error
            best_params = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement+=1
        if want_to_print and (epoch == 1 or epoch % 10 == 0):
            print(f'Epoch {epoch}/{n_epochs} - Train error: {train_error:.4f}, Val error: {val_error:.4f}')

        if epochs_without_improvement >= 8: #early stopping
            break

    model.load_state_dict(best_params)
    return model, history


def find_best_params(X, y, layers_array, act_functions, dropouts, device, criterion,
                     cols_to_scale = [], scaler = None, n_epochs = 50):
    
    #cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    best_val_error = np.inf
    best_layers, best_act_fc = None, None

    for layers in layers_array:
        for act_fc in act_functions:
            for dropout in dropouts:
                val_loss = 0.0
    
                if scaler is not None and len(cols_to_scale) > 0:
                    X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
                    X_val.loc[:, cols_to_scale] = scaler.transform(X_val[cols_to_scale])
        
                train_loader = create_loader(X_train, y_train, train=True)
                val_loader = create_loader(X_val, y_val)
                
                model = MyMLP(
                    input_size=X.shape[1], 
                    output_size=1,
                    hidden_layers=layers,
                    activation_function=act_fc,
                    dropout_rate = dropout
                ).to(device)
                
                optim = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
                
                _, history = fit_mlp(model, config={
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'criterion': criterion,
                    'optim': optim,
                    'device': device,
                    'n_epochs': n_epochs
                })
                
                val_loss = min(history['val_error'])
    
                print(f"Validation loss: {val_loss:.4f} | layers={layers}, activation={act_fc.__name__}, dropout_rate={dropout}")
                
                if val_loss < best_val_error:
                    best_val_error = val_loss
                    best_layers = layers
                    best_act_fc = act_fc
                    best_dropout = dropout
                
    return best_layers, best_act_fc, best_dropout



Xtrain_mlp = Xtrain.drop(stats, axis = 1)


if TRAIN:
    layers_array = [
        [128, 64],
        [256, 128],
        [512, 256],
        [512, 256, 128]
    ]
    act_functions = [
        nn.ReLU
    ]
    dropouts = [ 
        0.3 
    ]
    best_layers, _, _ = find_best_params(Xtrain_mlp, ytrain, layers_array, act_functions, dropouts, DEVICE, CRITERION, 
                                                 cols_to_scale = cols_to_scale, scaler = scaler, n_epochs = 30)
    print(best_layers)


if TRAIN:
    layers_array = [
        [512, 256, 128]
    ]
    act_functions = [
        nn.ReLU,
        nn.LeakyReLU,
        nn.Sigmoid,
        nn.Tanh
    ]
    dropouts = [ 
        0.3 
    ]
    _, best_act_fc, _ = find_best_params(Xtrain_mlp, ytrain, layers_array, act_functions, dropouts, DEVICE, CRITERION, 
                                                 cols_to_scale = cols_to_scale, scaler = scaler, n_epochs = 30)
    print(best_act_fc.__name__)


if TRAIN:
    layers_array = [
        [512, 256, 128]
    ]
    act_functions = [
        nn.Sigmoid
    ]
    dropouts = [
        0.2, 
        0.25, 
        0.3, 
        0.35
    ]
    _, _, best_dropout = find_best_params(Xtrain_mlp, ytrain, layers_array, act_functions, dropouts, DEVICE, CRITERION, 
                                                 cols_to_scale = cols_to_scale, scaler = scaler, n_epochs = 35)
    print(best_dropout)


model_mlp = MyMLP(
    input_size=Xtrain_mlp.shape[1], 
    output_size=1,
    hidden_layers=[512, 256, 128],
    activation_function=nn.Sigmoid,
    dropout_rate = 0.25
).to(DEVICE)


from sklearn.base import clone
class OOFMetaFeatureGenerator():
    def __init__(self, drop_mlp_cols = [],
                 scaler = None, 
                 cols_to_scale = [], 
                 n_splits = 5):
        self.scaler = scaler
        self.cols_to_scale = cols_to_scale
        self.n_splits = n_splits
        self.models = {}
        self.oof_predictions = None
        self.fitted_scaler = None
        self.fitted_models = {}
        self.drop_mlp_cols = drop_mlp_cols
        
    def add_model(self, name, model):
        self.models[name] = model

    def create_train_features(self, X, y):
        kf = KFold(n_splits = self.n_splits, shuffle = True, random_state = 42)
        self.oof_predictions = {name: np.zeros(len(X)) for name in self.models.keys()}
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if self.scaler is not None and len(self.cols_to_scale) > 0:
                scaler = clone(self.scaler)
                X_train[self.cols_to_scale] = scaler.fit_transform(X_train[self.cols_to_scale])
                X_val[self.cols_to_scale] = scaler.transform(X_val[self.cols_to_scale])

            for name, model in self.models.items():
            
                if name == 'mlp':
                    model_instance = copy.deepcopy(model)
                    optim = torch.optim.Adam(model_instance.parameters(), lr=0.0005, weight_decay=0.001)
                    train_loader = create_loader(X_train.drop(self.drop_mlp_cols, axis = 1), y_train, train=True)
                    val_loader = create_loader(X_val.drop(self.drop_mlp_cols, axis = 1), y_val)
                    model, _ = fit_mlp(model_instance, config={
                        'train_loader': train_loader,
                        'val_loader': val_loader,
                        'criterion': CRITERION,
                        'optim': optim,
                        'device': DEVICE,
                        'n_epochs': 35
                    })
                    error, _, preds = evaluate_mlp(model, val_loader, CRITERION, DEVICE)
                else:
                    model_instance = clone(model)
                    model_instance.fit(X_train, y_train)
                    preds = model_instance.predict(X_val)
            
                self.oof_predictions[name][val_idx] = preds

    def train_models(self, X, y):
        self.fitted_models = {}
        X = X.copy()
        if self.scaler is not None and len(self.cols_to_scale) > 0:
            self.fitted_scaler = clone(self.scaler)
            X[self.cols_to_scale] = self.fitted_scaler.fit_transform(X[self.cols_to_scale])

        for name, model in self.models.items():
            if name == 'mlp':
                model_instance = copy.deepcopy(model)
                optim = torch.optim.Adam(model_instance.parameters(), lr=0.0005, weight_decay=0.001)
                X_train, X_val, y_train, y_val = train_test_split(X.drop(self.drop_mlp_cols, axis = 1), y, test_size=0.2, random_state=42)
                train_loader = create_loader(X_train, y_train, train=True)
                val_loader = create_loader(X_val, y_val)
                model, _ = fit_mlp(model_instance, config={
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'criterion': CRITERION,
                    'optim': optim,
                    'device': DEVICE,
                    'n_epochs': 35
                })
                self.fitted_models[name] = model
            else:
                model_instance = clone(model)
                model_instance.fit(X, y)
                self.fitted_models[name] = model_instance
        
    def create_test_features(self, Xtest):
        Xtest = Xtest.copy()
        self.oof_predictions = {name: np.zeros(len(Xtest)) for name in self.fitted_models.keys()}
        if self.scaler is not None and len(self.cols_to_scale) > 0:
            Xtest[self.cols_to_scale] = self.fitted_scaler.transform(Xtest[self.cols_to_scale])
        
        for name, model in self.fitted_models.items():
            if name == 'mlp':
                test_loader = create_loader(Xtest.drop(self.drop_mlp_cols, axis = 1), None)
                preds = evaluate_mlp(model, test_loader, CRITERION, DEVICE, has_labels = False)
            else:
                preds = model.predict(Xtest)
        
            self.oof_predictions[name] = preds
        
    def get_meta_features(self):
        if self.oof_predictions is None:
            raise ValueError("Call create_features() first!")
        return pd.DataFrame(self.oof_predictions)
                


meta_generator = OOFMetaFeatureGenerator(drop_mlp_cols = stats, scaler = scaler, cols_to_scale = cols_to_scale, n_splits = 4)
meta_generator.add_model('mlp', model_mlp)
meta_generator.add_model('lgbm', model_lgbm)
meta_generator.add_model('xgb', model_xgb)
meta_generator.create_train_features(Xtrain, ytrain)
meta_train = meta_generator.get_meta_features()


meta_train.head()


from sklearn.linear_model import LinearRegression
meta_model = LinearRegression()
meta_model.fit(meta_train, ytrain)


# Coefs of models
print(meta_model.coef_)

# intercept
print(meta_model.intercept_)


print(mean_squared_error(ytrain, meta_model.predict(meta_train), squared=False))


meta_generator.train_models(Xtrain, ytrain)


def pipe(Xtrain, ytrain, Xdata, meta_generator, meta_model):
    Xdata = transform_ms_to_minutes(Xdata)
    create_features(Xdata, quartile_edges, decile_edges)
    meta_generator.create_test_features(Xdata)
    meta = meta_generator.get_meta_features()
    return meta_model.predict(meta)


test_predictions = pipe(Xtrain, ytrain, Xtest, meta_generator, meta_model)
print(mean_squared_error(ytest, test_predictions))


def print_preds(y_true, y_pred):
    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    spec = gridspec.GridSpec(nrows=1, ncols=3, figure=fig)
    
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])

    # --- Scatter plot ---
    ax1.scatter(y_true, y_pred, alpha=0.6)
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    ax1.set_xlabel("True values")
    ax1.set_ylabel("Predicted values")
    ax1.set_title("True vs predicted values")
    ax1.grid()

    # --- Histogram of residuals ---
    residuals = np.array(y_true) - np.array(y_pred)
    sns.histplot(residuals, bins=30, kde=True, ax=ax2, edgecolor="black")
    ax2.set_xlabel('Error (y_true - y_pred)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of residuals')
    ax2.grid()

    # --- Boxplot of residuals ---
    ax3.boxplot(residuals, vert=True)
    ax3.set_ylabel('Error')
    ax3.set_title('BoxPlot of Errors')
    ax3.grid()

    plt.show()


print_preds(ytest, test_predictions)


print_preds(ytrain, meta_model.predict(meta_train))


ids = df_test['id']
df_test =  df_test.drop('id', axis = 1)
final_preds = pipe(Xtrain, ytrain, df_test, meta_generator, meta_model)
result = pd.DataFrame()
result['id'] = ids
result['BeatsPerMinute'] = final_preds
result.to_csv('submission.csv', index = False, sep = ',')


result.head()

