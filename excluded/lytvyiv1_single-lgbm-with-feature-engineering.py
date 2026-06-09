!pip install optuna-integration[lightgbm]


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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, RepeatedKFold, learning_curve, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, f1_score, log_loss
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, StackingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
# Warnings
import warnings
warnings.filterwarnings("ignore")

# optuna
import optuna
from optuna.integration import LightGBMPruningCallback, CatBoostPruningCallback
#Rich
from rich.console import Console
from rich.table import Table

from scipy.stats import chi2_contingency


df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", delimiter = ',')

df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", delimiter = ',')


df_train.head(5)


# Drop the 'id' column from the training DataFrame
df_train = df_train.drop("id", axis=1)

# Display summary information about the DataFrame (column types, non-null counts, memory usage)
df_train.info()

# Define a function to display the percentage of missing values in each column
def show_null(df):
    null_stats = pd.DataFrame({
        '%NaN': df.isna().mean() * 100  # Calculate percentage of missing values per column
    })
    print(null_stats)

# Separator for better readability in output
print("-------------------------------------------------------")

# Show missing value statistics for the training DataFrame
show_null(df_train)



categories = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
continuous = ['age', 'balance', 'duration', 'pdays']
discrete = ['campaign', 'previous']


df_train[categories] = df_train[categories].astype('category')


def spit_data(Xdata, Ydata, random_seed = 42):
    # Split into training (80%) and test (20%)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, test_size=0.2, random_state=random_seed)

    print(f"Train shape, X: {Xtrain.shape}, y: {Ytrain.shape}")
    print(f"Test shape, X: {Xtest.shape}, y: {Ytest.shape}")

    return Xtrain, Xtest, Ytrain, Ytest
    


X = df_train.drop("y", axis = 1)
Y = df_train['y']

Xtrain, Xtest, Ytrain, Ytest = spit_data(X, Y)


def create_features(df):
    #month_map = {'dec': 'winter', 'jan': 'winter', 'feb': 'winter',
    #         'mar': 'spring', 'apr': 'spring', 'may': 'spring',
    #         'jun': 'summer', 'jul': 'summer', 'aug': 'summer',
    #         'sep': 'autumn', 'oct': 'autumn', 'nov': 'autumn'}
    #df['season'] = df['month'].map(month_map)
    
    df['log_duration'] = np.log1p(df['duration'])
    df['log_balance'] = np.sign(df['balance']) * np.log1p(np.abs(df['balance']) + 1)
    df['balance_per_age'] = df['balance'] / df['age']
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    
    #df['duration_x_campaign'] = df['duration'] * df['campaign']
    #df['balance_status'] = pd.cut(df['balance'], bins = [-1000000, -2000, -1, 1000, 5000, 1000000], labels = 
    #                             ['hight_dept', 'dept', 'low', 'medium', 'hight'])
    #df['age_group'] = pd.cut(df['age'], bins = [0, 17, 25, 40, 60, 100], labels = 
    #                         ['child', 'young', 'middle', 'senior', 'elder'])

create_features(Xtrain)
create_features(Xtest)
continuous = np.union1d(continuous, ['log_balance', 'log_balance', 'balance_per_age', 'duration_per_campaign'])


def visualize_categorical_data(categories, Xtrain):
    count = len(categories)
    
    # Create figure with enough rows for each categorical feature (two plots per row)
    fig = plt.figure(figsize=(16, 5 * count), constrained_layout=True)
    spec = gridspec.GridSpec(nrows=count, ncols=2, figure=fig)

    for i, col in enumerate(categories):
        # Left plot: bar chart of category counts
        hist = fig.add_subplot(spec[i, 0])
        data = Xtrain[col]
        
        # Get unique categories and assign colors
        cats = data.value_counts().index.tolist()
        palette = sns.color_palette("husl", n_colors=len(cats))
        color_map = dict(zip(cats, palette))
        
        # Countplot for category distribution
        sns.countplot(x=col, data=Xtrain, palette=color_map, ax=hist, order=cats)
        
        # Rotate labels if too many categories
        if len(cats) > 6:
            hist.set_xticklabels(hist.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add value labels on bars
        for label in hist.containers:
            hist.bar_label(label)
            
        hist.set_title(f"{col} Distribution")

        # Right plot: pie chart for percentage distribution
        counts = data.value_counts()
        pie = fig.add_subplot(spec[i, 1])

        wedges, texts = pie.pie(
            counts,
            startangle=90,
            colors=[color_map[c] for c in counts.index],
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        )

        # Add legend with percentage for each category
        total = counts.sum()
        legend_labels = [f"{cat} ({val / total:.1%})" for cat, val in zip(cats, counts)]
        
        # Add white circle in center to make it look like a donut chart
        centre_circle = plt.Circle((0, 0), 0.6, color='white')
        
        # Place legend outside the pie chart
        pie.legend(wedges, legend_labels, title='Categories', loc='center left', bbox_to_anchor=(1, 0.5))
        pie.add_artist(centre_circle)
        pie.set_title(f"{col} Distribution (%)")
        
    plt.show()



visualize_categorical_data(categories, Xtrain)


def visualize_continious_data(continuous, Xtrain, Y):
    count = len(continuous)

    # Create figure with enough rows (two plots per feature)
    fig = plt.figure(figsize=(16, 5 * count), constrained_layout=True)

    # Define grid layout: 2 columns per feature (histogram + violin plot)
    spec = gridspec.GridSpec(nrows=count, ncols=2, figure=fig)

    # Generate a color palette for all continuous features
    palette = sns.color_palette("husl", n_colors=len(continuous))
    
    for i, col in enumerate(continuous):
        # Left subplot: histogram for distribution of feature values
        hist = fig.add_subplot(spec[i, 0])
        data = Xtrain[col]
        
        sns.histplot(data=data, label='Histogram', bins=14, color=palette[i])
        hist.set_xlabel('Value')       # X-axis label
        hist.set_ylabel("Count")       # Y-axis label
        hist.set_title(f'Distribution of {col}')  # Title
        hist.legend()                  # Add legend
        hist.grid()                    # Enable grid lines

        # Right subplot: violin plot to compare feature distribution by target class
        box = fig.add_subplot(spec[i, 1])
        sns.violinplot(x=Y, y=data, color=palette[i], ax=box)
        box.set_xlabel('Target')       # Target on X-axis
        box.set_ylabel(f"{col}")       # Feature on Y-axis
        box.set_title(f'Boxplot of {col}')  # Title (actually it's violin plot)

    plt.show()



visualize_continious_data(continuous, Xtrain, Ytrain)


Xtrain[continuous].describe()


def visualize_discrete_data(discrete, Xtrain):
    count = len(discrete)
    
    # Create a figure with enough rows for all discrete features
    fig = plt.figure(figsize=(16, 6 * count), constrained_layout=True)
    spec = gridspec.GridSpec(nrows=count, ncols=1, figure=fig)

    for i, col in enumerate(discrete):
        # Create a subplot for each discrete feature
        ax = fig.add_subplot(spec[i, 0])
        data = Xtrain[col]
        # Calculate the mode (most frequent value) of the feature
        mode_val = data.mode()[0]  
        # Plot a countplot (bar chart) for the feature
        bars = sns.countplot(x=data, ax=ax, color='skyblue', edgecolor='black')
    
        # Get the list of categories in order on the X-axis
        categories = [t.get_text() for t in ax.get_xticklabels()]
    
        # Highlight the mode bar in a different color
        for patch, category in zip(ax.patches, categories):
            if category == str(mode_val):  # compare by name
                patch.set_facecolor('darkorchid')
            else:
                patch.set_facecolor('cornflowerblue')
    
        # Add a caption above the mod
        mode_index = categories.index(str(mode_val))
        mode_patch = ax.patches[mode_index]
        ax.text(mode_patch.get_x() + mode_patch.get_width() / 2, 
                mode_patch.get_height() + 1, 
                f'{mode_val}',
                ha='center', va='bottom', color='purple',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
    
        mode_patch_legend = mpatches.Patch(color='darkorchid', label='Mode')
        other_patch = mpatches.Patch(color='cornflowerblue', label='Other values')
        ax.legend(handles=[mode_patch_legend, other_patch])
    
        ax.grid(True)

    plt.show()


visualize_discrete_data(discrete, Xtrain)


def plot_heatmap(Xtrain, cols, name, method='pearson'):
    plt.figure(figsize = (8, 6))
    corr_matrix = Xtrain[cols].corr(method=method)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title(f"Matrix of {method} correlations of {name} features")
    plt.show()

# Function to calculate CramÃ©r's V
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

# Function for CramÃ©r's V matrix
def cramers_v_matrix(Xtrain, categorical_cols):
    n = len(categorical_cols)
    matrix = pd.DataFrame(np.zeros((n, n)), columns=categorical_cols, index=categorical_cols)
    
    for i, col1 in enumerate(categorical_cols):
        for j, col2 in enumerate(categorical_cols):
            if i == j:
                matrix.iloc[i, j] = 1.0
            elif i < j:
                v = cramers_v(Xtrain[col1], Xtrain[col2])
                matrix.iloc[i, j] = v
                matrix.iloc[j, i] = v
    return matrix


numeric = np.union1d(discrete, continuous)
plot_heatmap(Xtrain, numeric, 'numeric', 'spearman')


df = Xtrain[['previous', 'pdays']]
print(df.head(20))


df_original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", delimiter = ';')
corr = df_original['previous'].corr(df_original['pdays'], method='spearman')
print(corr)


matrix = cramers_v_matrix(Xtrain, categories)
plot_heatmap(matrix, categories, 'categorical')


Xtrain['y'] = Ytrain
Xtrain['y'] = Xtrain['y'].astype(str)
visualize_categorical_data(['y'], Xtrain)
Xtrain = Xtrain.drop('y', axis = 1)


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
            #x_col = Xtrain[col].dropna()
            #y_aligned = Ytrain.loc[x_col.index]
            correlation = Xtrain[col].corr(Ytrain, method = method_for_continuous_data)
            table.add_row(col, f"{correlation:.4f}")
    if len(categories) > 0:
        for col in categories:
            #x_col = Xtrain[col].dropna()
            #y_aligned = Ytrain.loc[x_col.index]
            correlation = cramers_v(Xtrain[col], Ytrain)
            table.add_row(col, f"{correlation:.4f}")

    console.print(table)


print_correlation_with_target(Xtrain, Ytrain, numeric, categories, 'spearman')


def objective_lgbm(trial, X, y, n_splits = 5):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1600),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-7, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 120),
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }


    cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    val_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)

        pruning_callback = LightGBMPruningCallback(trial, "auc")
        
        callbacks = [
            #lgb.early_stopping(stopping_rounds = 50, verbose = False),
            #lgb.log_evaluation(0),
            pruning_callback
        ]

        model.fit(X_train, y_train,
                 eval_set = [(X_val, y_val)],
                 eval_metric = 'auc',
                 callbacks = callbacks)
        y_pred = model.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, y_pred)
        val_scores.append(val_score)

    return np.mean(val_scores)


train = False
if train:
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lambda trial: objective_lgbm(trial, Xtrain, Ytrain, 3), n_trials = 100)
    
    print("The best parameters for LGBM:")
    print(study_lgbm.best_params)


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


best_params = {
        'n_estimators': 918,
        'boosting_type': 'gbdt',
        'max_depth': 15,
        'learning_rate': 0.03349197737315359,
        'num_leaves': 169,
        'subsample': 0.8048657579957542,
        'colsample_bytree': 0.6049772787575896,
        'reg_alpha': 4.580722717433487,
        'reg_lambda': 2.3898973933319488e-06,
        'min_child_samples': 87,
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
    }

states = [42, 75, 99, 126, 486, 222, 526 ]
models = []
for state in states:
    model = lgb.LGBMClassifier(**best_params, random_state = state)
    model.fit(Xtrain, Ytrain)
    models.append(model)
    print_importances(model, Xtrain)


def predict_probs(Xdata, models):
    count = len(models)
    all_probs = np.zeros(Xdata.shape[0])
    for model in models:
        all_probs += model.predict_proba(Xdata)[:, 1]

    return all_probs / count
    
train_probs = predict_probs(Xtrain, models)
test_probs = predict_probs(Xtest, models)


print(roc_auc_score(Ytrain, train_probs))
print(roc_auc_score(Ytest, test_probs))


print(log_loss(Ytrain, train_probs))
print(log_loss(Ytest, test_probs))


submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv', delimiter = ',')
df_test[categories] = df_test[categories].astype('category')
submission['id'] = df_test['id']
df_test = df_test.drop('id', axis = 1)

create_features(df_test)
sub_probs = predict_probs(df_test, models)
submission['y'] = sub_probs


submission.head(5)


submission.to_csv('submission.csv', index=False)

