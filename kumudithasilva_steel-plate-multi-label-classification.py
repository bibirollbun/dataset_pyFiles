# Data Manipulation and Analysis
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Interact with the operating system 
import os

# Warnings
import warnings

# Config parser
import configparser

# Statistical functions
from scipy import stats

# Machine Learning and Preprocessing 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight

# Model Selection and Evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# XGBoost Classifier
from xgboost import XGBClassifier

# optuna for Hyperparameter Tuning
import optuna


# Walk through '/kaggle/input' and build full file paths
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)


# Filtering warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="pyarrow")


# Determine if running in Kaggle environment

is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if is_kaggle:
    train_path = "/kaggle/input/playground-series-s4e3/train.csv"
    test_path  = "/kaggle/input/playground-series-s4e3/test.csv"
else:
    config = configparser.ConfigParser()
    config.read("config.ini")
    base_path = config.get("paths", "STEEL_DATASET_PATH")
    
    train_path = os.path.join(base_path, "train.csv")
    test_path  = os.path.join(base_path, "test.csv")

train = pd.read_csv(train_path).set_index("id")
test  = pd.read_csv(test_path).set_index("id")


# Create a fully independent copy
train_deep = train.copy(deep=True)
test_deep = test.copy(deep=True)


# All columns in the training dataset
train_deep.columns


# Generate summary statistics for all numeric columns
train_deep.describe().T


target_classes = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
targets_bin = train_deep[target_classes]

train_deep_drop_class = train_deep.drop(target_classes, axis="columns")


targets_bin.sum(axis=0)


targets_bin.sum(axis=1).value_counts()


targets_bin[targets_bin.sum(axis=1)==2]


# Copy the original multi-label targets
targets_bin_multi_lable = targets_bin.copy(deep=True)


# Identify samples with multiple defects and reassign them to K_Scatch
num_defects = targets_bin_multi_lable.sum(axis=1)
multi_defect_mask = num_defects >= 2

targets_bin_multi_lable.loc[multi_defect_mask, :] = 0
targets_bin_multi_lable.loc[multi_defect_mask, 'K_Scatch'] = 1



# Identify zero-defect rows and assign them to a new 'Zero_Defects' class
targets_bin_multi_lable['Zero_Defects'] = 0

zero_defect_mask = (targets_bin_multi_lable.sum(axis=1) == 0)
targets_bin_multi_lable.loc[zero_defect_mask, 'Zero_Defects'] = 1


# Check counts
targets_bin_multi_lable.sum(axis=0)


import matplotlib.pyplot as plt
import seaborn as sns

# --- Minimal styling for clean plots ---
plt.rcParams.update({
    'figure.facecolor': 'white',  # Set background of the figure
    'axes.facecolor': 'white',    # Set background of axes
    'grid.color': 'white'         # Hide gridlines
})

# --- Compute class percentages from one-hot multiclass DataFrame ---
total_count = targets_bin_multi_lable.shape[0]  # Total number of samples
target_columns = targets_bin_multi_lable.columns

percentage_values = [
    (targets_bin_multi_lable[col].sum() / total_count) * 100  # % of samples per class
    for col in target_columns
]

# --- Create figure with two subplots: pie and bar ---
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# --- Define a complementary color palette inspired by #0a4a56 ---
colours = ["#0a4a56", "#137c8e", "#1da9b6", "#60c2d9", 
           "#a0e1eb", "#4f7a85", "#2e5c66", "#14474f"]

# --- Pie chart on left subplot ---
explode = [0.002] * len(target_columns)  # Slightly separate slices
pie_chart = axs[0].pie(
    percentage_values, 
    autopct='%1.1f%%',                   # Show percentage on each slice
    colors=colours,                       # Custom color palette
    textprops=dict(size=9, color='white', fontweight='bold'),  # Slice text style
    radius=1.2, 
    pctdistance=0.80,                     # Distance of pct text from center
    startangle=80, 
    explode=explode,                      # Slight separation between slices
    wedgeprops=dict(width=0.5, linewidth=0.5, antialiased=True),  # Donut style
    shadow=True
)

# --- Horizontal bar chart on right subplot ---
# Extract percentage labels from pie chart for annotation
autopct_values = [item.get_text() for item in pie_chart[2]]

axs[1].barh(target_columns, percentage_values, color=colours)  # Plot horizontal bars

# Annotate bars with percentages
for i, autopct_value in enumerate(autopct_values):
    axs[1].text(float(autopct_value.strip('%')) + 1, i, autopct_value,
                va='center', ha='left', size=9, fontfamily='serif')

# --- Clean axes for minimalist look ---
sns.despine(left=True, bottom=True)
plt.yticks(fontsize=9, color='black', fontfamily='serif')
axs[1].set(xticks=[], ylabel=None, xlabel=None)

# --- Main title ---
plt.suptitle('Fault Percentage Distribution', y=1.04, fontfamily='serif', fontsize=15, fontweight='bold')

# --- Adjust layout ---
plt.subplots_adjust(wspace=0.4)
plt.tight_layout(rect=[0, 0, 0.80, 0])

# --- Display plots ---
plt.show()



# Show info for all columns

col_summary = pd.DataFrame({
    'Column': train_deep_drop_class.columns,
    'Dtype': train_deep_drop_class.dtypes,
    'Non-Null Count': train_deep_drop_class.notna().sum().values,
    'Unique Values': train_deep_drop_class.nunique().values
})

col_summary


train_deep_drop_class.Outside_Global_Index.value_counts()


train_deep_drop_class.TypeOfSteel_A400.value_counts()


train_deep_drop_class.TypeOfSteel_A300.value_counts()


def numericalplot(train_deep, test_deep, col):
    """
    Compare a numerical feature between train and test datasets.
    Left: KDE overlay.
    Right: Boxplots side by side (train vs test).
    Skewness annotated under KDE plot for clarity.
    """

    # Save original rcParams
    rc = plt.rcParams.copy()

    # Minimal style
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'grid.color': '#e0e0e0',
        'font.family': 'serif',
        'axes.titleweight': 'normal',
        'axes.labelweight': 'normal'
    })

    # Create subplots: 1 row, 3 columns with spacing
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios':[2,1,1]})

    colours = ["#0a4a56", "#137c8e"]  # train, test

    # --- Left: KDE plot ---
    sns.kdeplot(train_deep[col], ax=axs[0], color=colours[0], fill=True, alpha=0.3, linewidth=1.2, edgecolor=colours[0], label='Train')
    sns.kdeplot(test_deep[col], ax=axs[0], color=colours[1], fill=True, alpha=0.3, linewidth=1.2, edgecolor=colours[1], label='Test')
    axs[0].grid(True, linestyle='--', alpha=0.3)
    axs[0].set_facecolor('#f8f9fa')
    axs[0].set_title(f'KDE Plot: {col}', fontsize=10)
    axs[0].legend(fontsize=9)

    # --- Skewness annotation under KDE ---
    train_skew = stats.skew(train_deep[col])
    test_skew = stats.skew(test_deep[col])
    axs[0].text(0.5, -0.25, f"Train Skewness: {train_skew:.2f}    |    Test Skewness: {test_skew:.2f}",
                ha='center', va='top', fontsize=10, fontfamily='serif', transform=axs[0].transAxes)

    # --- Middle: Boxplot for train ---
    sns.boxplot(y=train_deep[col], ax=axs[1], color=colours[0], width=0.4)
    axs[1].set_title(f'Train Boxplot: {col}', fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.3)
    axs[1].set_facecolor('#f8f9fa')
    axs[1].set_xticks([])

    # --- Right: Boxplot for test ---
    sns.boxplot(y=test_deep[col], ax=axs[2], color=colours[1], width=0.4)
    axs[2].set_title(f'Test Boxplot: {col}', fontsize=10)
    axs[2].grid(True, linestyle='--', alpha=0.3)
    axs[2].set_facecolor('#f8f9fa')
    axs[2].set_xticks([])

    # --- Formatting spines and ticks ---
    for ax in axs:
        ax.yaxis.set_tick_params(labelsize=8, colors='black')
        ax.xaxis.set_tick_params(labelsize=8, colors='black')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.set_xlabel(ax.get_xlabel(), fontsize=10, fontfamily='serif')
        ax.set_ylabel(ax.get_ylabel(), fontsize=10, fontfamily='serif')


    # Main title
    plt.suptitle(f'Comparison of {col}', fontsize=14, fontfamily='serif', y=1.05)
    plt.tight_layout()

    plt.show()
    plt.rcParams.update(rc)


for feature in train_deep_drop_class.columns:
    numericalplot(train, test, feature)


class CoordinateRangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, x_min='X_Minimum', x_max='X_Maximum', y_min='Y_Minimum', y_max='Y_Maximum'):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['X_Range'] = X_[self.x_max] - X_[self.x_min]
        X_['Y_Range'] = X_[self.y_max] - X_[self.y_min]
        return X_


class SizeRatioTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, area='Pixels_Areas', x_perim='X_Perimeter', y_perim='Y_Perimeter'):
        self.area = area
        self.x_perim = x_perim
        self.y_perim = y_perim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['Area_Perimeter_Ratio'] = X_[self.area] / (X_[self.x_perim] + X_[self.y_perim])
        return X_


class AspectRatioTransformer(BaseEstimator, TransformerMixin):
    """Adds Aspect_Ratio = X_Range / Y_Range, capturing defect elongation."""
    def __init__(self, x_min='X_Minimum', x_max='X_Maximum', y_min='Y_Minimum', y_max='Y_Maximum', eps=1e-6):
        self.x_min = x_min; self.x_max = x_max; self.y_min = y_min; self.y_max = y_max
        self.eps = eps
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        X_['X_Range'] = X_[self.x_max] - X_[self.x_min]
        X_['Y_Range'] = X_[self.y_max] - X_[self.y_min]
        X_['Aspect_Ratio'] = X_['X_Range'] / (X_['Y_Range'] + self.eps)
        return X_



class VolumeTransformer(BaseEstimator, TransformerMixin):
    """Adds Volume = X_Range * Y_Range * Steel_Plate_Thickness."""
    def __init__(self, x_min='X_Minimum', x_max='X_Maximum',
                       y_min='Y_Minimum', y_max='Y_Maximum',
                       thickness='Steel_Plate_Thickness'):
        self.x_min = x_min; self.x_max = x_max; self.y_min = y_min; self.y_max = y_max
        self.thickness = thickness
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        width  = X_[self.x_max] - X_[self.x_min]
        height = X_[self.y_max] - X_[self.y_min]
        X_['Volume'] = width * height * X_[self.thickness]
        return X_


class LuminositySpreadTransformer(BaseEstimator, TransformerMixin):
    """Adds Luminosity_Spread_Ratio = (Max - Min) / Sum_of_Luminosity."""
    def __init__(self, max_col='Maximum_of_Luminosity', min_col='Minimum_of_Luminosity', sum_col='Sum_of_Luminosity'):
        self.max_col = max_col
        self.min_col = min_col
        self.sum_col = sum_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        X_['Luminosity_Spread_Ratio'] = (X_[self.max_col] - X_[self.min_col]) / X_[self.sum_col]
        return X_


class MeanLuminosityTransformer(BaseEstimator, TransformerMixin):
    """Adds Mean_Luminosity = Sum_of_Luminosity / Pixels_Areas."""
    def __init__(self, sum_col='Sum_of_Luminosity', area='Pixels_Areas', eps=1e-6):
        self.sum_col = sum_col; self.area = area; self.eps = eps
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        X_['Mean_Luminosity'] = X_[self.sum_col] / (X_[self.area] + self.eps)
        return X_


# Numeric features to scale
numeric_features = train_deep_drop_class.columns.to_list()


# ColumnTransformer for scaling numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features)
    ],
    remainder='passthrough'  # keep engineered features
)

# --- Full Pipeline ---
full_pipeline = Pipeline(steps=[
    ('coordinate_range', CoordinateRangeTransformer()),
    ('size_ratio', SizeRatioTransformer()),
    ('aspect_ratio', AspectRatioTransformer()),
    ('volume', VolumeTransformer()),
    ('luminosity_spread_ratio', LuminositySpreadTransformer()),
    ('mean_luminosity', MeanLuminosityTransformer()),
    ('scaler', preprocessor)
])

full_pipeline


# --- Prepare targets ---
y = targets_bin_multi_lable.idxmax(axis=1)

# Get unique classes and map to integers
classes = y.unique()
# get class to integer mapping
class_to_int = {cls : i for i, cls in enumerate(classes)}
# map to integers
y_int = y.map(class_to_int).to_numpy()


best_params = {'n_estimators': 776, 'learning_rate': 0.010437320605153394,
                  'max_depth': 8, 'min_child_weight': 9, 'subsample': 0.6710763610128309, 
                  'colsample_bytree': 0.5832069811739635, 'gamma': 0.006044403587111807,
                  'reg_lambda': 4.767030228369322, 'reg_alpha': 0.7267840359307755}


# %%time
#     def objective(trial):
#         """
#         Optuna will run this function many times with different hyperparameters.
#         Optimize XGBoost hyperparameters using Stratified K-Fold CV and Macro ROC-AUC.
#         """
#         # ----------------------------------------
#         # Suggest hyperparameters for XGBoost
#         # ----------------------------------------
#         param = {
#             'n_estimators': trial.suggest_int('n_estimators', 20, 1000), 
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#             'max_depth': trial.suggest_int('max_depth', 3, 12),
#             'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'gamma': trial.suggest_float('gamma', 0, 5),
#             'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
#             'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
#             'eval_metric': 'mlogloss',
#             'booster': 'gbtree',
#             'random_state': 42,

#             # Multiclass
#             'objective': 'multi:softprob',
#             'num_class': len(classes),

#             # GPU-specific
#             'tree_method': 'hist',          
#             'device': 'cuda',                
#             'verbosity': 1
#         }
        
#         # -------------------------------
#         # Stratified 5-fold CV setup
#         # -------------------------------
#         stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         roc_scores = []

#         # ----------------------------------
#         # Train + evaluate for each fold
#         # ----------------------------------
#         for train_idx, test_idx in stratified_kf.split(train_deep_drop_class, y_int):
            
#             X_train, X_test = train_deep_drop_class.iloc[train_idx], train_deep_drop_class.iloc[test_idx]
#             y_train, y_test = y_int[train_idx], y_int[test_idx]

#             # --- Compute sample weights for imbalance ---
#             classes_unique = np.unique(y_train)
#             class_weights = compute_class_weight(class_weight='balanced',classes=classes_unique,y=y_train)
#             sample_weights = np.array([class_weights[c] for c in y_train])
            

#             model = XGBClassifier(**param)
#             pipe = make_pipeline(full_pipeline, model)

#             # --- Fit on training fold ---
#             pipe.fit(X_train, y_train, **{'xgbclassifier__sample_weight': sample_weights})

#             # --- Predict probabilities ---
#             y_proba = pipe.predict_proba(X_test)

#             # Macro ROC-AUC (best for imbalanced multiclass)
#             roc = roc_auc_score(
#                 y_test,
#                 y_proba,
#                 multi_class='ovr',
#                 average='macro'
#             )
#             roc_scores.append(roc)
#         return np.mean(roc_scores)
        
#     # --------------------------------
#     # Optuna Study Setup    
#     # --------------------------------
#     pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=2)

#     # Run it the optimization
#     if __name__ == "__main__":
#         study = optuna.create_study(
#             study_name="xgb_multiclass_rocauc",
#             direction="maximize",
#             pruner=pruner,
#             storage="sqlite:///optuna_study.db",
#             load_if_exists=True 
#             )
        
#         study.optimize(
#             objective,
#             n_trials=200,
#             n_jobs=-1,
#             show_progress_bar=True
#         )
        

#         best_params = study.best_params
#         best_roc_auc = study.best_value

#         print(f"\nBest Parameters: {best_params}")
#         print(f"Best Mean ROC-AUC: {best_roc_auc:.6f}\n")


best_model = XGBClassifier(**best_params)


best_pipe = make_pipeline(full_pipeline, best_model)
best_pipe.fit(train_deep_drop_class, y_int)


def submission(model, X_test, classes, filename="submission.csv"):
    """
    Generate a submission CSV for multiclass probability predictions.
    """
    # Predict probabilities
    test_proba = model.predict_proba(X_test)
    
    # Create DataFrame with original index
    submission_df = pd.DataFrame(
        test_proba,
        columns=classes,
        index=X_test.index
    )
    
    # Drop Zero_Defects column
    if 'Zero_Defects' in submission_df.columns:
        submission_df = submission_df.drop(columns=['Zero_Defects'])
    # Save CSV
    submission_df.to_csv(filename, index=True, index_label="id")
    
    return submission_df


submission_df = submission(
    model=best_pipe,
    X_test=test_deep,
    classes=classes,
    filename="xgb_multiclass_submission.csv"
)
submission_df.head()

