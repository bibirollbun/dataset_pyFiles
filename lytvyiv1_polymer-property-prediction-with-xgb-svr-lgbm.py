!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 


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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, RepeatedKFold, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

# Warnings
import warnings
warnings.filterwarnings("ignore")

#RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

# optuna
import optuna

#Rich
from rich.console import Console
from rich.table import Table


# Load the training dataset from a CSV file
df_train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv", delimiter=',')

# Load the test dataset from a CSV file
df_test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv", delimiter=',')



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



# Load supplemental dataset containing 'TC_mean' values (related to critical temperature)
train_supplement_Tc = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv", delimiter=',')

# Rename the 'TC_mean' column to 'Tc' for consistency with the main dataset
train_supplement_Tc['Tc'] = train_supplement_Tc['TC_mean']

# Drop the original 'TC_mean' column after renaming
train_supplement_Tc = train_supplement_Tc.drop("TC_mean", axis=1)

# Load supplemental dataset containing SMILES representations
train_supplement_SMILES = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv", delimiter=',')

# Load supplemental dataset containing 'Tg' values (e.g., glass transition temperatures)
train_supplement_Tg = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv", delimiter=',')

# Load supplemental dataset containing 'FFV' values (e.g., fractional free volume)
train_supplement_FFV = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv", delimiter=',')


def calculate_group_mean(df, group_by_column, value_column):
    """
    Groups the DataFrame by a specified column and calculates the mean of another column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_by_column (str): Column to group by.
        value_column (str): Column for which the mean will be calculated.
    
    Returns:
        pd.DataFrame: A DataFrame with the group-by column and the corresponding mean values.
    """
    # Group and calculate the mean
    grouped_mean = df.groupby(group_by_column)[value_column].mean().reset_index()
    
    # Rename columns for clarity
    grouped_mean.columns = [group_by_column, value_column]
    
    return grouped_mean



def print_row_counts(dfs):
    """
    Nicely prints the number of rows for each DataFrame with column headers.
    
    :param dfs: A list of tuples in the format (dataframe_name, dataframe)
    """
    console = Console()
    table = Table(
        title="Row Count per DataFrame",
        show_header=True,            # Show column headers
        header_style="bold magenta", # Style for headers
        highlight=True,
        show_lines=True              # Show lines between rows
    )
    
    # Add columns with headers
    table.add_column("DataFrame Name", style="cyan", justify="left")
    table.add_column("Row Count", style="green", justify="right")
    
    for name, df in dfs:
        table.add_row(name, f"{df.shape[0]:,}")
    
    console.print(table)


# Create separate DataFrames for each target property, keeping only rows without missing values

# Extract 'SMILES' and 'Tg' columns, drop rows with NaN, and reset index
df_Tg = df_train[['SMILES', 'Tg']].dropna().reset_index(drop=True)

# Extract 'SMILES' and 'FFV' columns, drop rows with NaN, and reset index
df_FFV = df_train[['SMILES', 'FFV']].dropna().reset_index(drop=True)

# Extract 'SMILES' and 'Tc' columns, drop rows with NaN, and reset index
df_Tc = df_train[['SMILES', 'Tc']].dropna().reset_index(drop=True)

# Extract 'SMILES' and 'Density' columns, drop rows with NaN, and reset index
df_density = df_train[['SMILES', 'Density']].dropna().reset_index(drop=True)

# Extract 'SMILES' and 'Rg' columns, drop rows with NaN, and reset index
df_Rg = df_train[['SMILES', 'Rg']].dropna().reset_index(drop=True)

# Combine all DataFrames with their corresponding task name
dfs = [
    ('Tg_train', df_Tg),
    ('FFV_train', df_FFV),
    ('Tc_train', df_Tc),
    ('density_train', df_density),
    ('Rg_train', df_Rg)
]


print_row_counts(dfs)


df_Tg = calculate_group_mean(df_Tg, 'SMILES', 'Tg')
df_FFV = calculate_group_mean(df_FFV, 'SMILES', 'FFV')
df_Tc = calculate_group_mean(df_Tc, 'SMILES', 'Tc')
df_density = calculate_group_mean(df_density, 'SMILES', 'Density')
df_Rg = calculate_group_mean(df_Rg, 'SMILES', 'Rg')


dfs = [('Tg_train', df_Tg),
      ('FFV_train', df_FFV),
      ('Tc_train', df_Tc),
      ('density_train', df_density),
      ('Rg_train', df_Rg)]
print_row_counts(dfs)


def create_features(row):
    """Extract extended molecular descriptors from a SMILES string."""
    mol = Chem.MolFromSmiles(row['SMILES'])  # Convert SMILES to molecule

    if mol is None:
        return pd.Series({
            'MW': 0,
            'LogP': 0,
            'RotBonds': 0,
            'TPSA': 0,
            'FractionCSP3': 0,
            'RingCount': 0,
            'HeavyAtomCount': 0,
            'NumHDonors': 0,
            'NumHAcceptors': 0,
            'MolMR': 0,
            #'NumAliphaticRings': 0,
            'NumAromaticRings': 0,
            #'NumSaturatedRings': 0
        })

    return pd.Series({
        'MW': Descriptors.MolWt(mol),  # Molecular weight
        'LogP': Descriptors.MolLogP(mol),  # Octanol-water partition coefficient (lipophilicity)
        'RotBonds': Lipinski.NumRotatableBonds(mol),  # Number of rotatable bonds
        'TPSA': Descriptors.TPSA(mol),  # Topological polar surface area
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),  # Fraction of sp3-hybridized carbons
        'RingCount': Lipinski.RingCount(mol),  # Total number of rings
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),  # Number of non-hydrogen atoms
        'NumHDonors': Lipinski.NumHDonors(mol),  # Number of hydrogen bond donors (usually -OH, -NH)
        'NumHAcceptors': Lipinski.NumHAcceptors(mol),  # Number of hydrogen bond acceptors (e.g., O, N atoms)
        'MolMR': Descriptors.MolMR(mol),  # Molecular refractivity (related to volume and polarizability)
        #'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),  # Number of aliphatic (non-aromatic) rings
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),  # Number of aromatic rings
        #'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol)  # Number of saturated rings
    })



feature_df = df_Tg.apply(create_features, axis=1)
df_Tg = pd.concat([df_Tg, feature_df], axis=1)

feature_df = df_Tc.apply(create_features, axis=1)
df_Tc = pd.concat([df_Tc, feature_df], axis=1)

feature_df = df_FFV.apply(create_features, axis=1)
df_FFV = pd.concat([df_FFV, feature_df], axis=1)

feature_df = df_density.apply(create_features, axis=1)
df_density = pd.concat([df_density, feature_df], axis=1)

feature_df = df_Rg.apply(create_features, axis=1)
df_Rg = pd.concat([df_Rg, feature_df], axis=1)


df_Tg.head(5)


from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
# Parameters for fingerprint generation
NBITS = 1024

# Initialize the Morgan fingerprint generators
generator_r2 = GetMorganGenerator(radius=2, fpSize=NBITS)
#generator_r3 = GetMorganGenerator(radius=3, fpSize=NBITS)

def smiles_to_fp(smiles, generator):
    """
    Converts a SMILES string to a Morgan fingerprint using a pre-initialized RDKit generator.
    
    Parameters:
        smiles (str): SMILES representation of the molecule.
        generator: An RDKit fingerprint generator (e.g., from GetMorganGenerator).
        
    Returns:
        pd.Series: Fingerprint as a pandas Series. Returns zeros if SMILES is invalid.
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    
    # Return a zero vector if the SMILES is invalid
    if mol is None:
        return np.zeros(NBITS)
    
    # Generate the fingerprint and convert it to a NumPy array
    fp = generator.GetFingerprint(mol)
    return pd.Series(np.array(fp))


def create_df(df):
    feature_df_r2 = df['SMILES'].apply(lambda x: smiles_to_fp(x, generator_r2))
    #feature_df_r3 = df['SMILES'].apply(lambda x: smiles_to_fp(x, generator_r3))
    feature_df_r2.columns = [f'bit_{i+1}' for i in range(NBITS)]
    #feature_df_r3.columns = [f'bit_{i+NBITS+1}' for i in range(NBITS)]
    
    df = df.drop('SMILES', axis = 1)
    # Concatenate features with original DataFrame
    return pd.concat([feature_df_r2, df.reset_index(drop=True)], axis=1)

df_Tg = create_df(df_Tg)

df_FFV = create_df(df_FFV)

df_Tc = create_df(df_Tc)

df_density = create_df(df_density)

df_Rg = create_df(df_Rg)


df_FFV.head(5)


X_Tg = df_Tg.drop("Tg", axis = 1)
X_FFV = df_FFV.drop("FFV", axis = 1)
X_Tc = df_Tc.drop("Tc", axis = 1)
X_Density = df_density.drop("Density", axis = 1)
X_Rg = df_Rg.drop("Rg", axis = 1)


Y_Tg = df_Tg['Tg']
Y_FFV = df_FFV['FFV']
Y_Tc = df_Tc['Tc']
Y_Density = df_density['Density']
Y_Rg = df_Rg['Rg']


def spit_data(name, Xdata, Ydata, random_seed = 42):
    # Split into training (80%) and test (20%)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, test_size=0.2, random_state=random_seed)

    print(f"----------------split for {name}_df----------------")
    # Print shapes of the splits
    print(f"Train shape, X: {Xtrain.shape}, y: {Ytrain.shape}")
    print(f"Test shape, X: {Xtest.shape}, y: {Ytest.shape}")
    print()
    return Xtrain, Xtest, Ytrain, Ytest
    



X_Tg_Train, X_Tg_Test, Y_Tg_Train, Y_Tg_Test = spit_data("Tg" ,X_Tg, Y_Tg)

X_FFV_Train, X_FFV_Test, Y_FFV_Train, Y_FFV_Test = spit_data("FFV" ,X_FFV, Y_FFV)

X_Tc_Train, X_Tc_Test, Y_Tc_Train, Y_Tc_Test = spit_data("Tc" ,X_Tc, Y_Tc)

X_Density_Train, X_Density_Test, Y_Density_Train, Y_Density_Test = spit_data("density" ,X_Density, Y_Density)

X_Rg_Train, X_Rg_Test, Y_Rg_Train, Y_Rg_Test = spit_data("Rg" ,X_Rg, Y_Rg)


fig = plt.figure(figsize = (12,15), constrained_layout = True)
spec = gridspec.GridSpec(nrows = 3, ncols = 2, figure = fig)

dfs = [('Tg', Y_Tg_Train),
      ('Rg', Y_Rg_Train),
      ('Tc', Y_Tc_Train),
      ('Density', Y_Density_Train),
      ('FFV', Y_FFV_Train)]

i = 0
for name, df in dfs:
    if name == 'FFV':
        ax = fig.add_subplot(spec[2, :])
    else:
        ax = fig.add_subplot(spec[i // 2, i % 2])

    mean = df.mean()

    sns.histplot(df, kde = True, label = 'Histogram', bins = 14, color = "cornflowerblue")
    ax.axvline(mean, color = 'darkorchid', linestyle = "--", linewidth = 2, label = 'Mean')

    ax.text(mean, plt.ylim()[1]*(-0.1), f'{mean:.2f}', 
         ha='center', va='bottom', color='darkorchid',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_title(f"Distribution of {name}(Train)")
    ax.set_xlabel('Value')
    ax.set_ylabel('Count/Density')

    ax.grid(True)
    ax.legend()
    i+=1

plt.show()


import scipy.stats as stats
import statsmodels.api as sm

sm.qqplot(Y_FFV_Train, line='s')
plt.title("QQ-Plot for FFV")
plt.show()

# Shapiro-Wilk Test
shapiro_stat, shapiro_p = stats.shapiro(Y_FFV_Train)
print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p}")


def visualize_continious_data(continuous, Xdata_combined, Ydata_combined, Xdata_array):
    count = len(continuous)
    
    # Create figure with enough rows (two plots per feature)
    fig = plt.figure(figsize=(16, 5 * count), constrained_layout=True)

    # Define grid layout: 2 columns per feature (histogram + violin plot)
    spec = gridspec.GridSpec(nrows=count, ncols=2, figure=fig)

    # Generate a color palette for all continuous features
    palette_hist = sns.color_palette("husl", n_colors=len(continuous))
    palette_scatter = sns.color_palette("tab10", n_colors=Ydata_combined.shape[1])
    for i, feature_name in enumerate(continuous):
        # Left subplot: histogram for distribution of feature values
        hist = fig.add_subplot(spec[i, 0])
        data = Xdata_combined[feature_name]
        
        sns.histplot(data=data, label='Histogram', bins=14, color=palette_hist[i], kde = True)
        hist.set_xlabel('Value')       # X-axis label
        hist.set_ylabel("Count")       # Y-axis label
        hist.set_title(f'Distribution of {feature_name}')  # Title
        hist.legend()                  # Add legend
        hist.grid()                    # Enable grid lines

        # Right subplot: violin plot to compare feature distribution by target class
        scatter = fig.add_subplot(spec[i, 1])
        for (j, target_col), Xdata in zip(enumerate(Ydata_combined), Xdata_array):
            sns.scatterplot(
                x=Xdata[feature_name], 
                y=Ydata_combined[target_col].dropna(), 
                label=target_col, 
                color=palette_scatter[j], 
                alpha=0.6
            )
        scatter.set_xlabel(feature_name)       # Feature on X-axis
        scatter.set_ylabel("Target value")       # Target on Y-axis
        scatter.set_title(f'Scatterplot of {feature_name}')  # Title (actually it's violin plot)

    plt.show()



Ytrain_combined = pd.DataFrame()
Ytrain_combined['Tg'] = Y_Tg_Train
Ytrain_combined['FFV'] = Y_FFV_Train
Ytrain_combined['Tc'] = Y_Tc_Train
Ytrain_combined['Density'] = Y_Density_Train
Ytrain_combined['Rg'] = Y_Rg_Train

# only for visialization
scaler = MinMaxScaler()
Y_scaled = pd.DataFrame(scaler.fit_transform(Ytrain_combined), columns=Ytrain_combined.columns)


Xtrain_array = [X_Tg_Train, X_FFV_Train, X_Tc_Train, X_Density_Train, X_Rg_Train]
Xtrain_combined = pd.concat(Xtrain_array)

continuous_cols = ['MW', 'LogP', 'TPSA', 'FractionCSP3', 'MolMR', 'HeavyAtomCount']
visualize_continious_data(continuous_cols, Xtrain_combined, Y_scaled, Xtrain_array)


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

        if len(data.unique()) > 100:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        # Add a caption above the mod
        mode_index = categories.index(str(mode_val))
        mode_patch = ax.patches[mode_index]
        ax.text(mode_patch.get_x() + mode_patch.get_width() / 2, 
                mode_patch.get_height() + 10, 
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


discrete_cols = ['RotBonds', 
                 'RingCount', 
                 'NumHDonors', 
                 'NumHAcceptors',
                 #'NumAliphaticRings',
                  'NumAromaticRings', 
                 #'NumSaturatedRings'
                ]
Xtrain_combined[discrete_cols] = Xtrain_combined[discrete_cols].astype(int)
visualize_discrete_data(discrete_cols, Xtrain_combined)


target_cols = np.union1d(continuous_cols, discrete_cols)
display(Xtrain_combined[target_cols].describe())
print("-----------------------------TARGET-----------------------------")
Ytrain_combined.describe()


corr = Xtrain_combined[target_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrix of correlations of numeric features")
plt.show()


targets = ['Tg', 'Tc', 'FFV', 'Density', 'Rg']
numeric_cols = target_cols
dfs_X = [X_Tg_Train, X_Rg_Train, X_Tc_Train, X_Density_Train, X_FFV_Train]
Ys = [Y_Tg_Train, Y_Rg_Train, Y_Tc_Train, Y_Density_Train, Y_FFV_Train]

console = Console()
table = Table(
    title="Correlations of numeric features with target variables",
    show_header=True,
    header_style="bold magenta",
    highlight=True,
    show_lines=True
)
table.add_column("Feature Name", style="cyan", justify="left")
    
for col_name in targets:
    table.add_column(col_name, style="green", justify="right")

# Cycle through targets and datasets
for col_name in numeric_cols:
    correlations = [f"{dfs_X[i][col_name].corr(Ys[i]):.3f}" for i, _ in enumerate(Ys)]
    table.add_row(col_name, *correlations)

console.print(table)



def objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 5, 20),
        'reg_lambda': trial.suggest_float('reg_lambda', 5, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
        'random_state': 42,
        'tree_method': 'hist'  
    }
    
    model = xgb.XGBRegressor(**params)

    #if X.shape[0] < 1000:
    #    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    #else:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring = 'neg_mean_absolute_error')

    return -scores.mean()


import lightgbm as lgb
def objective_lgbm(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 550),
        'max_depth': trial.suggest_int('max_depth', 3, 8),  
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 40, 100),  
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # bagging_fraction
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # feature_fraction
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': 42,
        'n_jobs': -1,
        'boosting_type': 'gbdt'
    }

    model = lgb.LGBMRegressor(**params, verbose = -1)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    
    return -scores.mean()


from sklearn.model_selection import validation_curve
def SVR_fit(Xtrain, Ytrain):
    scaler = StandardScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    eps = [0.0005, 0.001, 0.01, 0.1]
    param_range = np.logspace(-4, 3, 10)
    Xdata = [('no scaler', Xtrain),
            ('standard scaler', Xtrain_scaled)]
    for name, X in Xdata:
        print(f"-------------------------------------with {name}-------------------------------------")
        fig = plt.figure(figsize=(15,13), constrained_layout = True)

        spec = gridspec.GridSpec(nrows = 2, ncols = 2, figure = fig)
        for i, epsilon in enumerate(eps):
            ax = fig.add_subplot(spec[i//2, i%2])
            model = SVR(kernel='rbf', epsilon = epsilon)
            train_scores, valid_scores = validation_curve(
                model, X, Ytrain,
                param_name="C",
                param_range=param_range,
                cv=5,  
                scoring="neg_mean_absolute_error"
            )
    
            train_scores_mean = -train_scores.mean(axis=1)
            valid_scores_mean = -valid_scores.mean(axis=1)
    
            ax.semilogx(param_range, train_scores_mean, label="Train", color="blue")
            ax.semilogx(param_range, valid_scores_mean, label="Validation", color="orange")
            ax.set_xlabel("C")
            ax.set_ylabel("MAE")
            ax.grid(True)
            ax.set_title(f"Validation curve for SVR(eps = {epsilon})")
            ax.legend()
        plt.show()



def print_learning_curve(Xdata, Ydata, model, name):
    fig = plt.figure(figsize = (8,6), constrained_layout = True)
    
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        Xdata,
        Ydata,
        cv = cv,
        scoring = 'neg_mean_absolute_error',
        train_sizes = np.linspace(0.1,1.0,10),
        n_jobs = -1
    )

    train_scores_mean = -np.mean(train_scores, axis = 1)
    val_scores_mean = - np.mean(val_scores, axis = 1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'blue', label = 'Train error')
    plt.plot(train_sizes, val_scores_mean, 'o-', color = 'orange', label = 'Validation error')
    plt.xlabel('Training size')
    plt.ylabel('MAE')
    plt.title(f'Learning curve for {name}')
    plt.legend()
    plt.grid()

    plt.show()


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


study_xgb = optuna.create_study(direction = 'minimize')
study_xgb.optimize(lambda trial : objective_xgb(trial, X_Tg_Train, Y_Tg_Train), n_trials=50)
#'exact'
print("The best paramters for XGBoost:")
print(study_xgb.best_params)


model_Tg = xgb.XGBRegressor(
    n_estimators = 242,
    max_depth = 3,
    learning_rate = 0.0303208330506349,
    subsample =0.5500874437994591,
    colsample_bytree = 0.9280163578357548,
    gamma =  3.871255234151034,
    reg_alpha =  17.513324644715365,
    reg_lambda = 11.081761976009199,
    min_child_weight = 8,
    random_state = 42
)


model_Tg.fit(X_Tg_Train, Y_Tg_Train)
print(mean_absolute_error(Y_Tg_Train, model_Tg.predict(X_Tg_Train)))
print_importances(model_Tg, X_Tg_Train)
print_learning_curve(X_Tg_Train, Y_Tg_Train, model_Tg, 'Tg')


SVR_fit(X_Tc_Train, Y_Tc_Train)


model_Tc = SVR(kernel='rbf', C = 35, epsilon = 0.01)
model_Tc.fit(X_Tc_Train, Y_Tc_Train)
print(mean_absolute_error(Y_Tc_Train, model_Tc.predict(X_Tc_Train)))


print_learning_curve(X_Tc_Train, Y_Tc_Train, model_Tc, 'TC')


study_lgbm = optuna.create_study(direction = 'minimize')
study_lgbm.optimize(lambda trial : objective_lgbm(trial, X_FFV_Train, Y_FFV_Train), n_trials=50)
#'exact'
print("The best paramters for LGBM:")
print(study_lgbm.best_params)


model_FFV = lgb.LGBMRegressor(
    n_estimators = 550,
    max_depth = 8,
    learning_rate = 0.08295741273367471,
    num_leaves = 62,
    subsample =0.7266331413804197,
    colsample_bytree =  0.6057269061426819,
    reg_alpha =0.010475975758178348,
    reg_lambda =5.4381441857520495e-05,
    min_child_samples = 9,
    random_state =  42,
    boosting_type = 'gbdt',
    verbose = -1
)

model_FFV.fit(X_FFV_Train, Y_FFV_Train)
print(mean_absolute_error(Y_FFV_Train, model_FFV.predict(X_FFV_Train)))
print_importances(model_FFV, X_FFV_Train)
print_learning_curve(X_FFV_Train, Y_FFV_Train, model_FFV, 'FFV')


study_lgb = optuna.create_study(direction = 'minimize')
study_lgb.optimize(lambda trial : objective_lgbm(trial, X_Density_Train, Y_Density_Train), n_trials=50)
#'exact'
print("The best paramters for LGBM:")
print(study_lgb.best_params)


model_Density = lgb.LGBMRegressor(
    n_estimators = 435,
    max_depth = 4,
    learning_rate = 0.04664090172676795,
    num_leaves = 63,
    subsample =0.799159121162961,
    colsample_bytree = 0.7017635466470631,
    reg_alpha = 0.000234370152853541,
    reg_lambda =1.2492545107195823,
    min_child_samples = 5,
    random_state =  42,
    boosting_type = 'gbdt'
)

model_Density.fit(X_Density_Train, Y_Density_Train)
print(mean_absolute_error(Y_Density_Train, model_Density.predict(X_Density_Train)))
print_importances(model_Density, X_Density_Train)
print_learning_curve(X_Density_Train, Y_Density_Train, model_Density, 'Density')


study_lgb = optuna.create_study(direction = 'minimize')
study_lgb.optimize(lambda trial : objective_lgbm(trial, X_Rg_Train, Y_Rg_Train), n_trials=50)
#'exact'
print("The best paramters for LGBM:")
print(study_lgb.best_params)


model_Rg = lgb.LGBMRegressor(
    n_estimators = 193,
    max_depth = 8,
    learning_rate = 0.04990693504949003,
    num_leaves = 100,
    subsample =0.5432111375969867,
    colsample_bytree = 0.5096907007873305,
    reg_alpha = 0.0002397446144844921,
    reg_lambda =0.27382854560801956,
    min_child_samples = 6,
    random_state =  42,
    boosting_type = 'gbdt'
)

model_Rg.fit(X_Rg_Train, Y_Rg_Train)
print(mean_absolute_error(Y_Rg_Train, model_Rg.predict(X_Rg_Train)))
print_importances(model_Rg, X_Rg_Train)
print_learning_curve(X_Rg_Train, Y_Rg_Train, model_Rg, 'Rg')


all_models = [model_Tg, model_FFV, model_Tc, model_Density, model_Rg]
col_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
all_X_Train = [X_Tg_Train, X_FFV_Train, X_Tc_Train, X_Density_Train, X_Rg_Train]
all_X_Test = [X_Tg_Test, X_FFV_Test, X_Tc_Test, X_Density_Test, X_Rg_Test]

all_Y_Train = [Y_Tg_Train, Y_FFV_Train, Y_Tc_Train, Y_Density_Train, Y_Rg_Train]
all_Y_Test = [Y_Tg_Test, Y_FFV_Test, Y_Tc_Test, Y_Density_Test, Y_Rg_Test]


from sklearn.model_selection import learning_curve, KFold

fig = plt.figure(figsize = (18,15), constrained_layout = True)

spec = gridspec.GridSpec(nrows = 3, ncols = 2, figure = fig)
i = 0
for name, model, X_train, Y_train in zip(col_names, all_models, all_X_Train, all_Y_Train):
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        Y_train,
        cv = cv,
        scoring = 'neg_mean_absolute_error',
        train_sizes = np.linspace(0.1,1.0,10),
        n_jobs = -1
    )

    train_scores_mean = -np.mean(train_scores, axis = 1)
    val_scores_mean = - np.mean(val_scores, axis = 1)

    ax = fig.add_subplot(spec[i//2, i%2])

    ax.plot(train_sizes, train_scores_mean, 'o-', color = 'blue', label = 'Train error')
    ax.plot(train_sizes, val_scores_mean, 'o-', color = 'orange', label = 'Validation error')
    ax.set_xlabel('Training size')
    ax.set_ylabel('MAE')
    ax.set_title(f'Learning curve for {name}')
    ax.legend()
    ax.grid()
    i+=1

plt.show()


def compute_wmae(all_models, all_X_Data, all_Y_Data):
    K = len(all_models)
    
    # Counting n_i and r_i for each property
    n_values = [len(y) for y in all_Y_Data]
    r_values = [y.max() - y.min() for y in all_Y_Data]
    
    # Calculate the denominator for the second part of the formula
    denominator = sum(np.sqrt(1 / np.array(n_values)))
    
    # Calculate weights w_i
    weights = []
    for i in range(K):
        w_i = (1 / r_values[i]) * ((K * np.sqrt(1 / n_values[i])) / denominator)
        weights.append(w_i)
    
    # Calculate wMAE
    total_error = 0
    total_count = 0
    
    for i, (model, X_data, Y_data) in enumerate(zip(all_models, all_X_Data, all_Y_Data)):
        y_pred = model.predict(X_data)
        mae_i = np.abs(y_pred - Y_data).sum()
        total_error += weights[i] * mae_i
        total_count += len(Y_data)
    
    wmae = total_error / total_count
    return wmae, weights


wmae_train, _ = compute_wmae(all_models, all_X_Train, all_Y_Train)
wmae_test, _ = compute_wmae(all_models, all_X_Test, all_Y_Test)

print(f"wMAE for Train data: {wmae_train}, wMAE for Test data: {wmae_test}")


feature_df = df_test.apply(create_features, axis = 1)
df_test = pd.concat([df_test, feature_df], axis = 1)
df_test = create_df(df_test)

test_id = df_test["id"]
df_test = df_test.drop("id", axis = 1)

submission = pd.DataFrame()
submission["id"] = test_id
for model, col in zip(all_models, col_names):
    submission[col] = model.predict(df_test)
submission    



submission.to_csv("submission.csv", index = False, sep = ',')

