#HIDDEN CELL
# %%capture

!pip install itables
!pip install optuna-integration[keras]

import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itables import init_notebook_mode, show
init_notebook_mode(all_interactive=False,connected=True)


#HIDDEN CELL

# Sets the seed for reproducibility in numpy, random, torch CPU, and CUDA.

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # For multi-GPU setups.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #May slightly slow down training, but ensures reproducibility


# Import datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",index_col='id')


#HIDDEN CELL

# Display datasets
show(train_df)
show(test_df)


# train_df overview
train_df.info()


#HIDDEN CELL

print('='*20,'Number features descriptive stats','='*20)
display(train_df.describe(include='number'))
print('\n','='*20,'Object features descriptive stats','='*20)
display(train_df.describe(exclude='number'))


#HIDDEN CELL

# simplify col names
train_df.columns = train_df.columns.str.lower().str.replace(' ','_')
test_df.columns = test_df.columns.str.lower().str.replace(' ','_')
train_extra.columns = train_extra.columns.str.lower().str.replace(' ','_')


#HIDDEN CELL

# Get unique values for each categorical column in exploratory_df
print('OBJECT FEATURES UNIQUE VALUES \n')
for col in train_df.select_dtypes(include='object'):
    print(f"Unique values in {col}: {train_df[col].unique()}")


# Identify Target
target = 'price'


# HIDDEN CELL

plt.figure(figsize=(14, 6))

# Violinplots for Train and Train Extra side by side
sns.violinplot(data=[train_df[target].values, train_extra[target].values],
               inner='box', palette=['#ef233c', '#007bff'], orient='h', linewidth=2, cut=0)
sns.violinplot(data=[train_df[target].values, train_extra[target].values],
               inner='quartile', palette=['#ef233c', '#007bff'], orient='h', linewidth=2, cut=0)

plt.yticks([0, 1], ["Train", "Train Extra"]) #Set yticks to label each violinplot

plt.title(f'Violinplot of {target} (Train vs. Train Extra)', fontsize=11)
plt.xlabel(target, fontsize=10)
plt.ylabel('Dataset', fontsize=10) #Change label to reflect what the y axis actually represents.
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Skewness and kurtosis for both datasets
for train, train_name in [(train_df, "Train"), (train_extra, "Train Extra")]:
    print(f"Skewness ({train_name}): " + str(train[target].skew()))
    print(f"Kurtosis ({train_name}): " + str(train[target].kurt()))
    display(round(train[target].to_frame().describe()))


# HIDDEN CELL

print("Visualize Missing Data in Train, Train Extra and Test Datasets")

import missingno as msno
plt.figure(figsize=(12, 6))

# Visualize missing values as a matrix 
for df, df_name in [
    (train_df, "Train"), 
    (train_extra, "Train Extra"), 
    (test_df, "Test")]:
    
    msno.matrix(df)
    plt.title(f"Missing Data in {df_name} Dataset", fontsize = 30)
    plt.show()


# HIDDEN CELL

# Calculate feature missing value counts and proportions and make barplots
for df, df_name in [(train_df, "Train"), (train_extra, "Train Extra"), (test_df, "Test")]:
    total_values = df.shape[0]
    missing_values = df.isnull().sum()  # by feature

    # Create a DataFrame for display
    missing_df = pd.concat(
        [missing_values, round(missing_values / total_values, 2), df.dtypes],
        axis=1,
        keys=['N', '%', 'dtype']
    )
    
    # Horizontal barplot of missing value counts
    plt.figure(figsize=(12, 5))
    sns.barplot(y=missing_values.index, x=missing_values.values, orient='h', color="skyblue")
    plt.title(f"Missing Values in {df_name}",fontsize=20)
    plt.ylabel("Features")
    plt.xlabel("Number of Missing Values")
    plt.tight_layout()
    plt.grid(axis='x',alpha=0.5)
    plt.show()

    # # Horizontal barplot of missing value proportions
    # plt.figure(figsize=(10, 6))
    # sns.barplot(y=missing_values.index, x=round(missing_values / total_values, 2), orient='h')
    # plt.title(f"Missing Value Proportions in {df_name}")
    # plt.ylabel("Features")
    # plt.xlabel("Proportion of Missing Values")
    # plt.tight_layout()
    # plt.show()


#HIDDEN CELL

import warnings

# Hide deprecation warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create a copy of the DataFrame to avoid modifying the original data
exploratory_df = train_df.copy()

# Select categorical and numerical columns
categorical_cols = exploratory_df.select_dtypes(include='object').columns
numerical_cols = exploratory_df.select_dtypes(exclude='object').columns

# Handle NaN values by filling them with a placeholder
exploratory_df[numerical_cols] = exploratory_df[numerical_cols].fillna(exploratory_df[numerical_cols].median())
# exploratory_df[categorical_cols] = exploratory_df[categorical_cols].fillna(exploratory_df[categorical_cols].mode())
exploratory_df[categorical_cols] = exploratory_df[categorical_cols].fillna('Missing')

print(len(numerical_cols))
print(len(categorical_cols))


#HIDDEN CELL

from IPython.display import clear_output

# Plot categorical features with count plots
fig_cat, axes_cat = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))  # Adjust grid size based on data
axes_cat = axes_cat.flatten()

for i, column in enumerate(categorical_cols):
    clear_output(wait=False)
    print(f'Processing categorical feature: {column}')
    sns.countplot(x=exploratory_df[column], ax=axes_cat[i], order=exploratory_df[column].value_counts().index)
    axes_cat[i].set_title(column, fontsize=9)
    axes_cat[i].tick_params(axis='both', which='major', labelsize=6)

fig_cat.suptitle('Categorical Feature Distributions', fontsize=11)
plt.tight_layout()
plt.show()

# Categorical Cols insights
display(train_df[categorical_cols].describe())


#HIDDEN CELL

# Loop through each categorical feature to display summary statistics and box plot
for column in categorical_cols:
    # Calculate summary statistics grouped by the categorical column
    stats = exploratory_df.groupby(column)[target].describe()
    
    # Display summary statistics
    print(f"\nSummary Statistics for price by {column}:")
    display(stats)
    
    # Plot violin plot
    plt.figure(figsize=(10, 4))
    sns.violinplot(data=exploratory_df, x=column, y=target, palette='Spectral',inner="quartile", linewidth=2, cut=0)
    sns.violinplot(data=exploratory_df, x=column, y=target, palette='Spectral',inner="box", linewidth=2, cut=0,)
    plt.title(f'price by {column}', fontsize=12)
    plt.xlabel(column, fontsize=11)
    plt.ylabel(target, fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


from sklearn.model_selection import train_test_split

# Merge the two train set
full_train_df = pd.concat([train_df, train_extra],axis=0)

# Split into train and validation sets
X = full_train_df.copy()
y = X.pop(target)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=SEED)

# Rename test_df (optional)
X_test = test_df.copy()


from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Separate numerical and categorical columns, excluding 'occupation' from cat_cols
num_cols = X_train.select_dtypes(include=['number']).columns
cat_cols = X_train.select_dtypes(exclude=['number']).columns #.difference(['occupation']) in case you need a diff strategy for other columns

# Original column names
cols = list(num_cols) + list(cat_cols)

# Store original dtypes for later conversion
original_dtypes = X_train.dtypes[cols] # Store dtypes of selected columns

# Define the imputers
num_imputer = SimpleImputer(strategy='median')  # Use 'median', 'mean', or other strategies
cat_imputer_mode = SimpleImputer(strategy='most_frequent')  # For categorical columns
cat_imputer_missing = SimpleImputer(strategy='constant', fill_value='missing')  # Specific strategy for 'occupation'

# Create a column transformer to apply different imputers
column_transformer = make_column_transformer(
    (num_imputer, num_cols),             # Impute numerical columns
    (cat_imputer_mode, cat_cols),        # Impute other categorical columns
    # (cat_imputer_missing, ['feature']),  # Impute a column separately
    
)

# Fit and transform the training data
X_train_imputed = pd.DataFrame(
    column_transformer.fit_transform(X_train),  # Fit and transform the pipeline
    columns=cols,                     # Assign original column names
    index=X_train.index,               # Retain the original index
)

# Transform validation data
X_valid_imputed = pd.DataFrame(
    column_transformer.transform(X_valid),      # Transform only
    columns=cols,          
    index=X_valid.index               
)

# Transform test data
X_test_imputed = pd.DataFrame(
    column_transformer.transform(X_test),       # Transform only
    columns=cols,           
    index=X_test.index                
)

# Assign original dtypes to imputed DataFrames
X_train_imputed = X_train_imputed.astype(original_dtypes)
X_valid_imputed = X_valid_imputed.astype(original_dtypes)
X_test_imputed = X_test_imputed.astype(original_dtypes)


from sklearn.preprocessing import FunctionTransformer

# Create feature engineering function
def create_new_features(df):
    # Combine 'brand' and 'material'
    df['brand_material'] = df['brand'] + '_' + df['material']
    
    # Convert 'compartments' to an ordinal feature
    df['compartments_ordinal'] = pd.cut(df['compartments'], bins=[0, 3, 6, 10], labels=['few', 'moderate', 'abundant'])
    
    # Combine 'laptop_compartment' and 'waterproof'
    df['laptop_waterproof'] = df.apply(lambda row: 'both' if row['laptop_compartment'] == 'Yes' and row['waterproof'] == 'Yes' else 'One or None',axis=1)
    
    # Create an interaction term between 'Weight Capacity' and 'Compartments'
    df['weightXcompartment'] = df['weight_capacity_(kg)'] * df['compartments']

    return df

# ... and incorporate it into FunctionTransformer..
# feature_engineer = FunctionTransformer(create_new_features)
# feature_engineer.fit_transform(X_train_imputed.copy())

# or apply directly
X_train_transformed = create_new_features(X_train_imputed.copy())
X_valid_transformed = create_new_features(X_valid_imputed.copy())
X_test_transformed = create_new_features(X_test_imputed.copy())


# from sklearn.feature_selection import mutual_info_regression

# # Sample a subset of the data
# sampled_X = pd.get_dummies(X_train_transformed.copy()).sample(frac=0.25, random_state=SEED)
# sampled_y = y_train[sampled_X.index]

# mi_scores = mutual_info_regression(sampled_X, sampled_y, discrete_features='auto',random_state=SEED)
# mi_scores = pd.Series(mi_scores, name="MI Scores", index=sampled_X.columns)
# mi_scores = mi_scores.sort_values(ascending=False)
# mi_features = mi_scores[:10].keys()
# mi_features


# def plot_mi_scores(scores):
#     scores = scores.sort_values(ascending=True)
#     width = np.arange(len(scores))
#     ticks = list(scores.index)
#     plt.barh(width, scores)
#     plt.yticks(width, ticks)
#     plt.grid(alpha=0.2)
#     plt.title("Mutual Information Scores")

# plt.figure(dpi=100, figsize=(15, 10))
# plot_mi_scores(mi_scores)


def encode_ordinal(df):
    
    """Encodes ordinal categorical features into numerical values."""
    size = {"Small": 0, "Medium": 1, "Large": 2}
    df['size'] = df['size'].map(size)
    
    return df

X_train_transformed = encode_ordinal(X_train_transformed.copy())
X_valid_transformed = encode_ordinal(X_valid_transformed.copy())
X_test_transformed = encode_ordinal(X_test_transformed.copy())


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer

# Define the preprocessing for numerical...
numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
])

# and categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, X_train_transformed.select_dtypes(include='number').columns),
        ("cat", categorical_transformer, X_train_transformed.select_dtypes(exclude='number').columns),
    ],
    remainder = 'passthrough'
)

# Preprocess data (and convert to dense array)
X_train_preprocessed = pd.DataFrame(preprocessor.fit_transform(X_train_transformed.copy()).toarray())
X_valid_preprocessed = pd.DataFrame(preprocessor.transform(X_valid_transformed.copy()).toarray())
X_test_preprocessed = pd.DataFrame(preprocessor.transform(X_test_transformed.copy()).toarray())


import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_squared_error
import numpy as np


def create_model(trial):
    """
    Creates a DNN model with hyperparameters tuned by Optuna.
    """
    num_layers = trial.suggest_int('num_layers', 1, 6)
    units = trial.suggest_int('units', 32, 128, step=32)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    regularizer_type = trial.suggest_categorical("regularizer_type", ["l1", "l2", "l1_l2"])
    regularizer_strength = trial.suggest_float('regularizer_strength', 1e-6, 1e-2, log=True)

    model = Sequential()
    for _ in range(num_layers):
        if regularizer_type == "l1":
            model.add(Dense(units, kernel_regularizer=l1(regularizer_strength), activation='relu'))
        elif regularizer_type == "l2":
            model.add(Dense(units, kernel_regularizer=l2(regularizer_strength), activation='relu'))
        else:
            model.add(Dense(units, kernel_regularizer=l1_l2(l1=regularizer_strength, l2=regularizer_strength), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer

    model.compile(loss='mean_squared_error', metrics=[RootMeanSquaredError()], optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model

def objective(trial):
    """
    Objective function for Optuna optimization.
    """
    model = create_model(trial)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    pruning_callback = optuna.integration.KerasPruningCallback(trial, "val_loss")

    model.fit(X_train_preprocessed, y_train, epochs=50, batch_size=2048, verbose=0, callbacks=[early_stopping, pruning_callback], validation_data=(X_valid_preprocessed, y_valid))
    preds = model.predict(X_valid_preprocessed)
    rmse = mean_squared_error(y_valid, np.maximum(preds, 0), squared=False)
    return rmse


# # Create Optuna study with pruning
# study = optuna.study.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
# study.optimize(objective, n_trials=50) # Adjust n_trials as needed

# # Get the best trial
# best_trial = study.best_trial
# print("Best trial:")
# print(f"  Value: {best_trial.value}")
# print("  Params: ")
# for key, value in best_trial.params.items():
#     print(f"    {key}: {value}")


def create_model_from_dict(params):
    """
    Creates a DNN model with hyperparameters from a dictionary.
    """
    num_layers = params["num_layers"]
    units = params["units"]
    learning_rate = params["learning_rate"]
    dropout_rate = params["dropout_rate"]
    regularizer_type = params["regularizer_type"]
    regularizer_strength = params["regularizer_strength"]

    model = Sequential()
    for _ in range(num_layers):
        if regularizer_type == "l1":
            model.add(Dense(units, kernel_regularizer=l1(regularizer_strength), activation='relu'))
        elif regularizer_type == "l2":
            model.add(Dense(units, kernel_regularizer=l2(regularizer_strength), activation='relu'))
        else:
            model.add(Dense(units, kernel_regularizer=l1_l2(l1=regularizer_strength, l2=regularizer_strength), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer

    model.compile(loss='mean_squared_error', metrics=[RootMeanSquaredError()], optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model


dnn_params = dict({"num_layers": 1, "units": 64, "learning_rate": 0.004779618316879261, "dropout_rate": 0.2438695028689119, "regularizer_type": l1_l2, "regularizer_strength": 1.6620552627187294e-05})
dnn_params


# Retrain the best model 
best_model = create_model_from_dict(dnn_params)
# best_model = create_model(best_trial.params)
best_model.fit(X_train_preprocessed, y_train, epochs=100, batch_size=2048, validation_data=(X_valid_preprocessed, y_valid)) #train for longer on the best model.


# Make predictions 
preds = best_model.predict(X_test_preprocessed)

# Create the submission DataFrame using the correct index
submission_df = pd.DataFrame({'id': test_df.index, f'{target}': preds.flatten()})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)  # Avoid including the index in the CSV

