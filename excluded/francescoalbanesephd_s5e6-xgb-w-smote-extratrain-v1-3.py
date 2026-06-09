from IPython.display import HTML, display

url = "https://wallpapers.com/images/hd/plant-aesthetic-laptop-wallpaper-feju1k06clndn3dt.jpg"

html_code = f'''
<img src="{url}" style="width:100%; height:700px; object-fit: cover; border-radius: 20px;">
'''
display(HTML(html_code))


%%capture
!pip install itables
!pip install imbalanced-learn==0.12.2


import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from itables import init_notebook_mode, show
init_notebook_mode(all_interactive=False,connected=True)

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

# Set plot style
sns.set_style('whitegrid')

# Silence FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Import datasets (on Kaggle)
TRAIN_DF = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')
TEST_DF = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv',index_col='id')
TRAIN_EXTRA = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')

# # Import datasets (on Colab)
# import os
# TRAIN_DF = pd.read_csv(os.path.join(playground_series_s5e6_path, 'train.csv'),index_col='id')
# TEST_DF = pd.read_csv(os.path.join(playground_series_s5e6_path, 'test.csv'),index_col='id')


# Extend train_df
TRAIN_DF = pd.concat([TRAIN_DF,TRAIN_EXTRA],ignore_index=True)


# For each dataset...
for dataset in [TRAIN_DF, TEST_DF]:
    # check their shape
    print(dataset.shape)
    # and fix column names
    dataset.columns = [col.lower() for col in dataset.columns]
    # replace spaces with _
    dataset.columns = dataset.columns.str.replace(' ', '_')


TRAIN_DF.describe().T.round(2)


# check skew of dtypes numeric only
print(TRAIN_DF.select_dtypes(include='number').skew())
print(TEST_DF.select_dtypes(include='number').skew())


# Display descriptive stats TRAIN_DF
print('\n',"="*50,f"TRAIN_DF description","="*50)
descriptive_stats_train = TRAIN_DF.describe().T.round(2)
descriptive_stats_train['Skew'] = TRAIN_DF.select_dtypes(include='number').skew()
descriptive_stats_train['Kurt'] = TRAIN_DF.select_dtypes(include='number').kurt()
display(descriptive_stats_train)

# Display descriptive stats of TEST_DF
print('\n',"="*50,f"TEST_DF description","="*50)
descriptive_stats_test = TEST_DF.describe().T.round(2)
descriptive_stats_test['Skew'] = TEST_DF.select_dtypes(include='number').skew()
descriptive_stats_test['Kurt'] = TEST_DF.select_dtypes(include='number').kurt()
display(descriptive_stats_test)


# Check for missing values in both sets
print(f"TRAIN_DF has {TRAIN_DF.isnull().sum().sum()} missing values")
print(f"TEST_DF has {TEST_DF.isnull().sum().sum()} missing values")


# TRAIN_DF Overview
print("="*30,"Show Training Dataset for initial data assessment","="*30)
show(TRAIN_DF)


# Identify target
target = 'fertilizer_name'

# Distribution of the target variable
plt.figure(figsize=(12, 5))
sns.countplot(data=TRAIN_DF, y=target, order = TRAIN_DF[target].value_counts().index, palette='viridis')
plt.title('Distribution of Fertilizer Types (Target Variable)')
plt.xlabel('Count')
plt.ylabel('Fertilizer Name')
plt.show()


# Relationship between numerical features and the target variable (using boxplots)

# Get the numerical features excluding the target and the index
numerical_features = TRAIN_DF.select_dtypes(include=np.number).columns.tolist()

# Set up the subplot grid
fig, axes = plt.subplots(3, 2, figsize=(18, 3 * 6))
axes = axes.flatten()

# Iterate through the numerical features and create boxplots
for i, feature in enumerate(numerical_features):
  sns.boxplot(x=target, y=feature, data=TRAIN_DF, ax=axes[i], palette='viridis')
  axes[i].set_title(f'Target vs {feature}')
  axes[i].set_xlabel('Target')
  axes[i].set_ylabel(feature)

plt.tight_layout()
plt.show()


# Analysis of all NUMERICAL features
# Define a custom color palette
custom_palette = ['#219ebc', '#c1121f']

# Function to create and display plots for a single numerical variable
def create_variable_plots(train, test, variable):

    # Merge data for visualization (without modifying original DataFrames)
    train_temp = train.copy()
    test_temp = test.copy()
    train_temp["Dataset"] = "Train"
    test_temp["Dataset"] = "Test"
    combined_data = pd.concat([train_temp, test_temp])

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    annot_kws = {'xy': (0.03, 0.75), 'xycoords': 'axes fraction', 'fontsize': 10}

    # Box plot
    sns.boxplot(data=combined_data, x=variable, y="Dataset", palette=custom_palette, ax=axes[0])
    axes[0].set_xlabel(variable)
    axes[0].set_title(f"Box Plot of {variable}")

    # Histogram
    sns.histplot(data=train, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train", ax=axes[1])
    sns.histplot(data=test, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test", ax=axes[1])
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram of {variable} [Train, Test]")
    axes[1].legend()
    axes[1].annotate(f"Skewness (TRAIN): {train[variable].skew():.2f}\nKurtosis (TRAIN): {train[variable].kurt():.2f}",
                     xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])


    # Adjust spacing and show
    plt.tight_layout()
    plt.show()


# Perform univariate analysis for each numerical variable
for variable in TRAIN_DF.select_dtypes(include='number'):
    create_variable_plots(TRAIN_DF, TEST_DF, variable)


# Define features to investigate
cat_cols = TRAIN_DF.select_dtypes(exclude='number').columns.difference([target])

# Visualise categorical variables
fig, axes = plt.subplots(1,2,figsize=(15, 5))
ax = axes.flatten()

for i, col in enumerate(TRAIN_DF[cat_cols]):
    sns.countplot(data=TRAIN_DF, y=col, order = TRAIN_DF[col].value_counts().index, palette='viridis', ax=ax[i])


# Create a heatmap to visualize the correlation matrix of the TRAIN_DF DataFrame
plt.figure(figsize=(12,8))
sns.heatmap(data=TRAIN_DF.select_dtypes(include='number').corr().round(4),
            annot=True,
            cmap='coolwarm',
            linewidth = 2
           ); plt.show(); plt.tight_layout()


# # # Pairplot
# sns.pairplot(data=TRAIN_DF.select_dtypes(include='number'),
#              palette='viridis',
#              kind='reg',             # Use 'reg' for regression plots (including scatter)
#              diag_kind='kde',        # Show KDE on the diagonal
#              plot_kws={'scatter_kws': {'alpha': 0.05, 'color': custom_palette[1]},
#                        'lowess': True}) # Enable LOWESS and set colors

# plt.suptitle('Pairplot with LOWESS Regression Lines', y=1.02)
# plt.show()


def iqr_outlier_capping(train, valid=None, test=None, columns=None):
    """
    Applies IQR-based outlier capping to specified columns of one, two, or three DataFrames.

    Parameters:
        train (pd.DataFrame): The training DataFrame used to calculate IQR thresholds.
        valid (pd.DataFrame, optional): The validation DataFrame to cap using train thresholds.
        test (pd.DataFrame, optional): The test DataFrame to cap using train thresholds.
        columns (list, optional): List of column names to apply capping to. If None, applies to all numerical columns.

    Returns:
        tuple: A tuple containing:
            - train_capped (pd.DataFrame): Capped training DataFrame.
            - valid_capped (pd.DataFrame or None): Capped validation DataFrame (if provided).
            - test_capped (pd.DataFrame or None): Capped test DataFrame (if provided).

    Note: Make sure there are no nans
    """
    train_capped = train.copy() # Avoid modifying the original DataFrame
    valid_capped = valid.copy() if valid is not None else None
    test_capped = test.copy() if test is not None else None

    if columns is None:
        columns = train.select_dtypes(include='number').columns.tolist()  # All numerical columns

    # Calculate IQR-based thresholds from the training set
    for col in columns:
        Q1 = np.percentile(train[col], 25)
        Q3 = np.percentile(train[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Show Values
        print(f'Columns {col}: \tLower Bound is: {lower_bound:.2f} \tUpper Bound is: {upper_bound:.2f}')

        # Cap outliers in the training set
        train_capped[col] = np.clip(train_capped[col], lower_bound, upper_bound)

        # If validation set is provided, cap using training set thresholds
        if valid is not None:
            valid_capped[col] = np.clip(valid[col], lower_bound, upper_bound)

        # If test set is provided, cap using training set thresholds
        if test is not None:
            test_capped[col] = np.clip(test[col], lower_bound, upper_bound)

    return train_capped, valid_capped, test_capped

# Cap target outliers in train and validation sets
# TRAIN_capped, _, TEST_capped = iqr_outlier_capping(TRAIN_DF, None, TEST_DF, columns=TRAIN_DF.select_dtypes('number').columns.difference([target]))


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Identify categorical columns excluding the target
categorical_cols = TRAIN_DF.select_dtypes(include='object').columns.tolist()
categorical_cols.remove(target)

# Feature Encoding
oe = OrdinalEncoder()
TRAIN_DF[cat_cols] = oe.fit_transform(TRAIN_DF[cat_cols])
TEST_DF[cat_cols] = oe.transform(TEST_DF[cat_cols])

# Label encode the target variable
label_encoder = LabelEncoder()
TRAIN_DF['fertilizer_name_encoded'] = label_encoder.fit_transform(TRAIN_DF[target])

print("Categorical features encoded.")
print("Encoded TRAIN_DF head:")
display(TRAIN_DF.head())
print("\nEncoded TEST_DF head:")
display(TEST_DF.head())


# Define map@3 metric function
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import time
from imblearn.over_sampling import SMOTE # Import SMOTE

# Define XGBoost model parameters
model = XGBClassifier(
    max_depth=17,
    colsample_bytree=0.4,
    subsample=0.86,
    n_estimators=5000,
    learning_rate=0.02,
    gamma=0.25,
    max_delta_step=5,
    reg_alpha=3,
    reg_lambda=1.4,
    early_stopping_rounds=100,
    objective='multi:softprob',
    min_child_weight=5,
    random_state=SEED,
    tree_method='hist',
    device='cuda',
)

# Select k folds and initialize skf
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# Define variables
encoded_target = 'fertilizer_name_encoded'
X = TRAIN_DF.select_dtypes(include='number').copy()
y = X.pop(encoded_target)
X_test = TEST_DF.select_dtypes(include='number').copy()

# Define empty variables to fill
oof_preds = np.zeros(shape = (len(TRAIN_DF) ,y.nunique()))
test_preds = np.zeros(shape = (len(TEST_DF),y.nunique()))
fold_scores = []

# Initialize SMOTE
smote = SMOTE(random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y.values)):
    print(f"\n{'#'*10} Fold {fold+1}/{FOLDS} {'#'*10}")

    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    start = time.time()

    # Apply SMOTE to the training data
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    # Fit the model with early stopping on resampled data (remove sample_weight)
    model.fit(x_train_resampled, y_train_resampled,
              eval_set=[(x_train, y_train), (x_valid, y_valid)], # Keep original validation set
              verbose=200,
    )

    # Get probabilities and Predict OOF and test
    oof_preds[valid_idx] = model.predict_proba(x_valid)
    test_preds += model.predict_proba(X_test)

    # Calculate fold score (on the original scale using expm1 and rmsle)
    top_3_preds = np.argsort(oof_preds[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    fold_score = mapk(actual, top_3_preds)
    fold_scores.append(fold_score)
    print(f" Fold {fold+1}: MAP@3 Score: {fold_score:.5f}")

    end = time.time()
    print(f"Fold {fold+1} finished in {end - start:.2f} seconds")

mean_valid_score = np.mean(fold_scores); print(f"Mean Average Precision @3 (MAP@3): {mean_valid_score:.3f}")
test_predictions = test_preds / FOLDS


top_indices = np.argsort(test_predictions, axis=1)[:, -3:][:, ::-1] # Get top 3 indices
top_fertilizers = label_encoder.inverse_transform(top_indices.ravel()).reshape(top_indices.shape)

# create the submission
submission = pd.DataFrame({'id': TEST_DF.index,
                           'Fertilizer Name': [' '.join(row) for row in top_fertilizers]})


submission.to_csv('submission.csv', index=False)

print("Submission file created. Displaying first 5 rows:")
submission.head()

