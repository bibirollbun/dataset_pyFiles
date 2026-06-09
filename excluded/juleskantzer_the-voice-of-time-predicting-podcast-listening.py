# --- Installing Libraries ---
!pip install ydata-profiling
!pip install Pillow


# ----- Handling data -----
import pandas as pd
import numpy as np


# ----- Graphics -----
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ----- EDA Univariate -----
from ydata_profiling import ProfileReport


# ----- Remove the warnings -----
import warnings


# Remove the warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ----- Read the dataset -----
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col="id")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col="id")


df_train.head(3).style.background_gradient(cmap='Blues').hide(axis="index")


df_test.head(3).style.background_gradient(cmap='Blues').hide(axis="index")


# --- Convert Episode Title into category ---
df_train["Episode_Title"] = df_train["Episode_Title"].astype("category")


# ----- Dataset Report with ProfileReport for the train set -----

ProfileReport(df_train, title='Train Dataset', 
              minimal = False, 
              progress_bar = False, 
              samples = None, 
              interactions = None,
              correlations = None,
              explorative = True,
              notebook = {'iframe':{'height': '600px'}},
              missing_diagrams = {'heatmap': False, 'dendrogram': True}).to_notebook_iframe()


# ----- Dataset Report with ProfileReport for the test set -----
"""
ProfileReport(df_test, title='Test Dataset', 
              minimal = False, 
              progress_bar = False, 
              samples = None, 
              interactions = None,
              correlations = None,
              explorative = True,
              notebook = {'iframe':{'height': '600px'}},
              missing_diagrams = {'heatmap': False, 'dendrogram': True}).to_notebook_iframe()
"""


### -------------------------------- Handling outlier --------------------------------


# Limit episode length to a max of 119.99
df_train['Episode_Length_minutes'] = df_train['Episode_Length_minutes'].clip(upper=119.99)
print("Max Episode Length Minutes:", df_train["Episode_Length_minutes"].max())

# Limit  host popularity percentange to a max of 100%
df_train['Host_Popularity_percentage'] = df_train['Host_Popularity_percentage'].clip(upper=100)
print("Max Host Popularity Percentage:", df_train["Host_Popularity_percentage"].max())

# Limit  guest popularity percentange to a max of 100%
df_train['Guest_Popularity_percentage'] = df_train['Guest_Popularity_percentage'].clip(upper=100)
print("Max Guest Popularity Percentage:", df_train["Guest_Popularity_percentage"].max())

# Limit number of ads to a max of 3 ads
df_train['Number_of_Ads'] = df_train['Number_of_Ads'].clip(upper=3)
print("Max Number of Ads:", df_train["Number_of_Ads"].max())


### -------------------------------- Simplifying a variable --------------------------------


# Import Library
import re

# Extract episode number from title
df_train['Episode_Number'] = df_train['Episode_Title'].apply(lambda x: int(re.search(r'\d+', x).group()))

# Create categories by episode number
df_train['Episode_Category'] = pd.cut(df_train['Episode_Number'], 
                                    bins=[0, 25, 50, 75, 100], 
                                    labels=['Start', 'Early Middle', 'Late Middle', 'End'], 
                                    right=True)

df_train.head()


# Show null values for Episode Length Minutes
df_train[
    (df_train['Podcast_Name'] == "Mystery Matters") & (df_train['Episode_Length_minutes'].isna())
].head()


# Show not null values for Episode Length Minutes
df_train[
    (df_train['Podcast_Name'] == "Mystery Matters") & (df_train['Episode_Length_minutes'].notna())
].head()


# ----- Handling missing values -----
from sklearn.impute import KNNImputer

# ----- Evaluating the quality of imputation -----
from sklearn.metrics import mean_squared_error, mean_absolute_error


### -------------------------------- 1. Test for missing values --------------------------------


### ---------------- 2. Setting up the test ----------------

# Take non-missing values
df_complete = df_train[df_train['Episode_Length_minutes'].notna()].copy()

# Take a sample of 10,000 users
df_sample = df_complete.sample(n=10_000, random_state=42).copy()

# Take only independent variables
df_sample = df_sample[["Episode_Length_minutes", "Genre", "Host_Popularity_percentage",
            "Publication_Day", "Publication_Time", "Guest_Popularity_percentage",
            "Number_of_Ads", "Episode_Sentiment", "Episode_Category"]]

# Encode qualitative variables into quantitative ones
df_sample_encoded = pd.get_dummies(data = df_sample , drop_first = True)

# Make a copy
df_nan = df_sample_encoded.copy()

# Take the index of 20% of the dataset
missing_indices = df_nan.sample(frac=0.2, random_state=42).index

# Take indexes and set them to missing values
df_nan.loc[missing_indices, 'Episode_Length_minutes'] = np.nan
### -------------------------------------------------------



### ---------------- 3. KNN Imputation ----------------

# Take the 10 nearest neighbors
imputer = KNNImputer(n_neighbors=10)

# Apply KNN
df_imputed = pd.DataFrame(imputer.fit_transform(df_nan), 
                          columns = df_nan.columns, 
                          index = df_nan.index)

# Take the values predicted by KNN and the true values
imputed_values = df_imputed.loc[missing_indices, 'Episode_Length_minutes']
original_values = df_sample_encoded.loc[missing_indices, 'Episode_Length_minutes']
### -------------------------------------------------------



### ---------------- 4. Median Imputation ----------------

# Take the median
median_value = df_train['Episode_Length_minutes'].median()

# Fill in missing values (2,000 data out of 10,000 data) with the median
median_imputed_values = df_nan['Episode_Length_minutes'].fillna(median_value, inplace=False)

# Take only the 2,000 data items for evaluation
median_imputed_values = median_imputed_values.loc[missing_indices]
### -------------------------------------------------------



### ---------------- 5. Evaluation ----------------

# Evaluate imputation with KNN
rmse_knn = np.sqrt(mean_squared_error(original_values, imputed_values))
mae_knn = mean_absolute_error(original_values, imputed_values)

# Evaluate imputation with the median
rmse_median = np.sqrt(mean_squared_error(original_values, median_imputed_values))
mae_median = mean_absolute_error(original_values, median_imputed_values)


# Display table
print("\nğŸ�¯ RÃ©sultats de l'imputation :\n")
print("| Method         |   RMSE   |   MAE   |")
print("|-----------------|----------|---------|")
print(f"| Median        |  {rmse_median:7.2f} |  {mae_median:6.2f} |")
print(f"| KNN (k=10)      |  {rmse_knn:7.2f} |  {mae_knn:6.2f} |")
### -------------------------------------------------------


# ------- Fill in missing values -------

# Fill in missing values with the median
df_train['Episode_Length_minutes'].fillna(df_train['Episode_Length_minutes'].median(), inplace = True)

# Remove missing value
df_train.dropna(subset=['Number_of_Ads'], inplace=True)


# Check the dataset
df_train.info()


def correlation_matrix_3(df, color_0, color_1, color_2, subtitle_1, subtitle_2, subtitle_3):
    
    # Correlation matrix from -1 to 1 (with 3 colors)
    
    # If we want a correlation matrix from 0 to 1, we need 2 colors
    
    # Import the library: import matplotlib.colors as mcolors
    
    # Highlight text properties
    highlight_textprops = [{"fontsize":12, "color":f'#{color_0}', "fontname": "Cover sans", "fontweight": "bold"},
                           {"fontsize":12, "color":f'#33363F', "fontname": "Cover sans"},
                           {"fontsize":12, "color":f'#{color_0}', "fontname": "Cover sans", "fontweight": "bold"},]


    # Axis labels color
    variable_name_textprops = [{"fontsize":8, "color":f'#33363F', "fontname": "Cover sans", "fontweight": "bold"}]
    
    # Correlation matrix
    correlation_matrix = df.corr(numeric_only=True)

    # Figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=500)
    

    # Remove the upper half of the matrix
    mask = np.triu(np.ones_like(df.corr(numeric_only=True), dtype=bool))

    
    color1 = mcolors.to_rgba(f'#{color_0}')  # Negative value -1
    color_intermediate = mcolors.to_rgba(f'#{color_1}')  # Intermediate color (Value 0)
    color2 = mcolors.to_rgba(f'#{color_2}')  # Positive value 1

    
    # Create a custom color palette
    n, m = 256, 1
    cmap_custom = mcolors.LinearSegmentedColormap.from_list('custom', [color1, color_intermediate, color2], N=n, gamma=m)

    # Correlation matrix heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=cmap_custom, fmt=".2f", linewidths=0.2, cbar=False,
               annot_kws={"size": 10})
    
    
    # Horizontal and vertical labels
    xy_label = dict(size=6)
    yticks, ylabels = plt.yticks()
    xticks, xlabels = plt.xticks()
    ax.set_xticklabels(xlabels, rotation=0, **xy_label, **variable_name_textprops[0])
    ax.set_yticklabels(ylabels, **xy_label, **variable_name_textprops[0])
    
    
    # Add a title to the heatmap axis
    ax.set_title('Correlation of Numerical Variables', fontsize=20, fontweight='bold', 
             fontname='Lisboa Sans OSF', color = "#33363F")
    
    
    # Title
    # Ajouter du texte Ã  l'axe avec ax.text()
    ax.text(0.40, 0.845, f"{subtitle_1} {subtitle_2} {subtitle_3}", 
        va='bottom', ha='center', fontsize=12, 
        bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.3'), 
        color='black')  # Vous pouvez ajuster les propriÃ©tÃ©s comme la couleur, la taille de police, etc.
    



%matplotlib inline
correlation_matrix_3(df_train,"243B6E", "FFFCF9", "EA7F1B", "", "", "")


# 1. Correlation analysis between numerical variables
numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                'Guest_Popularity_percentage', 'Number_of_Ads',
                'Episode_Number', 'Listening_Time_minutes']

# Random sampling
df_train_viz = df_train.sample(n=10000, random_state=42)

# 2. Interaction: host popularity, genre, and listening time
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df_train_viz, x='Host_Popularity_percentage', y='Listening_Time_minutes',
                hue='Genre', size='Guest_Popularity_percentage', sizes=(20, 200), alpha=0.7)
plt.title("Host Popularity, Genre, and Listening Time")
plt.xlabel("Host Popularity (%)")
plt.ylabel("Listening Time (minutes)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3. Impact of publication day and number of ads
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_train_viz, x='Publication_Day', y='Listening_Time_minutes', hue='Number_of_Ads')
plt.title("Listening Time by Publication Day and Number of Ads")
plt.xlabel("Publication Day")
plt.ylabel("Listening Time (minutes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Listening time / episode length ratio by sentiment and category
df_train_viz['Listening_Ratio'] = df_train_viz['Listening_Time_minutes'] / df_train_viz['Episode_Length_minutes']

plt.figure(figsize=(16, 10))
sns.boxplot(data=df_train_viz, x='Episode_Category', y='Listening_Ratio', hue='Episode_Sentiment')
plt.title("Listening Ratio by Episode Category and Sentiment")
plt.xlabel("Episode Category")
plt.ylabel("Listening Time / Episode Length Ratio")
plt.xticks(rotation=45)
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)  # Reference line at 100%
plt.tight_layout()
plt.show()

# Define the order of the days of the week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create the pivot table with specific order for Publication_Day
pivot_table = df_train_viz.pivot_table(values='Listening_Time_minutes',
                                       index='Publication_Day',
                                       columns='Episode_Category',
                                       aggfunc='mean',
                                       observed=True)

# Reindex to order the days
pivot_table = pivot_table.reindex(day_order)

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=0.5)
plt.title("Average Listening Time by Day and Category")
plt.xlabel("Category")
plt.ylabel("Publication Day")
plt.tight_layout()
plt.show()


# --- Preprocessing ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import train_test_split


# ----- Create a transform for a new variable -----

def add_episode_category_column(df):
    
    # Convert to category
    df = df.copy()
    df["Episode_Title"] = df["Episode_Title"].astype("category")
    
    # Extract episode number
    df['Episode_Number'] = df['Episode_Title'].apply(lambda x: int(re.search(r'\d+', x).group()))
    
    # Categorize episodes
    df['Episode_Category'] = pd.cut(df['Episode_Number'], 
                                    bins=[0, 25, 50, 75, 100], 
                                    labels=['Start', 'Early Middle', 'Late Middle', 'End'], 
                                    right=True)
    return df

episode_category_transformer = FunctionTransformer(add_episode_category_column, validate=False)
# --------------------------------------------------


# ----- Handling outliers -----

def outlier_features(df):
    df = df.copy()
    
    if 'Episode_Length_minutes' in df.columns:
        df['Episode_Length_minutes'] = df['Episode_Length_minutes'].clip(upper=119.99)
        
    if 'Host_Popularity_percentage' in df.columns:
        df['Host_Popularity_percentage'] = df['Host_Popularity_percentage'].clip(upper=100)
        
    if 'Guest_Popularity_percentage' in df.columns:
        df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].clip(upper=100)
        
    if 'Number_of_Ads' in df.columns:
        df['Number_of_Ads'] = df['Number_of_Ads'].clip(upper=3)
    
    return df

outlier_transformer = FunctionTransformer(outlier_features, validate=False)
# --------------------------------------------------


# ----- Create a transform for missing values -----

def fill_missing_features(df):
    df = df.copy()
    
    median_val = df['Episode_Length_minutes'].median()
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(median_val)
    
    return df

fill_missing_transformer = FunctionTransformer(fill_missing_features, validate=False)
# --------------------------------------------------


# ----- Create a transform to delete selected variables -----

def remove_columns(df):
    cols_to_drop = ['Podcast_Name', 'Episode_Title', 'Episode_Number']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    return df

remove_columns_transformer = FunctionTransformer(remove_columns, validate=False)
# --------------------------------------------------


# ----- Create an encoding transformer -----

def apply_get_dummies(X):
    return pd.get_dummies(X, drop_first=True)

encoding = FunctionTransformer(apply_get_dummies, validate=False)
# --------------------------------------------------


# ----- Create a transformer to standardize quantitative variables -----

class SelectiveStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        # Automatically select numeric columns
        self.continuous_columns = X.select_dtypes(include=['float64', 'int64']).columns
        self.scaler.fit(X[self.continuous_columns])
        return self
    
    def transform(self, X):
        # Apply scaler only to continuous columns
        X_scaled = X.copy()
        X_scaled[self.continuous_columns] = self.scaler.transform(X[self.continuous_columns])
        return X_scaled
# --------------------------------------------------


# ----- Preprocessing Pipeline -----

preprocessing_pipeline = Pipeline([
    
    ("episode_category", episode_category_transformer), # Add "Episode_Category"
    ('outlier', outlier_transformer),                   # Clean outlier
    ('fill_missing', fill_missing_transformer),         # Fill missing values
    ('remove_columns', remove_columns_transformer),     # Remove columns
    ("encoder", encoding),                              # Encoding
    ('scaler', SelectiveStandardScaler())               # Standardisation

])


# ----- Testing our pipeline -----

data = {
    "Podcast_Name" : ["Broad"]*5,
    "Episode_Title" : ["Episode 12", "Episode 95", "Episode 50", "Episode 75", "Episode 35"],
    'Episode_Length_minutes': [130, 150, 2, 3, 60],
    'Genre' : ["True Crime", "Comedy", "Education", "Technology", "Health"],
    'Host_Popularity_percentage': [30, 150, 2, 3, 600],
    "Publication_Day" : ["Thursday", "Saturday", "Tuesday", "Monday", "Monday"],
    'Publication_Time': ["Night", "Afternoon", "Evening", "Morning", "Afternoon"],
    "Guest_Popularity_percentage" : [30, 150, 2, 3, 600],
    "Number_of_Ads" : [3, 150, 2, 3, 4],
    "Episode_Sentiment" : ["Positive", "Negative", "Neutral", "Negative", "Positive"]
}

data = pd.DataFrame(data)

processed_data = preprocessing_pipeline.fit_transform(data)

print("Created data :")
print(data.head())
print("\n \n \n")
print("-"*100)
print("\nData after transformation :")
print(processed_data.head())


# --- DataFrame with non-missing Guest_Popularity_percentage ---
df_with_guest_pop = df_train[df_train['Guest_Popularity_percentage'].notna()].copy()

# --- DataFrame with missing Guest_Popularity_percentage ---
df_without_guest_pop = df_train[df_train['Guest_Popularity_percentage'].isna()].copy()
df_without_guest_pop = df_without_guest_pop.drop(columns=['Guest_Popularity_percentage'])


# --- Take a sample of 20,000 samples ---

df_with_guest_pop_sample = df_with_guest_pop.sample(n=20_000, random_state=42)

df_without_guest_pop_sample = df_without_guest_pop.sample(n=20_000, random_state=42)


# ----- Separating vector and matrix -----

X_with_gp = df_with_guest_pop_sample.drop(columns = 'Listening_Time_minutes')
y_with_gp = df_with_guest_pop_sample['Listening_Time_minutes']

X_without_gp = df_without_guest_pop_sample.drop(columns = 'Listening_Time_minutes')
y_without_gp = df_without_guest_pop_sample['Listening_Time_minutes']


# ----- Split into train and test sets for two models -----

# With Guest_Popularity_Percentange
X_with_gp_train, X_with_gp_test, y_with_gp_train, y_with_gp_test = train_test_split(
    X_with_gp, y_with_gp, test_size=0.30, random_state=42
)

# Without Guest_Popularity_Percentange
X_without_gp_train, X_without_gp_test, y_without_gp_train, y_without_gp_test = train_test_split(
    X_without_gp, y_without_gp, test_size=0.30, random_state=42
)


# --- Installing Libraries ---
!pip install xgboost
!pip install optuna


# ----- Preprocessing -----
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

# ----- Hyperparameters -----
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import optuna

# ----- Models -----
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ----- Metrics -----
from sklearn.metrics import mean_squared_error


# --------------- Train model nÂ°01 with Guest Popularity using linear regression ---------------
"""


# ----- Pipeline with Linear Regression -----

lr_pipeline_with_gp = make_pipeline(
    preprocessing_pipeline,
    LinearRegression()
)

# ----- Hyperparameters -----

param_lr_with_gp = {
    'linearregression__fit_intercept': [True, False],
}

# ----- GridSearchCV -----

grid_search_lr_with_gp = GridSearchCV(
    lr_pipeline_with_gp,
    param_lr_with_gp,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

grid_search_lr_with_gp.fit(X_with_gp_train, y_with_gp_train)

# ----- Show the best hyperparameters -----

pd.DataFrame(grid_search_lr_with_gp.cv_results_).sort_values("rank_test_score").head()
"""


# --------------- Train model nÂ°02 without Guest Popularity using linear regression ---------------
"""


# ----- Pipeline with Linear Regression -----

lr_pipeline_without_gp = make_pipeline(
    preprocessing_pipeline,
    LinearRegression()
)

# ----- Hyperparameters -----

param_lr_without_gp = {
    'linearregression__fit_intercept': [True, False],
}

# ----- GridSearchCV -----

grid_search_lr_without_gp = GridSearchCV(
    lr_pipeline_without_gp,
    param_lr_without_gp,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

grid_search_lr_without_gp.fit(X_without_gp_train, y_without_gp_train)

# ----- Show the best hyperparameters -----

pd.DataFrame(grid_search_lr_without_gp.cv_results_).sort_values("rank_test_score").head()
"""


# --------------- Train model nÂ°01 with Guest Popularity using RandomForest ---------------
"""


# ----- Pipeline with RandomForest Regressor -----

rf_pipeline_with_gp = make_pipeline(
    preprocessing_pipeline,
    RandomForestRegressor(random_state = 450)
)

# ----- Hyperparameters -----

param_rf_with_gp = {
    'randomforestregressor__n_estimators': [100, 200, 300],
    'randomforestregressor__max_depth': [2, 5, 7, 15],      
    'randomforestregressor__min_samples_split': [2, 5, 10],           
    'randomforestregressor__min_samples_leaf': [1, 4, 7],            
}

# ----- GridSearchCv -----

grid_search_rf_with_gp = GridSearchCV(rf_pipeline_with_gp, 
                                      param_rf_with_gp, 
                                      cv = 5, 
                                      scoring='neg_root_mean_squared_error', 
                                      n_jobs=-1)

grid_search_rf_with_gp.fit(X_with_gp_train, y_with_gp_train)

# ----- Show the best hyperparameters -----

pd.DataFrame(grid_search_rf_with_gp.cv_results_).sort_values("rank_test_score").head()
"""


# --------------- Train model nÂ°02 without Guest Popularity using RandomForest ---------------
"""


# ----- Pipeline with RandomForest Regressor -----

rf_pipeline_without_gp = make_pipeline(
    preprocessing_pipeline,
    RandomForestRegressor(random_state = 450)
)

# ----- Hyperparameters -----

param_rf_without_gp = {
    'randomforestregressor__n_estimators': [100, 200, 300],
    'randomforestregressor__max_depth': [2, 5, 7, 15],  
    'randomforestregressor__min_samples_split': [2, 5, 10], 
    'randomforestregressor__min_samples_leaf': [1, 4, 7], 
}

# ----- GridSearchCv -----

grid_search_rf_without_gp = GridSearchCV(rf_pipeline_without_gp, 
                                      param_rf_without_gp, 
                                      cv = 5, 
                                      scoring='neg_root_mean_squared_error', 
                                      n_jobs=-1)

grid_search_rf_without_gp.fit(X_without_gp_train, y_without_gp_train)

# ----- Show the best hyperparameters -----

pd.DataFrame(grid_search_rf_without_gp.cv_results_).sort_values("rank_test_score").head()
"""


# --------------- Train model nÂ°01 with Guest Popularity using XGBoost and Optuna ---------------
"""
def objective(trial):
    
    # Hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0)
    }

    # Pipeline
    model = make_pipeline(
        preprocessing_pipeline,
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **params
        )
    )

    # Cross-validation with RMSE
    scores = cross_val_score(
        model,
        X_with_gp_train,
        y_with_gp_train,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1
    )

    return np.mean(scores)

# Create and lunch Optuna
study = optuna.create_study(direction="maximize") 
study.optimize(objective, n_trials=100)

# Best parameters
print("Best trial:")
print(study.best_trial)
best_params_with_gp = study.best_trial.params
"""


# --------------- Train model nÂ°02 without Guest Popularity using XGBoost and Optuna ---------------
"""
def objective(trial):
    
    # Hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0)
    }

    # Pipeline
    model = make_pipeline(
        preprocessing_pipeline,
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **params
        )
    )

    # Cross-validation with RMSE
    scores = cross_val_score(
        model,
        X_without_gp_train,
        y_without_gp_train,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1
    )

    return np.mean(scores)

# Create and lunch Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Best parameters
print("Best trial:")
print(study.best_trial)
best_params_without_gp = study.best_trial.params
"""


# -------------------- Train the entire train set --------------------

# ----- Separating vector and matrix -----

X_with_gp = df_with_guest_pop.drop(columns = 'Listening_Time_minutes')
y_with_gp = df_with_guest_pop['Listening_Time_minutes']

X_without_gp = df_without_guest_pop.drop(columns = 'Listening_Time_minutes')
y_without_gp = df_without_guest_pop['Listening_Time_minutes']


# ----- Split into train and test sets for two models -----

# With Guest_Popularity_Percentange (Model nÂ°01)
X_with_gp_train, X_with_gp_test, y_with_gp_train, y_with_gp_test = train_test_split(
    X_with_gp, y_with_gp, test_size=0.30, random_state=42
)

# Without Guest_Popularity_Percentange (Model nÂ°02)
X_without_gp_train, X_without_gp_test, y_without_gp_train, y_without_gp_test = train_test_split(
    X_without_gp, y_without_gp, test_size=0.30, random_state=42
)


# ---------- Pipeline with the final model nÂ°01 (with Guest Popularity) ----------

# Best hyperparameters
final_param_xgb_with_gp = {
    'n_estimators': 425,
    'max_depth': 4,
    'learning_rate': 0.01359661519238633,
    'subsample': 0.4008805073963276,
    'colsample_bytree': 0.8325187832312463,
}

# Final pipeline
pipeline_with_gp = make_pipeline(
    preprocessing_pipeline,
    XGBRegressor(random_state = 42, **final_param_xgb_with_gp)
)


# ---------- Pipeline with the final model nÂ°02 (without Guest Popularity) ----------

# Best hyperparameters
final_param_xgb_without_gp = {
    'n_estimators': 339,
    'max_depth': 4,
    'learning_rate': 0.01530999750231462,
    'subsample': 0.40013910962061794,
    'colsample_bytree': 0.9824077721256953,
}

# Final pipeline
pipeline_without_gp = make_pipeline(
    preprocessing_pipeline,
    XGBRegressor(random_state = 450, **final_param_xgb_without_gp)
)


# ---------- Train the final models on the entire train set without cross validation ----------

# With Guest Popularity (Model nÂ°01)
pipeline_with_gp.fit(X_with_gp_train, y_with_gp_train)


# Without Guest Popularity (Model nÂ°02)
pipeline_without_gp.fit(X_without_gp_train, y_without_gp_train)


# ---------- Prediction ----------

y_pred_with_gp = pipeline_with_gp.predict(X_with_gp_test)
y_pred_without_gp = pipeline_without_gp.predict(X_without_gp_test)


# ------ Residual ------
residuals_with_gp = y_pred_with_gp - y_with_gp_test
residuals_with_gp_abs = np.abs(residuals_with_gp)

# ------ Find the interval where 95% of errors are included ------
X_95_gp = 30
percentage_gp = 3.88

while percentage_gp <= 5 :
    X_95_gp -= 0.01
    percentage_gp = np.mean(residuals_with_gp_abs >= X_95_gp) * 100
    
print(np.round(X_95_gp, 3), np.round(percentage_gp, 3))


fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi = 300)


# ------------------------------------ Scatterplot of residues ------------------------------------
axes[0].scatter(y_with_gp_test, residuals_with_gp, alpha=0.02, color = "#0070C0")
axes[0].axhline(0, color="red", linestyle="--", linewidth=2, alpha = 0.5)


# Title
highlight_textprops = {"fontsize":14, "color":'#262626', "fontname": "Roboto", "fontweight": "semibold"}
axes[0].set_title("Distribution of errors based on \nthe actual listening duration of a podcast", 
                  pad=20,
                  **highlight_textprops)

# Define x and y axis name
highlight_textprops1 = [{"fontsize":12, "color":'#262626', "fontname": "Roboto", "fontweight": "semibold"}]
axes[0].set_xlabel(f"Real podcast listening time", **highlight_textprops1[-1])
axes[0].set_ylabel(f"Errors", **highlight_textprops1[-1])
axes[0].yaxis.labelpad = 20
axes[0].xaxis.labelpad = 20

# Remove top right frame
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
    
# Add a frame at bottom left
axes[0].spines['bottom'].set_linewidth(1.3)
axes[0].spines['bottom'].set_color('#CAC9CD')
axes[0].spines['left'].set_linewidth(1.3)
axes[0].spines['left'].set_color('#CAC9CD')
    
# Add grids
axes[0].grid(axis='x', which='major', alpha=0.5, linestyle='dotted', zorder=1)
axes[0].grid(axis='y', alpha=0, zorder=2)
       
# Change the color of the bars on the x axis
axes[0].tick_params(axis='x', colors='#CAC9CD', width=1.3)
axes[0].tick_params(axis='y', colors='#CAC9CD', width=1.3)
      
# Change the color of the bar values on the x axis
for tick in axes[0].get_xticklabels():
    tick.set_color('#202020') 

# Change color of y-axis scale bar values
for tick in axes[0].get_yticklabels():
    tick.set_color('#202020') 





# ------------------------------------ KDE of residues ------------------------------------
sns.kdeplot(residuals_with_gp, ax=axes[1], fill=True, alpha = 0.3)
    
# Title
highlight_textprops = {"fontsize":14, "color":'#262626', "fontname": "Roboto", "fontweight": "semibold"}
axes[1].set_title("Error distribution", 
                  pad=20,
                  **highlight_textprops)
    
# DÃ©finir le nom de l'axe des x et y
highlight_textprops1 = [{"fontsize":12, "color":'#262626', "fontname": "Roboto", "fontweight": "semibold"}]
axes[1].set_xlabel(f"Errors", **highlight_textprops1[-1])
axes[1].set_ylabel(f"Density", **highlight_textprops1[-1])    
axes[1].xaxis.labelpad = 20
axes[1].yaxis.labelpad = 20


# Remove top right frame
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add a frame at bottom left
axes[1].spines['bottom'].set_linewidth(1.2)
axes[1].spines['bottom'].set_color('#CAC9CD')
axes[1].spines['left'].set_linewidth(1.2)
axes[1].spines['left'].set_color('#CAC9CD')

# Add grids
axes[1].grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
axes[1].grid(axis='y', alpha=0, zorder=2)

# Change the color of the bars on the x axis
axes[1].tick_params(axis='x', colors='#CAC9CD', width=1.2)
axes[1].tick_params(axis='y', colors='#CAC9CD', width=1.2)

# Change the color of the bar values on the x axis
for tick in axes[1].get_xticklabels():
    tick.set_color('#202020') 

# Change color of y-axis scale bar values
for tick in axes[1].get_yticklabels():
    tick.set_color('#202020') 

# Add a general title for all graphics
fig.suptitle("Model nÂ°01: Evaluation Performance Report (with)", fontsize=22, 
             fontweight='semibold', 
             fontname="Roboto",
             y=1.02)

plt.tight_layout()
plt.subplots_adjust(hspace=0.45) 
plt.subplots_adjust(wspace=0.25)
plt.show()


# ------ Residual ------
residuals_without_gp = y_pred_without_gp - y_without_gp_test
residuals_without_gp_abs = np.abs(residuals_without_gp)

# ------ Find the interval where 95% of errors are included ------
X_95 = 30
percentage = 3.88

while percentage <= 5 :
    X_95 -= 0.01
    percentage = np.mean(residuals_without_gp_abs >= X_95) * 100
    
print(np.round(X_95, 3), np.round(percentage, 3))


fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi = 300)


# ------------------------------------ Scatterplot of residues ------------------------------------
axes[0].scatter(y_without_gp_test, residuals_without_gp, alpha=0.07, color = "#0070C0")
axes[0].axhline(0, color="red", linestyle="--", linewidth=2, alpha = 0.5)


# Title
highlight_textprops = {"fontsize":14, "color":'#262626', "fontname": "Roboto", "fontweight": "semibold"}
axes[0].set_title("Distribution of errors based on \nthe actual listening duration of a podcast", 
                  pad=20,
                  **highlight_textprops)

# Define x and y axis name
highlight_textprops1 = [{"fontsize":12, "color":'#262626', "fontname": "Roboto", "fontweight": "semibold"}]
axes[0].set_xlabel(f"Real podcast listening time", **highlight_textprops1[-1])
axes[0].set_ylabel(f"Errors", **highlight_textprops1[-1])
axes[0].yaxis.labelpad = 20
axes[0].xaxis.labelpad = 20

# Remove top right frame
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
    
# Add a frame at bottom left
axes[0].spines['bottom'].set_linewidth(1.3)
axes[0].spines['bottom'].set_color('#CAC9CD')
axes[0].spines['left'].set_linewidth(1.3)
axes[0].spines['left'].set_color('#CAC9CD')
    
# Add grids
axes[0].grid(axis='x', which='major', alpha=0.5, linestyle='dotted', zorder=1)
axes[0].grid(axis='y', alpha=0, zorder=2)
       
# Change the color of the bars on the x axis
axes[0].tick_params(axis='x', colors='#CAC9CD', width=1.3)
axes[0].tick_params(axis='y', colors='#CAC9CD', width=1.3)
      
# Change the color of the bar values on the x axis
for tick in axes[0].get_xticklabels():
    tick.set_color('#202020') 

# Change color of y-axis scale bar values
for tick in axes[0].get_yticklabels():
    tick.set_color('#202020') 





# ------------------------------------ KDE of residues ------------------------------------
sns.kdeplot(residuals_without_gp, ax=axes[1], fill=True, alpha = 0.3)
    
# Title
highlight_textprops = {"fontsize":14, "color":'#262626', "fontname": "Roboto", "fontweight": "semibold"}
axes[1].set_title("Error distribution", 
                  pad=20,
                  **highlight_textprops)
    
# DÃ©finir le nom de l'axe des x et y
highlight_textprops1 = [{"fontsize":12, "color":'#262626', "fontname": "Roboto", "fontweight": "semibold"}]
axes[1].set_xlabel(f"Errors", **highlight_textprops1[-1])
axes[1].set_ylabel(f"Density", **highlight_textprops1[-1])    
axes[1].xaxis.labelpad = 20
axes[1].yaxis.labelpad = 20


# Remove top right frame
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add a frame at bottom left
axes[1].spines['bottom'].set_linewidth(1.2)
axes[1].spines['bottom'].set_color('#CAC9CD')
axes[1].spines['left'].set_linewidth(1.2)
axes[1].spines['left'].set_color('#CAC9CD')

# Add grids
axes[1].grid(axis='x', which='major', alpha=0.75, linestyle='dotted', zorder=1)
axes[1].grid(axis='y', alpha=0, zorder=2)

# Change the color of the bars on the x axis
axes[1].tick_params(axis='x', colors='#CAC9CD', width=1.2)
axes[1].tick_params(axis='y', colors='#CAC9CD', width=1.2)

# Change the color of the bar values on the x axis
for tick in axes[1].get_xticklabels():
    tick.set_color('#202020') 

# Change color of y-axis scale bar values
for tick in axes[1].get_yticklabels():
    tick.set_color('#202020') 

# Add a general title for all graphics
fig.suptitle("Model nÂ°02: Evaluation Performance Report (without)", fontsize=22, 
             fontweight='semibold', 
             fontname="Roboto",
             y=1.02)

plt.tight_layout()
plt.subplots_adjust(hspace=0.45) 
plt.subplots_adjust(wspace=0.25)
plt.show()


# -------------------------------- Evaluation --------------------------------


# --- With Guest Popularity ---
# RMSE
rmse_with_gp = np.sqrt(np.mean((y_pred_with_gp - y_with_gp_test)**2))
rmse_with_gp

# MAE
mae_with_gp = np.mean(np.abs(y_pred_with_gp - y_with_gp_test))
mae_with_gp

# R2
r2_with_gp = ((np.var(y_with_gp_test) - np.var(y_with_gp_test - y_pred_with_gp)) / np.var(y_with_gp_test)) * 100
r2_with_gp


# --- Without Guest Popularity ---
# RMSE
rmse_without_gp = np.sqrt(np.mean((y_pred_without_gp - y_without_gp_test)**2))
rmse_without_gp

# MAE
mae_without_gp = np.mean(np.abs(y_pred_without_gp - y_without_gp_test))
mae_without_gp

# R2
r2_without_gp = ((np.var(y_without_gp_test) - np.var(y_without_gp_test - y_pred_without_gp)) / np.var(y_without_gp_test)) * 100
r2_without_gp


# --- Table ---
results = pd.DataFrame({
    "Model": ["With Guest Popularity", "Without Guest Popularity"],
    "RMSE": [rmse_with_gp, rmse_without_gp],
    "MAE": [mae_with_gp, mae_without_gp],
    "RÂ²": [r2_with_gp, r2_without_gp]
})

results.round(2)


# ------------- Train the models over the entire dataset -------------

X = df_train.drop(columns = 'Listening_Time_minutes')
y = df_train['Listening_Time_minutes']


# ------------- ConditionalModelRouter: Dynamic Prediction Based on Feature Availability -------------

class ConditionalModelRouter(BaseEstimator, RegressorMixin):
    def __init__(self, model_with_gp, model_without_gp):
        self.model_with_gp = model_with_gp
        self.model_without_gp = model_without_gp

    def fit(self, X, y):
        
        mask = ~X["Guest_Popularity_percentage"].isna() # Return true or false for each index
        self.model_with_gp.fit(X[mask], y[mask]) # Train only for data with Guest Popularity
        self.model_without_gp.fit(X[~mask].drop(columns=["Guest_Popularity_percentage"]), y[~mask]) # Train only for data without Guest Popularity
        return self

    def predict(self, X):
        X = X.copy()
        mask = ~X["Guest_Popularity_percentage"].isna() # Return true or false for each index
        y_pred = np.empty(len(X))
        
        y_pred[mask] = self.model_with_gp.predict(X[mask]) # Predict only for data with Guest Popularity
        y_pred[~mask] = self.model_without_gp.predict(X[~mask].drop(columns=["Guest_Popularity_percentage"])) # Predict only for data without Guest Popularity
        return y_pred


# ------------- Testing the ConditionalModelRouter -------------

# Model
from sklearn.dummy import DummyRegressor

# Data
X_router_test =  pd.DataFrame({
    "Feature1": [1, 2, 3, 4],
    "Guest_Popularity_percentage": [0.8, np.nan, 0.5, np.nan]
})

y_router_test = np.array([10, 20, 100, 30])

# Model
model_with = DummyRegressor(strategy="mean")
model_without = DummyRegressor(strategy="constant", constant=42)

# Router
router_test = ConditionalModelRouter(model_with_gp=model_with, model_without_gp=model_without)
router_test.fit(X_router_test, y_router_test)

# Preds
preds = router_test.predict(X_router_test)
print("PrÃ©dictions :", preds)


# ---------- Pipeline with the final model nÂ°01 (with Guest Popularity) ----------

# Best hyperparameters
final_param_xgb_with_gp = {
    'n_estimators': 425,
    'max_depth': 4,
    'learning_rate': 0.01359661519238633,
    'subsample': 0.4008805073963276,
    'colsample_bytree': 0.8325187832312463,
}

# Final pipeline
pipeline_with_gp = make_pipeline(
    preprocessing_pipeline,
    XGBRegressor(random_state = 450, **final_param_xgb_with_gp)
)


# ---------- Pipeline with the final model nÂ°02 (without Guest Popularity) ----------

# Best hyperparameters
final_param_xgb_without_gp = {
    'n_estimators': 339,
    'max_depth': 4,
    'learning_rate': 0.01530999750231462,
    'subsample': 0.40013910962061794,
    'colsample_bytree': 0.9824077721256953,
}

# Final pipeline
pipeline_without_gp = make_pipeline(
    preprocessing_pipeline,
    XGBRegressor(random_state = 450, **final_param_xgb_without_gp)
)


# ---------- Adding both models to a router ----------

router = ConditionalModelRouter(
    model_with_gp=pipeline_with_gp,
    model_without_gp=pipeline_without_gp
)


# ---------- Train the models over the entire dataset ----------

router.fit(X, y)


# ---------- Prediction ----------

y_deployment = router.predict(df_test)


# Submission
submission = pd.DataFrame({'id': df_test.index , 'Listening_Time_minutes': y_deployment.round(3)})
submission.to_csv("podcast_submission.csv", index = False)




