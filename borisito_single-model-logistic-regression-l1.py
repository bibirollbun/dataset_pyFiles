
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# SETUP
import pandas as pd
import numpy as np
import copy
import statsmodels.api as sm
import seaborn as sns
from matplotlib import pyplot as plt
#from pycm import ConfusionMatrix

# Transforms
from sklearn.preprocessing import PowerTransformer

# Models, evaluation, and pipeline
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    mean_squared_error,
    RocCurveDisplay
)

# Hyperparameter tuning
import optuna 

#Shap
import shap

# Set random seed for reproducibility
np.random.seed(22)


dtype_mapping = {
    'id': 'str',
    'day': 'int32',
    'pressure': 'float64',
    'maxtemp': 'float64',
    'temparature': 'float64',
    'mintemp': 'float64',
    'dewpoint': 'float64',
    'humidity': 'float64',
    'cloud': 'float64',
    'sunshine': 'float64',
    'winddirection': 'float64',
    'windspeed': 'float64',
    'rainfall': 'int32',
}

trainb = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', dtype=dtype_mapping).rename(columns={'temparature':'temp'})
testb = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', dtype=dtype_mapping).rename(columns={'temparature':'temp'})

combined_data = [trainb, testb]



import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import math
from matplotlib.ticker import MaxNLocator

df = combined_data[0]

numerical_features = [
	'pressure',
	'maxtemp',
	'temp',
	'mintemp',
	'dewpoint',
	'humidity',
	'cloud',
	'sunshine',
	'winddirection',
	'windspeed',
	'rainfall',
]

numerical_features_df = df[numerical_features]


#-----------
# descriptive stats
#-----------
# Display descriptive statistics for 'numerical_features'
num_desc = numerical_features_df.describe()
num_skew = numerical_features_df.skew()
num_kurtosis = numerical_features_df.kurtosis()

print(f'Descriptive Statistics for {numerical_features}')
print(num_desc)

#check nulls
print("Training NaN count:\n", combined_data[0].isnull().sum())
print("Test NaN count:\n", combined_data[1].isnull().sum())


import warnings
warnings.filterwarnings('ignore') # Disabling warnings for clearer outputs.
pd.options.display.max_columns = 50 # Pandas option to increase max number of columns to display.
plt.style.use('ggplot') # Setting default plot style.


def plot_numeric_features(df, numeric_features):
    """
    For each feature in numeric_features, plot:
      1) A histogram + KDE
      2) A boxplot + skew

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    numeric_features : list of str
        Column names for numeric features to plot.
    """
    n = len(numeric_features)
    if n == 0:
        print("No numeric features to plot.")
        return

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 2 * n))

    for i, feature in enumerate(numeric_features):
        # If there's only one feature, axes won't be a 2D array
        if n == 1:
            hist_ax, box_ax = axes
        else:
            hist_ax, box_ax = axes[i, 0], axes[i, 1]

        # 1) Histogram with KDE
        sns.histplot(
            data=df, x=feature, kde=True, bins=30,
            color='slategrey', ax=hist_ax
        )
        #hist_ax.set_title(f'{feature}')
        hist_ax.set_xlabel(feature)
        hist_ax.set_ylabel('Frequency')

        # 2) Boxplot
        sns.boxplot(
            data=df, x=feature,
            color='lightsteelblue', ax=box_ax
        )
        #box_ax.set_title(f'{feature}')
        box_ax.set_xlabel(feature)
        box_ax.set_ylabel('Value')

        # Calculate skew
        skewness = df[feature].skew()
        # Overlay skew on the boxplot
        box_ax.text(
            0.95, 0.95, f'Skew: {skewness:.2f}',
            horizontalalignment='right',
            verticalalignment='center',
            transform=box_ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5)
        )
    plt.tight_layout()

plot_numeric_features(numerical_features_df, ['pressure','maxtemp','temp','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed','rainfall'])


#fix corrupt (non-consecutive) days. ID previously checked for integrity
corrupt_days_count = (trainb['day'] != (trainb['id'].astype(int) % 365) + 1).sum()
print(f"Number of corrupt 'day' values: {corrupt_days_count}")



df = numerical_features_df
# sns.set(font_scale=1.1)
correlation_train = df.corr()
# Create a mask for the upper triangle of the correlation matrix to avoid redundant computations
mask = np.triu(correlation_train)
plt.figure(figsize=(18, 15))

# Plot the heatmap with detailed parameter explanations
sns.heatmap(correlation_train,
            annot=True,  # Annotate each cell with the numeric value
            fmt='.1f',  # String formatting code to use when adding annotations
            cmap='coolwarm_r',  # invert so rain is blue, and sunny is red :)
            square=True,  # Set the cells to be square-shaped
            mask=mask,  # Mask to hide the upper triangle of the matrix
            linewidths=1)  # Width of the lines that will divide each cell
plt.title('Multivariate Correlation Heatmap')


#plot just the highest correlated features
high_cross_corr_features = ['humidity', 'cloud', 'sunshine', 'rainfall']
correlation_train = df[high_cross_corr_features].corr()

#mask redundant computations
mask = np.triu(correlation_train)
plt.figure(figsize=(4, 4))

# Plot the heatmap with detailed parameter explanations
sns.heatmap(correlation_train,
            annot=True,  # Annotate each cell with the numeric value
            fmt='.1f',  # String formatting code to use when adding annotations
            cmap='coolwarm_r',  # invert colormap to so rain is blue :)
            square=True,  # Set the cells to be square-shaped
            mask=mask,  # Mask to hide the upper triangle of the matrix
            cbar= None, #remove colorbar
            linewidths=1)  # Width of the lines that will divide each cell
plt.title('Multivariate Correlation Heatmap of Highly Correlated Features')
print("")

#pairplot. have to keep redundant upper triangle as danged seaborn keeps dropping axes labels.
sns.pairplot(
    data=df,
    vars=['humidity', 'cloud', 'sunshine'],
    hue='rainfall',
    diag_kind='kde',
    palette='coolwarm_r',
    # Additional diag_kws to limit the KDE domain
    diag_kws={'clip': (0, None)}
)
plt.suptitle("Pairplot of Numeric Features by Rainfall", y=1.02)



from scipy import stats

def regplot_with_r2(x, y, **kwargs):
    """
    Scatter + red regression line, plus an annotation of R^2.
    """
    ax = plt.gca()
    sns.regplot(
        x=x, y=y, ax=ax,
        line_kws={"color": "red"},
        scatter_kws={"alpha": 0.6, "color": "gray"},
        **kwargs
    )
    r, _ = stats.pearsonr(x, y)
    r2 = r**2
    ax.text(
        0.05, 0.95, f"R² = {r2:.2f}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9
    )

high_cross_corr_features = ['humidity', 'cloud', 'sunshine', 'rainfall']

g = sns.PairGrid(df[high_cross_corr_features], corner=True)
g.map_lower(regplot_with_r2)
g.map_diag(sns.histplot, kde=True, color="gray")

# Manually set titles/labels on the diagonal subplots
for ax, col in zip(np.diag(g.axes), high_cross_corr_features):
    ax.set_title(col, fontsize=10)      # Title above the plot
    ax.set_xlabel(col)                  # X-axis label
    ax.set_ylabel("Density")            # Y-axis label
    ax.set_ylim(bottom=0)               # Start y-axis at 0 for better comparison

g.figure.suptitle("Mutual Regression Plot of Selected Features", y=1.02)



fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

# First scatter plot: x='cloud', y='humidity'
sns.scatterplot(
    data=combined_data[0],
    x='cloud',
    y='humidity',
    hue='rainfall',
    palette='coolwarm_r',
    alpha=0.7,
    ax=axes[0]
)
axes[0].set_title("Cloud vs. Humidity")

# Second scatter plot: x='dewpoint', y='humidity'
sns.scatterplot(
    data=combined_data[0],
    x='dewpoint',
    y='humidity',
    hue='rainfall',
    palette='coolwarm_r',
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title("Dewpoint vs. Humidity")

# Third scatter plot: x='sunshine', y='humidity'
sns.scatterplot(
    data=combined_data[0],
    x='sunshine',
    y='humidity',
    hue='rainfall',
    palette='coolwarm_r',
    alpha=0.7,
    ax=axes[2]
)
axes[2].set_title("Sunshine vs. Humidity")

# Reverse the x-axis for the sunshine subplot to show "darker leads to more rainfall"
axes[2].invert_xaxis()

plt.tight_layout()


#-------------------------
# Feature Engineering and Preprocessing
#-------------------------

#fix corrupt (non-consecutive) days in train (combined_data[0]. ID checked for sort and data integrity

combined_data[0]['day'] = (combined_data[0]['id'].astype(int) % 365) + 1

#  create week and month dummies
def create_day_week_mapping():
    mapping = {}
    for day in range(1, 366):  # Days 1 through 365
        week_num = (day - 1) // 7 + 1
        # Adjust if the computed week number is 53, force it to 52
        if week_num == 53:
            week_num = 52
        mapping[day] = week_num
    return mapping

def create_day_month_mapping():
    mapping = {}
    # Define the days in each month for a non-leap year
    months_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Create a list of cumulative day boundaries for each month
    cumulative = 0
    month_boundaries = []
    for days in months_days:
        cumulative += days
        month_boundaries.append(cumulative)
    
    # Map each day to its corresponding month
    for day in range(1, 366):
        for month, boundary in enumerate(month_boundaries, start=1):
            if day <= boundary:
                mapping[day] = month
                break
    
    return mapping

day_week_map = create_day_week_mapping()
day_month_map = create_day_month_mapping()

for dataset in combined_data:
    dataset['week'] = dataset['day'].map(day_week_map).astype('category')
    dataset['month'] = dataset['day'].map(day_month_map).astype('category')

#winddirection may have a null
for dataset in combined_data:
    median_winddirection = dataset['winddirection'].median()
    dataset['winddirection'] = dataset['winddirection'].fillna(median_winddirection)

# Yeo-Johnson transform on 'cloud'. 
pt = PowerTransformer(method='yeo-johnson')
trainb[['cloud']] = pt.fit_transform(trainb[['cloud']])
testb[['cloud']] = pt.transform(testb[['cloud']])

# interaction features
for dataset in combined_data:
    dataset['humidity_x_cloud'] = dataset['humidity'] * dataset['cloud']
    dataset['humidity_x_sunshine'] = dataset['humidity'] * dataset['sunshine']
    dataset['cloud_x_sunshine'] = dataset['cloud'] * dataset['sunshine']


def engineer_features(df):
    """
    Create new features based on meteorological understanding and data analysis,
    with 'day' representing day of the year (1-365).
    Ensures no data leakage by avoiding use of the target variable (rainfall).
    """
    # Make a copy to avoid modifying the original dataframe
    enhanced_df = copy.deepcopy(df).reset_index(drop=True)
    
    # 1. Temperature range (difference between max and min temperatures)
    enhanced_df['temp_range'] = enhanced_df['maxtemp'] - enhanced_df['mintemp']
    
    # 2. Dew point depression (difference between temperature and dew point)
    enhanced_df['dewpoint_depression'] = enhanced_df['temp'] - enhanced_df['dewpoint']
    
    # 3. Pressure change from previous day
    enhanced_df['pressure_change'] = enhanced_df['pressure'].diff().fillna(0)
    
    # 4. Humidity to dew point ratio
    enhanced_df['humidity_dewpoint_ratio'] = enhanced_df['humidity'] / enhanced_df['dewpoint'].clip(lower=0.1)
    
    # 5. Cloud coverage to sunshine ratio (inverse relationship)
    enhanced_df['cloud_sunshine_ratio'] = enhanced_df['cloud'] / enhanced_df['sunshine'].clip(lower=0.1)
    
    # 6. Wind intensity factor (combination of speed and humidity)
    enhanced_df['wind_humidity_factor'] = enhanced_df['windspeed'] * (enhanced_df['humidity'] / 100)
    
    # 7. Temperature-humidity index (simple version of heat index)
    enhanced_df['temp_humidity_index'] = (0.8 * enhanced_df['temp']) + \
                                        ((enhanced_df['humidity'] / 100) * \
                                        (enhanced_df['temp'] - 14.3)) + 46.4
    
    # 8. Pressure change rate (acceleration)
    enhanced_df['pressure_acceleration'] = enhanced_df['pressure_change'].diff().fillna(0)
    
    # 9. Seasonal features (based on day of year)
    #Convert day to month (1-365 to 1-12)
    enhanced_df['month'] = ((enhanced_df['day'] - 1) // 30) + 1
    enhanced_df['month'] = enhanced_df['month'].clip(upper=12)  # Ensure month doesn't exceed 12
    
    # 10. Convert day to season (1-365 to 1-4)
    enhanced_df['season'] = ((enhanced_df['month'] - 1) // 3) + 1
    
    # 11. Sine and cosine transformations to capture cyclical nature of days in a year
    enhanced_df['day_of_year_sin'] = np.sin(2 * np.pi * enhanced_df['day'] / 365)
    enhanced_df['day_of_year_cos'] = np.cos(2 * np.pi * enhanced_df['day'] / 365)
    
    # 12. Rolling averages for key meteorological variables
    for window in [3, 7, 14]:
        enhanced_df[f'temperature_rolling_{window}d'] = enhanced_df['temp'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'pressure_rolling_{window}d'] = enhanced_df['pressure'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'humidity_rolling_{window}d'] = enhanced_df['humidity'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'cloud_rolling_{window}d'] = enhanced_df['cloud'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'windspeed_rolling_{window}d'] = enhanced_df['windspeed'].rolling(window=window, min_periods=1).mean()
    
    # 13. Weather pattern change features
    # Temperature trend
    enhanced_df['temp_trend_3d'] = enhanced_df['temp'].diff(3).fillna(0)
    # Pressure trend
    enhanced_df['pressure_trend_3d'] = enhanced_df['pressure'].diff(3).fillna(0)
    # Humidity trend
    enhanced_df['humidity_trend_3d'] = enhanced_df['humidity'].diff(3).fillna(0)
    
    # 14. Extreme weather indicators
    enhanced_df['extreme_temp'] = (enhanced_df['temp'] > enhanced_df['temp'].quantile(0.95)) | \
                                 (enhanced_df['temp'] < enhanced_df['temp'].quantile(0.05))
    enhanced_df['extreme_temp'] = enhanced_df['extreme_temp'].astype(int)
    
    enhanced_df['extreme_humidity'] = (enhanced_df['humidity'] > enhanced_df['humidity'].quantile(0.95)) | \
                                     (enhanced_df['humidity'] < enhanced_df['humidity'].quantile(0.05))
    enhanced_df['extreme_humidity'] = enhanced_df['extreme_humidity'].astype(int)
    
    enhanced_df['extreme_pressure'] = (enhanced_df['pressure'] > enhanced_df['pressure'].quantile(0.95)) | \
                                     (enhanced_df['pressure'] < enhanced_df['pressure'].quantile(0.05))
    enhanced_df['extreme_pressure'] = enhanced_df['extreme_pressure'].astype(int)
    
    # 15. Interaction terms between key variables
    enhanced_df['temp_humidity_interaction'] = enhanced_df['temp'] * enhanced_df['humidity']
    enhanced_df['pressure_wind_interaction'] = enhanced_df['pressure'] * enhanced_df['windspeed']
    #enhanced_df['cloud_sunshine_interaction'] = enhanced_df['cloud'] * enhanced_df['sunshine']
    enhanced_df['humidity_x_dewpoint'] = enhanced_df['dewpoint'] * enhanced_df['humidity']
    
    # 16. Moving standard deviations for measuring variability
    for window in [7, 14]:
        enhanced_df[f'temp_std_{window}d'] = enhanced_df['temp'].rolling(window=window, min_periods=4).std().fillna(0)
        enhanced_df[f'pressure_std_{window}d'] = enhanced_df['pressure'].rolling(window=window, min_periods=4).std().fillna(0)
        enhanced_df[f'humidity_std_{window}d'] = enhanced_df['humidity'].rolling(window=window, min_periods=4).std().fillna(0)
    
    return enhanced_df
    

for i, dataset in enumerate(combined_data):
    combined_data[i] = engineer_features(dataset)


#-------------------------
# One-Hot Encoding and Cleanup
#-------------------------

encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoder.fit(combined_data[0][['week', 'month', 'season', 'winddirection']])

for i, df_ in enumerate(combined_data):
    categorical_cols = ['week', 'month', 'season', 'winddirection']
    X_cat = encoder.transform(df_[categorical_cols])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols), index=df_.index)
    df_.drop(columns=categorical_cols, inplace=True)
    combined_data[i] = pd.concat([df_, X_cat_df], axis=1)




#-------------------------
# Build Final Feature Set and Target
#-------------------------

# Drop columns not needed as features
X_full = combined_data[0].drop(['id', 'rainfall','day'], axis=1)
y_full = combined_data[0]['rainfall']

# Train/Validation Split on first four years for train and last two years as test.

combined_data[0]['id'] = combined_data[0]['id'].astype(int) 

X_train = combined_data[0].drop(['id', 'rainfall','day'], axis=1)
X_val   = combined_data[0].drop(['id', 'rainfall','day'], axis=1)

y_train  = combined_data[0]['rainfall']
y_val = combined_data[0]['rainfall']


# Create a TimeSeriesSplit. Make sure 'id' remains sorted. I did this manually at first, but the library makes multiple splits easier
tscv = TimeSeriesSplit(n_splits=15)



#-------------------------
# Logistic Regression with Regularization (L1/L2)
#-------------------------

def create_pipeline_with_pca(C, pca_n_components, penalty, class_weight):
    """
    Pipeline: StandardScaler -> PCA -> LogisticRegression
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_n_components)),
        ('clf', LogisticRegression(
            penalty=penalty,
            solver='liblinear',
            random_state=23,
            max_iter=1000,
            C=C,
            class_weight = class_weight
        ))
    ])

def create_pipeline_no_pca(C, penalty, class_weight):
    """
    Pipeline: StandardScaler -> LogisticRegression (No PCA)
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty=penalty,
            solver='liblinear',
            random_state=23,
            max_iter=1000,
            C=C,
            class_weight = class_weight
        ))
    ])


# Instantiate both pipelines. won't use PCA, so ignore those parameters
pipeline_with_pca = create_pipeline_with_pca(
    C=0.1,
    pca_n_components=14,
    penalty='l1',
    class_weight = 'balanced'
    )

pipeline_no_pca = create_pipeline_no_pca(
    C=.04, #started with optuna, then lowered C manually to restrict as much as possible, balancing AUC mean and std dev of folding as well as final AUC validation set. 
    penalty='l1',
    #class_weight = 'balanced'
    class_weight = None
    )

pipelines = {
    'With_PCA': pipeline_with_pca,
    'No_PCA': pipeline_no_pca
}



# We'll compare AUC (or Accuracy, F1, logloss, etc.) across folds
for name, pipe in pipelines.items():
    cv_scores = cross_val_score(
        pipe, 
        X_train, 
        y_train, 
        cv=tscv, 
        scoring='roc_auc'
    )
    print(f"{name} => Scores: {cv_scores}")
    print(f"{name} => Mean AUC: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")


# Choose best pipeline
best_pipeline = pipeline_no_pca  #  pipeline_with_pca OR pipeline_no_pca, went with NO PCA

print("pipeline --> LOGIT_NO_PCA")

# Fit on the entire training set
best_pipeline.fit(X_train, y_train)

# Predict on final test set
# (Ensure X_test has the same columns and is also sorted if needed)
X_test = combined_data[1][X_full.columns]
y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= 0.5).astype(int)


# Evaluate on validation set
y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba >= 0.5).astype(int)

val_acc = accuracy_score(y_val, y_val_pred)
val_auc = roc_auc_score(y_val, y_val_proba)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation AUC     : {val_auc:.4f}")



#---------------
#  NO PCA. Check saved coefficients. drop zeroes
#--------------

# get coefs to see which are saved:
if hasattr(best_pipeline.named_steps['clf'], 'coef_'):
    coefs = best_pipeline.named_steps['clf'].coef_[0]  # 1D array of coefficients for each feature
    features = X_train.columns[:len(coefs)]  # Ensure the number of features matches the number of coefficients

    # Apply threshold to filter out small coefficients
    threshold = 1e-5
    significant_mask = np.abs(coefs) > threshold
    significant_coefs = coefs[significant_mask]
    significant_features = features[significant_mask]

    # Combine into a DataFrame for readability
    coef_df = pd.DataFrame({
        'Feature': significant_features,
        'Coefficient': significant_coefs,
        'OddsRatio': np.exp(significant_coefs)  # exponentiate for odds ratio
    })

    # Sort by coefficient magnitude or sign
    coef_df.sort_values(by=['Coefficient'], ascending=False, inplace=True)

    print(coef_df.to_string())
else:
    print("The best model does not have coefficients to display.")



#---------------
#   VIF of L1 selected features - no PCA
#----------------

from statsmodels.stats.outliers_influence import variance_inflation_factor
def compute_vif(df):
    vif_data = []
    for i in range(df.shape[1]):
        vif_data.append({
            'feature': df.columns[i],
            'VIF': variance_inflation_factor(df.values, i)
        })
    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

#Extract the trained LogisticRegression model from selected pipeline
logit_model = best_pipeline.named_steps['clf']

#  Get the final (nonzero) coefficients
#    - logit_model.coef_ has shape (1, n_features) for binary classification
coefs = logit_model.coef_[0]
nonzero_mask = (coefs != 0)

#  Subset X_train to keep only the features with nonzero coefficients
selected_features = X_train.columns[nonzero_mask]
X_train_final = X_train[selected_features]

print(f"Number of selected features: {len(selected_features)}")
print(f"Selected features:\n{selected_features}")

# Scale them before computing VIF
scaler = StandardScaler()
X_train_scaled_final = scaler.fit_transform(X_train_final)
X_train_scaled_df = pd.DataFrame(X_train_scaled_final, columns=selected_features)

# Compute VIF on this final subset
vif_df = compute_vif(X_train_scaled_df)
print(vif_df)


#-------------------------
# Confusion Matrix
#-------------------------

cm = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix:")
print("[[TP  FN]")
print(" [FP  TN]]")
print(cm[[1, 0]][:, [1, 0]])
plt.show()

#-------------------------
# Classification Report
#-------------------------
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

#-------------------------
# Log Loss
#-------------------------
log_loss_val = mean_squared_error(y_val, y_val_proba)
print(f"Validation Log Loss: {log_loss_val:.4f}")


#-------------------------
#  ROC Curve and AUC
#-------------------------
# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f"AUC: {roc_auc:.4f}")

# Print top features ranked by AUC score
if hasattr(best_pipeline.named_steps['clf'], 'coef_'):
    coefs = best_pipeline.named_steps['clf'].coef_[0]
    features = X_train.columns[:len(coefs)]
    auc_scores = []

    for feature in features:
        fpr, tpr, _ = roc_curve(y_val, X_val[feature])
        auc = roc_auc_score(y_val, X_val[feature])
        auc_scores.append((feature, auc))

    auc_scores.sort(key=lambda x: x[1], reverse=True)
    print("Top features ranked by AUC score:")
    for feature, auc in auc_scores[:20]:  # Print top features
        print(f"{feature}: {auc:.4f}")
else:
    print("The best model does not have coefficients to display.")

#Plot ROC
RocCurveDisplay.from_estimator(best_pipeline, X_val, y_val)

plt.show()


#-------------------------
# SHAP
#-------------------------

# Grab  final logistic regression from pipeline
log_reg_model = best_pipeline.named_steps['clf']

explainer = shap.Explainer(log_reg_model, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)


# Submission (commented out)
submission = pd.DataFrame({'id': combined_data[1]['id'],'rainfall': y_test_proba})
submission.to_csv(f'submission.csv', index=False)





#----------------------------------
#     [optional]  PSEUDO- OLS
#---------------------------------


def pseudo_ols_from_logit_pipeline(
    pipeline, 
    X_full, 
    X_train, 
    y_train, 
    threshold=1e-5
):
    """
    Extracts non-trivial coefficients from a trained LogisticRegression pipeline,
    then fits and evaluates an OLS model on the selected features.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline that ends with a LogisticRegression step named 'clf'.
    X_full : pd.DataFrame
        Full dataset (used to get the correct feature names).
    X_train : pd.DataFrame
        Training features used for logistic regression.
    y_train : pd.Series or np.array
        Training labels used for logistic regression.
    threshold : float, optional
        Absolute coefficient threshold to select features.
    
    Returns
    -------
    ols_model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model.
    selected_features : list
        List of feature names selected based on the threshold.
    rmse_ols : float
        RMSE of the OLS model on the training set.
    """
   

    # 1. Extract coefficients from logistic regression
    coef_flat = pipeline.named_steps['clf'].coef_[0].ravel()
    
    # 2. Create a boolean mask for coefficients above threshold
    selected_mask = np.abs(coef_flat) > threshold
    
    # 3. Use the mask to select feature names
    selected_features = X_full.columns[selected_mask]
    print("Selected features:", list(selected_features))
    
    # 4. Prepare data for OLS
    X_selected = X_train[selected_features]
    X_selected_const = sm.add_constant(X_selected)  # add intercept
    
    # 5. Fit OLS using statsmodels
    ols_model = sm.OLS(y_train, X_selected_const).fit()
    print(ols_model.summary())
    
    # 6. Evaluate OLS on the training data (or validation/test if desired)
    y_pred_ols = ols_model.predict(X_selected_const)
    rmse_ols = np.sqrt(mean_squared_error(y_train, y_pred_ols))
    print("OLS RMSE:", rmse_ols)
    
    return ols_model, list(selected_features), rmse_ols



# run psuedo ols on pipeline
ols_model, selected_feats, rmse_ols = pseudo_ols_from_logit_pipeline(
    pipeline=best_pipeline,
    X_full=X_full,
    X_train=X_train,
    y_train=y_train,
    threshold=1e-5
)


