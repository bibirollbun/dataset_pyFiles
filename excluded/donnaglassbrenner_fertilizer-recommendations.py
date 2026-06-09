from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np 
import pandas as pd 
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from itertools import combinations

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, cross_validate
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier
from scipy.stats import boxcox

import os
import ipywidgets as widgets
from IPython.display import display

import logging
logging.getLogger("sklearnex").setLevel(logging.ERROR)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
X_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

# Display basic info
print("Training data overview:", df.info())
print("Test data overview:", X_test.info())
print("\nFirst few training data rows:")
print(df.head(3))

target = 'Fertilizer Name'


df = df.drop(columns=['id'])
numeric_vars = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]

# Set an initial number of bins
num_bins = 10

# Create grid layout for histograms
num_rows = math.ceil(len(numeric_vars) / 3)  # 3 columns per row
fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))  
axes = axes.flatten()

# Plot histograms for numeric variables with adjusted integer-based bins
for i, col in enumerate(numeric_vars):
    unique_values = sorted(df[col].dropna().astype(int).unique())  # Get sorted unique integer values
    total_values = len(unique_values)
    values_per_bin = max(1, total_values // num_bins)  # Ensure each bin has the same count of unique integers

    # Create bin edges based on integer groupings
    bin_edges = [unique_values[k] for k in range(0, total_values, values_per_bin)]
    bin_edges.append(unique_values[-1] + 1)  # Extend bin range to capture all values

    sns.histplot(df[col], bins=bin_edges, kde=True, kde_kws={"bw_adjust": 2}, ax=axes[i], color="#0072CE") 
    axes[i].set_title(f"Histogram of {col}")

# Hide extra subplots
for i in range(len(numeric_vars), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


categorical_vars = [cname for cname in df.columns if df[cname].dtype == "object"]

# Create grid layout for categorical distributions
num_rows = math.ceil(len(categorical_vars) / 3)  # 3 columns per row
fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
axes = axes.flatten()

# Plot bar charts for categorical variables with corrected `countplot()`
for i, col in enumerate(categorical_vars):
    total_count = len(df[col])  # Get total observations for normalization
    proportions = df[col].value_counts(normalize=True) * 100  # Convert to percentages

    # Extend y-axis limits to accommodate labels
    max_height = max(proportions)  # Get the highest percentage value
    axes[i].set_ylim(0, max_height + 2)  # Increase upper limit by 2% for padding

    ax = sns.barplot(
        x=proportions.index, 
        y=proportions,  
        palette="tab10", ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_ylabel("Percentage")

    # Format y-axis labels as percentages with a '%' symbol
    axes[i].yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))  # No decimal places
    
    # Add percentage annotations on top of bars
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2,  # Center the text over the bar
            p.get_height() + 0.5,  # Slightly above the bar
            f"{round(p.get_height())}%",  # Format as percentage
            ha="center", fontsize=12)

    axes[i].tick_params(axis="x", rotation=45)

# Hide extra subplots
for i in range(len(categorical_vars), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# Compute correlation matrix
corr_matrix = df[numeric_vars].corr().round(2)  # Round values for readability

# Create scatterplot matrix with `corner=True`
g = sns.pairplot(df[numeric_vars], corner=True, plot_kws={"alpha": 0.6})

# Adjust layout to create space below the scatterplot matrix
g.fig.subplots_adjust(bottom=0.15)  # Reduce spacing below scatterplot matrix

# Add correlation table below the scatterplot matrix
ax_table = g.fig.add_axes([0.1, -0.25, 0.8, 0.2])  # Move table closer
ax_table.axis("off")  # Hide unnecessary axes

# Insert correlation values into the table
table = ax_table.table(
    cellText=corr_matrix.values, 
    colLabels=corr_matrix.columns, 
    rowLabels=corr_matrix.index, 
    cellLoc="center", loc="center",
    fontsize=14  
)

plt.suptitle("Scatterplot Matrix with Correlation Table", y=1.02)
plt.show()


# Compute VIF for each predictor
X = df[numeric_vars].copy()  # Independent variables
X["Intercept"] = 1  # Add intercept (needed for VIF calculation)

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns[:-1]  # Exclude intercept
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns) - 1)]

print(vif_data)



# Create a scrollable output widget
output_widget = widgets.Output(layout={'height': '400px', 'overflow': 'auto', 'border': '1px solid black'})

# Set an initial number of bins
num_bins = 10

# Generate histograms and tables inside the widget
with output_widget:
    
    # Plot histograms for numeric variables with adjusted integer-based bins
    for i, col in enumerate(numeric_vars):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        unique_values = sorted(df[col].dropna().astype(int).unique())  # Get sorted unique integer values
        total_values = len(unique_values)
        values_per_bin = max(1, total_values // num_bins)  # Ensure each bin has the same count of unique integers
    
        # Create bin edges based on integer groupings
        bin_edges = [unique_values[k] for k in range(0, total_values, values_per_bin)]
        bin_edges.append(unique_values[-1] + 1)  # Extend bin range to capture all values
        bin_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
    
        # Histogram of the numeric variable
        sns.histplot(df[col], bins=bin_edges, color="gray", alpha=0.5, ax=ax1)
        ax1.set_ylabel("Frequency", color="gray")
    
        # Create a secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Fertilizer Share (%)")
    
        # Collect data for the table
        table_data = [[f"{midpoint:.2f}" for midpoint in bin_midpoints]]  # First row: bin midpoints
        sum_row = ["100%"] * len(bin_midpoints)  # Ensure each bin sums to 100%
    
        for target_value in df[target].unique():
            target_counts = np.histogram(df[df[target] == target_value][col], bins=bin_edges)[0]
            bin_totals = np.histogram(df[col], bins=bin_edges)[0]  # Total observations per bin
            target_shares = np.divide(target_counts, bin_totals, where=bin_totals > 0)  # Normalize within each bin
    
            # Store values formatted as percentages
            table_data.append([f"{share:.1%}" if total > 0 else "0%" for share, total in zip(target_shares, bin_totals)])
    
            # Plot target share as curves centered on bin midpoints
            ax2.plot(bin_midpoints, target_shares * 100, marker='o', linestyle='-', label=target_value)  # Convert to percentage
    
        # Append total row ensuring each column sums to 100%
        table_data.append(sum_row)  
    
        # Create table with aligned columns
        col_labels = ["" for _ in bin_midpoints]  # Empty column labels
        row_labels = ["Bin Midpoints"] + [str(target) for target in df[target].unique()] + ["Total"]
    
        table = plt.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels,
                          cellLoc='center', loc='bottom', colWidths=[0.1] * len(bin_midpoints), bbox=[0.2, -0.5, 0.6, 0.35])
    
        plt.title(f"Histogram of {col} with Fertilizer Share on Dual Axis")
        plt.xlabel(col)    
        plt.show()

# Show scrollable output
display(output_widget)


for col in ["Soil Type", "Crop Type"]:
    plt.figure(figsize=(8, 5))
    
    # Compute proportions
    prop_df = df.groupby([col, 'Fertilizer Name']).size().unstack().fillna(0)
    prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)  # Convert counts to proportions
    
    # Create stacked bar plot
    ax = prop_df.plot(kind='bar', stacked=True, colormap='Set2')

    plt.title(f"Stacked Bar Plot: {col} vs Fertilizer Name")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    
    # Annotate bars with proportions
    for container in ax.containers:
        for bar in container:
            if bar.get_height() > 0:  # Avoid placing labels on zero-height bars
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_y() + bar.get_height() / 2, 
                    f"{bar.get_height():.2f}", 
                    ha='center', va='center', fontsize=10, color='black'
                )

    # Move legend outside plot
    plt.legend(title="Fertilizer Name", loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()


cat_var1 = "Soil Type"
cat_var2 = "Crop Type"

# Get distinct categories
x_categories = df[cat_var1].unique()
y_categories = df[cat_var2].unique()
target_categories = df[target].unique()

# Use a colorful palette
color_palette = plt.cm.tab10(np.linspace(0, 1, len(target_categories)))  # Vibrant colors

# Create figure and subplots
fig, axes = plt.subplots(len(y_categories), len(x_categories), figsize=(12, 8), sharex=True, sharey=True)

# Loop through each categorical combination
for i, y_val in enumerate(y_categories):
    for j, x_val in enumerate(x_categories):
        ax = axes[i, j]

        # Extract data for this category combination
        subset = df[(df[cat_var1] == x_val) & (df[cat_var2] == y_val)]
        target_counts = subset[target].value_counts(normalize=True) * 100  # Normalize to get shares

        # Assign colors dynamically for each target label
        target_colors = {target_categories[k]: color_palette[k] for k in range(len(target_categories))}
        
        # Plot stacked bar chart
        ax.bar(target_categories, target_counts.reindex(target_categories, fill_value=0),
               color=[target_colors[tc] for tc in target_categories], width=0.9)

        # Remove axis labels inside cells
        ax.set_xticks([])
        ax.set_yticks([])

# **Add category labels around the perimeter**
# X-axis labels (bottom)
for j, x_val in enumerate(x_categories):
    axes[-1, j].set_xlabel(x_val, fontsize=12, rotation=45, ha="right")

# Y-axis labels (left side)
for i, y_val in enumerate(y_categories):
    axes[i, 0].set_ylabel(y_val, fontsize=12, rotation=0, ha="right", va="center")

# Add axis titles outside the grid
fig.text(0.5, 0.02, cat_var1, ha="center", va="center", fontsize=14)
fig.text(0.02, 0.5, cat_var2, ha="center", va="center", fontsize=14, rotation=90)

plt.suptitle(f"Fertilizer Distribution by {cat_var1} and {cat_var2}", fontsize=16)

plt.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])  # Adjust layout to avoid overlap
plt.show()


environment_vars = ['Temparature', 'Humidity', 'Moisture']
nutrient_vars = ['Nitrogen', 'Potassium', 'Phosphorous']

def add_features(dff):
    # Add nutrient ratios; I'm taking logs, otherwise they're pretty heavily skewed
    dff['log_N_P_ratio'] = np.log((dff['Nitrogen']+1) / (dff['Phosphorous'] + 1))  #Add 1 to numerator as well as denominator to avoid log(0)
    dff['log_N_K_ratio'] = np.log((dff['Nitrogen']+1) / (dff['Potassium'] + 1))
    dff['log_P_K_ratio'] = np.log((dff['Phosphorous']+1) / (dff['Potassium'] + 1))
    
    # Total nutrients
    dff['NPK_sum'] = dff['Nitrogen'] + dff['Phosphorous'] + dff['Potassium']
            
    # Normalized NPK values
    dff['N_proportion_bc'], fitted_lambda = boxcox(dff['Nitrogen'] / dff['NPK_sum'])  
    print(f"Box-Cox applied to 'N_proportion'. Lambda: {fitted_lambda}")
    dff['P_proportion_bc'], fitted_lambda = boxcox((dff['Phosphorous'] +1)/ dff['NPK_sum']) 
    print(f"Box-Cox applied to 'P_proportion'. Lambda: {fitted_lambda}")
    dff['K_proportion_bc'], fitted_lambda = boxcox((dff['Potassium'] + 1) / dff['NPK_sum']) 
    print(f"Box-Cox applied to 'K_proportion'. Lambda: {fitted_lambda}")
    
    # Moisture Temperature Ratio
    dff['MoistureTemp_Ratio_bc'], fitted_lambda = boxcox(dff['Moisture'] / (dff['Temparature'] + 1e-5))
    print(f"Box-Cox applied to 'MoistureTemp_Ratio'. Lambda: {fitted_lambda}")
    
    # Composite Weather Index
    dff['WaterStressIndex_bc'], fitted_lambda = boxcox((dff['Moisture'] + dff['Humidity']) / (dff['Temparature'] + 1e-5))
    print(f"Box-Cox applied to 'WaterStressIndex'. Lambda: {fitted_lambda}")
    
    # We add some second-degree interactions. We don't want to add e.g. an interaction between N_proportion and NPK_sum though.
    numeric_interactions = pd.DataFrame()
    
    for var1, var2 in list(combinations(environment_vars, 2)):
        colname = f"{var1}_x_{var2}_bc"
        numeric_interactions[colname], fitted_lambda = boxcox(dff[var1] * dff[var2])
        print(f"Box-Cox applied to '{var1}_x_{var2}'. Lambda: {fitted_lambda}")
 
    # I'm taking the log of the nutrient interactions, otherwise they're pretty heavily skewed   
    for var1, var2 in list(combinations(nutrient_vars, 2)):
        colname = f"{var1}_x_{var2}_bc"
        numeric_interactions[colname], fitted_lambda = boxcox(dff[var1] * dff[var2]+1)
        print(f"Box-Cox applied to '{var1}_x_{var2}'. Lambda: {fitted_lambda}")

    dff = pd.concat([dff.drop(columns=[col for col in dff.columns if col.endswith('_bin')]), numeric_interactions], axis=1)
        
    return dff
    
df = add_features(df)
X_test = add_features(X_test)

new_features = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64'] and cname not in numeric_vars and not cname.endswith("_bin")]
print("Our new features are:", new_features)
numeric_vars = numeric_vars + new_features
print(numeric_vars)



# Create a scrollable output widget
output_widget = widgets.Output(layout={'height': '400px', 'overflow': 'auto', 'border': '1px solid black'})

# Set an initial number of bins
num_bins = 10

with output_widget:
    # Plot histograms for numeric variables with adjusted integer-based bins
    for i, col in enumerate(new_features):
        fig, ax1 = plt.subplots(figsize=(10, 6))
    
        # Apply quantile-based binning. Am using quantile binning so the fertilizer share estimates are about equally reliable.
        df[f"{col}_bin"], bin_edges = pd.qcut(df[col], q=num_bins, labels=False, retbins=True, duplicates="drop")
    
        # Compute bin midpoints for visualization
        bin_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
        
        # Histogram of the numeric variable 
        sns.histplot(df[col], bins=bin_edges, kde=True, kde_kws={"bw_adjust": 2}, color="gray")
        ax1.set_ylabel("Frequency", color="gray")
    
        # Create a secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Fertilizer Share (%)")
    
        # Collect data for the table
        table_data = [[f"{midpoint:.2f}" for midpoint in bin_midpoints]]  # First row: bin midpoints
        sum_row = ["100%"] * len(bin_midpoints)  # Ensure each bin sums to 100%
    
        for target_value in df[target].unique():
            target_counts = np.histogram(df[df[target] == target_value][col], bins=bin_edges)[0]
            bin_totals = np.histogram(df[col], bins=bin_edges)[0]  # Total observations per bin
            target_shares = np.divide(target_counts, bin_totals, where=bin_totals > 0)  # Normalize within each bin
    
            # Store values formatted as percentages
            table_data.append([f"{share:.1%}" if total > 0 else "0%" for share, total in zip(target_shares, bin_totals)])
    
            # Plot target share as curves centered on bin midpoints
            ax2.plot(bin_midpoints, target_shares * 100, marker='o', linestyle='-', label=target_value)  # Convert to percentage
    
        # Append total row ensuring each column sums to 100%
        table_data.append(sum_row)  
    
        # Create table with aligned columns
        col_labels = ["" for _ in bin_midpoints]  # Empty column labels
        row_labels = ["Bin Midpoints"] + [str(target) for target in df[target].unique()] + ["Total"]
    
        table = plt.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels,
                          cellLoc='center', loc='bottom', colWidths=[0.1] * len(bin_midpoints), bbox=[0.2, -0.5, 0.6, 0.35])
    
        plt.title(f"Histogram of {col} with Fertilizer Share on Dual Axis")
        plt.xlabel(col)
        plt.show()
        
# Show scrollable output
display(output_widget)
df = df.drop(columns=[col for col in df.columns if col.endswith('_bin')])


#Label encode the target
L_encoder = LabelEncoder()
df[target] = L_encoder.fit_transform(df[target])

cat_features = [cname for cname in df.columns if df[cname].dtype == "object"]
# One-hot encode categorical variables, both have low cardinality 
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

def one_hot_encode_cat_vars(dff):    
    # Fit and transform the categorical variables
    OH_encoded_array = OH_encoder.fit_transform(dff[cat_features])
    
    OH_encoded_feature_names = OH_encoder.get_feature_names_out(cat_features)
    dummy_features = OH_encoded_feature_names.tolist()  # Features to exclude from standardization
    
    # Convert to DataFrame with proper column names
    OH_encoded_df = pd.DataFrame(OH_encoded_array, columns=OH_encoded_feature_names)
    
    # Reset index to align with original DataFrame
    OH_encoded_df.index = dff.index
    
    # Concatenate with original DataFrame 
    dff = pd.concat([dff.drop(columns=cat_features), OH_encoded_df], axis=1)

    soil_dummies = [col for col in dff.columns if col.startswith('Soil Type_')]
    crop_dummies = [col for col in dff.columns if col.startswith('Crop Type_')]  
    
    # Cross-dummy interactions    
    interaction_dict = {}
    
    for c1 in soil_dummies:
        for c2 in crop_dummies:
            colname = f"{c1}_x_{c2}"
            interaction_dict[colname] = dff[c1] * dff[c2]
            
    cross_dummy_interactions = pd.DataFrame(interaction_dict)

    dff = pd.concat([dff, cross_dummy_interactions], axis=1) #numeric_dummy_interactions, 
    return dff

df = one_hot_encode_cat_vars(df)
X_test = one_hot_encode_cat_vars(X_test)



# Create a scrollable output widget
output_widget = widgets.Output(layout={'height': '400px', 'overflow': 'auto', 'border': '1px solid black'})

with output_widget:
    # Feature set
    
    
    # Define feature matrix (X) and target vector (y)
    X = df.drop(columns=target)
    y = df[target]
    num_features = numeric_vars # + numeric_interactions
    
    print("Our features under consideration are:", X.columns)
    print("Vars ending in _bin:", [col for col in df.columns if col.endswith("_bin")])
    print(X.head())
    print(y.head())
    print(num_features)
        
# Show scrollable output
display(output_widget)


# Define MAP@3 scoring function
def mapk(y_true, y_pred, k=3):
    """
    Compute MAP@k for single-label ground truths.
    y_true: list of lists, each inner list contains the single true label index.
    y_pred: list of lists, each inner list contains k predicted label indices.
    """
    N = len(y_true)
    scores = []
    for true, preds in zip(y_true, y_pred):
        score = 0.0
        found = False
        for i, p in enumerate(preds[:k], start=1):
            if p in true and not found:
                score = 1.0 / i
                found = True
                break
        scores.append(score)
    return np.mean(scores)
    
def map_at_3(y_true, y_pred_proba):
    top_3_preds = np.argsort(y_pred_proba, axis=1)[:, -3:]  # Get top-3 predictions
    y_pred = top_3_preds.tolist()
    y_true = [[i] for i in y_true]
    
    return mapk(y_true, y_pred)
    pass
    
# Create a scorer that we'll use in cross_val_score
# Detect Kaggle environment
is_kaggle = "KAGGLE_URL_BASE" in os.environ

# Set `map3_scorer` based on environment
if is_kaggle:
    map3_scorer = make_scorer(map_at_3, needs_proba=True, greater_is_better=True)
else:
    map3_scorer = make_scorer(map_at_3, response_method="predict_proba", greater_is_better=True)

print("Using Kaggle-specific scorer:", is_kaggle)


# Function to plot learning curves
def plot_learning_curve(model, model_name, X, y, preprocess_method):
    train_sizes, train_scores, val_scores = learning_curve(
        Pipeline(steps=[('preprocessor', preprocess_method), ('model', model)]),
        X, y, cv=5, scoring=map3_scorer, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Compute mean and standard deviation
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # Plot learning curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, marker='o', label='Training MAP@3')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, marker='o', label='Validation MAP@3', color='red')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')

    plt.xlabel('Training Set Size')
    plt.ylabel('MAP@3 Score')
    plt.title(f'Learning Curve ({model_name})')
    plt.legend()
    plt.grid()
    plt.show()


# Function to extract feature importance
def get_feature_importance(model, X, y, preprocess_method):
    pipeline = Pipeline(steps=[('preprocessor', preprocess_method), ('model', model)])
    pipeline.fit(X, y)

    if hasattr(model, "feature_importances_"):  
        importance = model.feature_importances_
        group = "tree"
    elif hasattr(model, "coef_"):  
        importance = np.abs(model.coef_).mean(axis=0)
        group = "logistic"
    elif hasattr(model, "feature_log_prob_"):  
        importance = np.mean(model.feature_log_prob_, axis=0)  
        group = "naive_bayes"
    else:
        return None, None

    feature_names = preprocess_method.get_feature_names_out()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).set_index("Feature")
    
    return importance_df.sort_values(by="Importance", ascending=False).head(9) if not importance_df.empty else None, group


# Preprocessing pipelines
# Pipelines other than naive bayes include categorical predictors
preprocessor_tree_logistic = ColumnTransformer([
    ("num", StandardScaler(), num_features),],  # Apply standardization only to numeric columns
    remainder="passthrough", verbose_feature_names_out=False)  # Keep one-hot encoded features unchanged

# Models with preprocessing methods
models = {
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=0), preprocessor_tree_logistic),
    "XGB": (XGBClassifier(tree_method="gpu_hist", predictor="gpu_predictor",n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42), preprocessor_tree_logistic),
    "Logistic": (LogisticRegression(solver='lbfgs', max_iter=500), preprocessor_tree_logistic),
}

# Collect feature importance for all models
importance_data_tree = {}
importance_data_logistic = {}
feature_order = set()
evaluation_results = []

# Compute feature importance and learning curves
for name, (model, preprocess_method) in models.items():
    
    importance_df, group = get_feature_importance(model, X, y, preprocess_method)
    
    if importance_df is not None:
        feature_order.update(importance_df.index)
        if group == "tree":
            importance_data_tree[name] = importance_df["Importance"]
        elif group == "logistic":
            importance_data_logistic[name] = importance_df["Importance"]
    
    plot_learning_curve(model, name, X, y, preprocess_method)

    # Compute evaluation scores
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = cross_validate(Pipeline([("preprocessor", preprocess_method), ("model", model)]),
                                X, y, cv=cv_strategy, scoring=map3_scorer, return_train_score=True)
    
    train_score = cv_results['train_score'].mean()
    val_score = cv_results['test_score'].mean()    
    
    evaluation_results.append([name, round(train_score, 4), round(val_score, 4)])

# Convert to DataFrames, ensuring feature consistency
feature_order = sorted(feature_order)
importance_table_tree = pd.DataFrame(importance_data_tree, index=feature_order).fillna(0)
importance_table_logistic = pd.DataFrame(importance_data_logistic, index=feature_order).fillna(0)

# Summary Table
summary_table = pd.DataFrame(evaluation_results, columns=["Model", "Training MAP@3", "Validation MAP@3"])
print("\nModel Evaluation Summary:")
print(summary_table.to_string(index=False))

# Create Feature Importance Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 2]})

# Left: Tree-Based Models
sns.heatmap(importance_table_tree, annot=True, cmap="Blues", fmt=".4f", linewidths=0.5, ax=axes[0])
axes[0].set_title("Top 9 Features (RFC & XGB)")
axes[0].set_xlabel("Model")
axes[0].set_ylabel("Feature")

# Right: Logistic Regression
sns.heatmap(importance_table_logistic, annot=True, cmap="Reds", fmt=".4f", linewidths=0.5, ax=axes[1])
axes[1].set_title("Top 9 Features (Logistic Regression)")
axes[1].set_xlabel("Model")
axes[1].set_ylabel("Feature")

# Adjust layout and show
plt.tight_layout()
plt.show()



# Define model and pipeline
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor_tree_logistic), ('model', model)])

# Train the model
pipeline.fit(X, y)

# Get predicted probabilities for all 7 labels
y_pred_proba = pipeline.predict_proba(X_test.drop(columns=["id"]))

# Get the top 3 labels for each row
top_3_labels = np.argsort(y_pred_proba, axis=1)[:, -3:]  # Indices of top 3 labels
top_3_labels = np.flip(top_3_labels, axis=1)  # Ensure descending order

# Decode top-3 label indices back to original string labels
decoded_labels = [L_encoder.inverse_transform(row) for row in top_3_labels]

# Now you can format for output if needed
predicted_strings = [' '.join(row) for row in decoded_labels]

# Create DataFrame with space-delimited predictions
df_results = pd.DataFrame({
    "id": X_test["id"],
    target: predicted_strings
})

print(df_results.head())

# Save to CSV
df_results.to_csv("predictions.csv", index=False, sep=',')  # comma-separated CSV with space-delimited labels
print("CSV file 'predictions.csv' has been saved.")




