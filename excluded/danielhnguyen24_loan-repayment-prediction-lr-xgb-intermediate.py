import kagglehub
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator
from IPython.display import Markdown, display
import missingno as msno
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
import warnings

# Suppress warnings that specifically deal with invalid value encounters
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in greater')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in less')


# List all file paths in the Kaggle input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Read the CSV file into a pandas DataFrame
train_set = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_set = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sub_samp = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


# Show the first 5 rows of the train_set dataset to check the data
display(train_set.head(5))

# Display the shape of the dataset (number of rows, number of columns)
print(f'The shape of the dataset by number of rows and columns: {train_set.shape}')


# Show the first 5 rows of the train_set dataset to check the data
display(test_set.head(5))

# Display the shape of the dataset (number of rows, number of columns)
print(f'The shape of the dataset by number of rows and columns: {test_set.shape}')


# Show the first 10 rows of the train_set dataset to check the data
display(sub_samp.head(10))

# Display the shape of the dataset (number of rows, number of columns)
print(f'The shape of the dataset by number of rows and columns: {sub_samp.shape}')

# Check the result of sample submission
sub_samp[sub_samp['loan_paid_back'] == '1'].sum()


# Summary of missing (NaN/Null) and zero values per column then sort by highest combined NaN/Null + Zeros
missing_summary = (
    # Get the data type (e.g., float, object) for every column in the DataFrame
    train_set.dtypes.to_frame(name="Type")
    # Join the 'Type' column with a new DataFrame containing calculated statistics
    .join(pd.DataFrame({
        "Missed values": train_set.isna().sum(), # Calculate the count of missing (NaN or Null) values for each column
        "Percentage Missed": round(train_set.isna().mean() * 100, 2), # Calculate the percentage of missing values, round to 2 decimal places
        "Zeros": (train_set == 0).sum(),         # Calculate the count of explicit zero (0) values for each column
        "Percentage Zeros": round((train_set == 0).mean() * 100, 2)})) # Calculate the percentage of explicit zero values, round to 2 decimal places
    # Use .assign() to create a new calculated column
    .assign(**{"NaN/Null + Zeros": lambda df: df["Missed values"] + df["Zeros"]})
    # Sort the resulting DataFrame in descending order based on the combined count
    .sort_values(by="NaN/Null + Zeros", ascending=False))

# Display the final summary table
display(missing_summary)
print(f'Dataset Dimensions: Rows and Columns {train_set.shape}')


#Â SortÂ train_setÂ byÂ annual_incomeÂ andÂ visualizeÂ missingnessÂ matrix
train_set = train_set.sort_values(by='annual_income')
msno.matrix(train_set)
plt.title("Missingness Matrix Sorted by Income", fontsize=25, fontweight='bold', pad=20)
plt.show()


# Drop duplicates if any
train_set = train_set.drop_duplicates()

# Re-check for duplicates
duplicates = train_set.duplicated(keep=False)
print(train_set[train_set.duplicated(keep=False)])
print(f'Number of duplicate rows: {duplicates.sum()}')


# Standardize column names
train_set.columns = (
    train_set.columns
    .str.strip()  
    .str.replace(r"\s+", "_", regex=True)    
    .str.replace(r"[^\w_]", "", regex=True)
    .str.lower())

# Print out the column name after standardized
print(train_set.columns)


# Select columns with numerical data types (integers, floats) and store their names in a list
numerical_columns = train_set.select_dtypes(include=[np.number]).columns.tolist()

# Select columns with object (string/text) or category data types and store their names in a list
categorical_columns = train_set.select_dtypes(include=["object", "category"]).columns.tolist()

# Preview the columns
print(f"Numeric data columns: {numerical_columns}")
print(f"Categorical data columns: {categorical_columns}")


# Ensure numeric columns use the correct dtype (float64 for calculations)
train_set[numerical_columns] = train_set[numerical_columns].astype("float64")

# Ensure categorical columns use the 'category' dtype except chosen columns
exclude_cols = []
# Identify object columns eligible for conversion
categorical_columns = train_set.select_dtypes(include=["object", "category"]).columns.difference(exclude_cols)
train_set[categorical_columns] = train_set[categorical_columns].astype("category")

# Recheck the data types for the train_set
print(train_set.dtypes)


# Calculate and print the descriptive statistics for numerical columns
if numerical_columns:
    # .T (transpose) is used to display the statistics as rows and column names as columns for better readability.
    display(f"Descriptive statistics (Numeric):", train_set[numerical_columns].describe().T)


# Setup for plotting
plt.figure(figsize=(12, 6))

# Visualization using histogram
plt.hist(train_set['loan_amount'], 
    bins=50, edgecolor='white', color='#1f77b4', 
    alpha=0.8, rwidth=0.98, linewidth=0.5)

# Calculate Statistics
mean_value = train_set['loan_amount'].mean()
median_value = train_set['loan_amount'].median()
lower_amount = train_set['loan_amount'].quantile(0.01)
upper_amount = train_set['loan_amount'].quantile(0.99)

# Add Reference lines
plt.axvline(mean_value, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_value:,.0f}')
plt.axvline(median_value, color='green', linestyle='-.', linewidth=1.5, label=f'Median: {median_value:,.0f}')
plt.axvline(lower_amount, color='purple', linestyle='solid', linewidth=2, label=f"1% = {lower_amount:.0f}")
plt.axvline(upper_amount, color='orange', linestyle='solid', linewidth=2, label=f"99% = {upper_amount:.0f}")

# Labels and Title
plt.xlabel('Loan Amount (in GBP)', fontsize=12, fontweight='bold')
plt.ylabel('Number of Loans', fontsize=12, fontweight='bold')
plt.title('Distribution of Loan Amounts', fontsize=16, fontweight='bold', loc='center', pad=20)
plt.legend(fontsize=10, frameon=True)

# --- Grid and Layout ---
plt.grid(axis='y', alpha=0.5, linestyle='--')
plt.tight_layout() 
plt.show()


# Calculate actual target distribution
target_counts = train_set['loan_paid_back'].value_counts()
target_pct = (target_counts / len(train_set) * 100).round(1)

# Ensure consistent label order (assuming '0' = Not Paid Back, '1' = Paid Back)
default_pct = target_pct.get(0, 0)
paid_back_pct = target_pct.get(1, 0)

# For a 100% stacked bar, only need a single row of data.
data = pd.DataFrame({'Status': ['Status'], 'Not Paid back': [default_pct], 'Paid back': [paid_back_pct]})

# Define custom colors
colour_default = "#ff7f0e"  
colour_paid = "#1f77b4" 
# palette={0:"#ff7f0e", 1: "#1f77b4"}

# Create the plot
plt.figure(figsize=(12, 6))
fig, ax = plt.subplots(figsize=(12, 2))

# Not Paid back bar (bottom/left segment)
ax.barh(y=data['Status'], width=data['Not Paid back'], color=colour_default, label='Not Paid back')

# Paid Back bar (stacked to the right)
ax.barh(y=data['Status'], width=data['Paid back'], left=data['Not Paid back'], color=colour_paid, label='Paid Back')

# Add percentage text labels ---
ax.text(x=data['Not Paid back'].iloc[0] / 2, y=0, s=f"{data['Not Paid back'].iloc[0]:.1f}%",
    ha='center', va='center', color='white', fontsize=16, fontweight='bold')

ax.text(x=data['Not Paid back'].iloc[0] + data['Paid back'].iloc[0] / 2, y=0, s=f"{data['Paid back'].iloc[0]:.1f}%", 
        ha='center', va='center', color='white', fontsize=16, fontweight='bold')

# Set X-axis limits and major ticks to 0-100 to match the screenshot
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 20)) # Ticks at 0, 20, 40, 60, 80, 100

# Title and Axes
ax.set_title('Loan Repayment Status Distribution', fontsize=18, fontweight='bold', loc='center', pad=16)
ax.set_xlabel('') # Remove X-axis label
ax.tick_params(axis='y', length=0) # Remove Y-axis ticks

# Hide the chart border (spines) for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True) # Keep the bottom axis line

# Legend (use a handle for each color and label)
legend_handles = [plt.Rectangle((0,0), 1, 1, fc=colour_paid), plt.Rectangle((0,0), 1, 1, fc=colour_default)]
legend_labels = ['Paid back', 'Not Paid back']
ax.legend(legend_handles, legend_labels, frameon=False, bbox_to_anchor=(1.05, 0.95), loc='upper left', fontsize=12)

# Visualize
plt.tight_layout()
plt.show()


# Set seaborn style
sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.2)
warnings.filterwarnings("ignore", module="seaborn")

# Create the scatter plot using seaborn
plt.figure(figsize=(12, 6))
sns.scatterplot(x='loan_amount', y='credit_score', data=train_set,
            hue="loan_paid_back", style="loan_paid_back", 
            alpha=0.8, edgecolor='w', linewidth=0.5, legend=False,
            palette={0:"#ff7f0e", 1: "#1f77b4"},
            markers={0: "X", 1: "o"}) 

# Set labels and title
plt.xlabel('Loan Amount (GBP)', fontsize=12, fontweight='bold')
plt.ylabel('Credit Score', fontsize=12, fontweight='bold')
plt.title('Loan Amount versus Credit Score (coloured by Payment Status)', fontsize=16, fontweight='bold', pad=20)

# Customize the legend to be more informative
handles = [
    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#ff7f0e', markersize=8, label='Not Paid back'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label='Paid back')]

plt.legend(handles=handles, title="Payment Status", fontsize=11, title_fontsize=13, loc='best', frameon=True, shadow=True)

# Display the visualization
plt.show()


# Remove id column from the numeric columns
feature = train_set[numerical_columns].drop(columns=['id'], errors='ignore')
print("Selected numerical features:", feature.columns.tolist())

# Compute correlation matrix
corr = feature.corr(numeric_only=True).round(3)

# Initialize a p-Values Matrix
pvals = pd.DataFrame(np.zeros((feature.shape[1], feature.shape[1])), columns=feature.columns, index=feature.columns)

# Compute p-Values for Each Pair
for i in feature.columns:
    for j in feature.columns:
        if i == j:
            pvals.loc[i, j] = 0
        else:
            _, p = pearsonr(feature[i], feature[j])
            pvals.loc[i, j] = p

# Define a Function for Significance Stars
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

# Create labels with correlation + significance stars
labels = corr.copy().astype(str)
for i in corr.columns:
    for j in corr.index:
        labels.loc[j, i] = f"{corr.loc[j, i]:.3f}{significance_stars(pvals.loc[j, i])}"

# Format index/columns: capitalize words and remove underscores
corr.index = [label.replace('_', ' ').title() for label in corr.index]
corr.columns = [label.replace('_', ' ').title() for label in corr.columns]

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', 
            annot=labels, fmt='', linewidths=0.5, linecolor='gray', center=0, annot_kws={"size": 12})

# Set label and title
plt.title('Numerical Feature Correlation Heatmap\n(with significance levels)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(fontsize=12, rotation=90, ha='right', fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Add caption for significance levels
caption = (r"$\bf{Note}$: Significance levels: *** p<0.001, ** p<0.01, * p<0.05. "
           "No star indicates pâ‰¥0.05 (not statistically significant).")
plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)

# Display the visualization
plt.tight_layout()
plt.show()


# Create categorical plot
plt.figure(figsize=(12, 6))
a = sns.catplot(x='gender', col='loan_purpose', kind='count',
    data=train_set, hue='loan_paid_back',
    palette={0: "#ff7f0e", 1: "#1f77b4"},
    height=5, aspect=1.0, col_wrap=4, legend=False)

# Set labels and titles
a.set_axis_labels("Gender", "Number of Loans")
a.set_titles(col_template="Loan Purpose: {col_name}")

# Adjust title and spacing
a.fig.suptitle('Loan Applications by Gender and Loan Purpose',
               x=0.5, y=0.98, fontsize=16, fontweight='bold')
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    a.fig.tight_layout(rect=[0, 0.1, 0.9, 0.95])


# Add legend manually with improved readability
legend_labels = ['0 = Not Paid back', '1 = Paid back']
legend = a.fig.legend(
    labels=legend_labels, title='Payment Status',
    loc='upper center', bbox_to_anchor=(0.5, 0.08), ncol=2,
    frameon=True,  # adds border
    edgecolor='black')

# Customize legend text appearance
plt.setp(legend.get_texts(), fontsize=11)
plt.setp(legend.get_title(), fontsize=12, fontweight='bold')

# Show plot
plt.show()


# Create crosstab between employment_status, education_level and loan_paid_back
matrix1 = pd.crosstab(train_set['employment_status'], train_set['education_level'],
    values=(1 - train_set['loan_paid_back']) * 100, aggfunc='mean').round(2)

# Format values as percentages for annotation
annot_labels = matrix1.astype(str) + '%'

# Set up figure
plt.figure(figsize=(12, 6))

# Define a custom color palette
custom_cmap = sns.color_palette(["#4e79a7", "#f28e2b", "#e15759"])
cmap = sns.color_palette(custom_cmap, as_cmap=True)

# Draw heatmap with colorbar settings
sns.heatmap(matrix1, annot=annot_labels,
    fmt="", cmap=cmap, linewidths=0.5, linecolor='gray',
    annot_kws={"size": 12, "weight": "bold", "color": "white"},
                cbar_kws={"label": "Percentage of Loans Not Paid Back (%)",
                          "ticks": [0, 20, 40, 60, 80, 100]},  # Set scale 0-100}
                          vmin=0, vmax=100) # Minimum and maximum value of colorbar

# Customize colorbar title font
colorbar = plt.gcf().axes[-1]  # Get the colorbar axis
colorbar.set_ylabel("Percentage of Loans Not Paid Back (%)", fontsize=12, fontweight='bold')

# Titles and labels
plt.title("Employment Ã— Education Level Matrix", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Marital Status", fontsize=12, fontweight='bold')
plt.ylabel("Employment Status", fontsize=12, fontweight='bold')

# Aesthetic adjustments
plt.xticks(fontsize=12, rotation=0, ha='center')
plt.yticks(fontsize=12)
plt.gca().set_facecolor("#f6f5f5")
plt.tight_layout()

# Add caption annotation
plt.figtext(0.95, -0.05, "Created by Daniel (Viet) Nguyen",
    horizontalalignment='right', fontsize=10, color='gray')

# Display the visualization
plt.show()


# Set up figure
plt.figure(figsize=(12, 6)) 

# Plotting the Bar Chart
sns.barplot(x='grade_subgrade', y='loan_paid_back', data=train_set, palette='RdYlGn_r', orient='v', errorbar=None)

# Title and Labels
plt.title('Repayment Rate by Loan Grade', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Loan Grade', fontsize=12, fontweight='bold')
plt.ylabel('Repayment rate', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.8) 
plt.xticks(rotation=0, ha='center', fontsize=9)
plt.ylim(0.0, 1.0) # Set limit to 1.0 to reflect a rate/proportion, matching the plot
plt.yticks(np.arange(0.0, 1.1, 0.2), fontsize=9) # Ticks at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
sns.despine(top=False, right=False, left=False, bottom=False, trim=True)

# Remove top and right ticks completely
plt.tick_params(axis='both', which='both', top=False, right=False)

# Visualize
plt.tight_layout()
plt.show()


# Calculate the median
median = train_set['loan_amount'].median()

# Calculate the Median Absolute Deviation (MAD)
mad = (train_set['loan_amount'] - median).abs().median()

# Compute the Modified Z-Score
modified_z_scores = 0.6745 * (train_set['loan_amount'] - median) / mad

# Identify outliers using a threshold of 3.5
outlier_mask_x = modified_z_scores.abs() > 3.5
num_outliers = outlier_mask_x.sum()
percent_outliers = (num_outliers / len(train_set)) * 100

print(f"Identified {num_outliers} outliers ({percent_outliers:.2f}% of the dataset) using a Modified Z-score threshold of 3.5.")

# Remove outliers
train_set_filtered = train_set[~outlier_mask_x].copy()

# Display dataset shapes
print(f"\nTrained dataset shape before outlier removal: {train_set.shape}")
print(f"Trained dataset shape after outlier removal: {train_set_filtered.shape}")

# Verify max loan amount after outlier removal
print(f"New max loan amount: {train_set_filtered['loan_amount'].max():,.2f}")


# Calculate the median
median = test_set['loan_amount'].median()

# Calculate the Median Absolute Deviation (MAD)
mad = (test_set['loan_amount'] - median).abs().median()

# Compute the Modified Z-Score
modified_z_scores = 0.6745 * (test_set['loan_amount'] - median) / mad

# Identify outliers using a threshold of 3.5
outlier_mask_y = modified_z_scores.abs() > 3.5
num_outliers = outlier_mask_y.sum()
percent_outliers = (num_outliers / len(test_set)) * 100

print(f"Identified {num_outliers} outliers ({percent_outliers:.2f}% of the dataset) using a Modified Z-score threshold of 3.5.")

# Remove outliers
test_set_filtered = test_set[~outlier_mask_y].copy()

# Display dataset shapes
print(f"Tested dataset shape before outlier removal: {test_set.shape}")
print(f"Tested dataset shape after outlier removal: {test_set_filtered.shape}")

# Verify max loan amount after outlier removal
print(f"New max loan amount: {test_set_filtered['loan_amount'].max():,.2f}")


# Install necessary library
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score


# Separate target and features
X_train = train_set.drop(columns=['id', 'loan_paid_back'])
y_train = train_set['loan_paid_back']
X_test = test_set.drop(columns=['id'])

# Identify numerical and categorical columns
numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# One-hot encode categorical variables
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

# Align columns between train and test sets
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# Standardize numerical features to ensures features with larger magnitudes don't disproportionately influence the model coefficients
scaler = StandardScaler()
X_train_encoded[numerical_columns] = scaler.fit_transform(X_train_encoded[numerical_columns])
X_test_encoded[numerical_columns] = scaler.transform(X_test_encoded[numerical_columns])


print('Processing... Please wait.')
# Setup: Define candidate C values and initialize results
# -------------------------------------------------------------------
C_values = [0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 5, 10]  # Smaller C -> stronger regularization
l1_metrics = []  # List to store [C, non-zero coefficients, ROC-AUC, Recall, Accuracy] for each C
## Flag: Decide whether to run cross-validation or just train on full set
use_cv = False  # Set True for 5-fold CV, False for training-set evaluation 
## Stratified Cross-Validation setup (keeps class distribution consistent across folds)
if use_cv:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
# Loop over each C to evaluate model
# -------------------------------------------------------------------
for c in C_values:
    ## Define Logistic Regression with L1 penalty
    logreg_L1 = LogisticRegression(
        penalty='l1', C=c, solver='liblinear', # L1 regularization and compatibility with L1
        class_weight='balanced',               # Handle imbalanced classes (Class Weighting = Algorithmic methods)
        max_iter=1000, random_state=42)        # Ensure Reproducibility
    
    if use_cv:
        ## Perform cross-validation and compute ROC-AUC, Recall and Accuracy for each fold
        cv_auc = cross_val_score(logreg_L1, X_train_encoded, y_train, cv=cv, scoring='roc_auc')
        mean_auc = np.mean(cv_auc)

        cv_recall = cross_val_score(logreg_L1, X_train_encoded, y_train, cv=cv, scoring='recall')
        mean_recall = np.mean(cv_recall)
        
        cv_accuracy = cross_val_score(logreg_L1, X_train_encoded, y_train, cv=cv, scoring='accuracy')
        mean_accuracy = np.mean(cv_accuracy)
        
        ## Fit model on full training data after CV to count non-zero coefficients
        logreg_L1.fit(X_train_encoded, y_train)
        
    else:
        ## Fit model once on full training data
        logreg_L1.fit(X_train_encoded, y_train)
        
        ## Predict labels and probabilities
        y_pred = logreg_L1.predict(X_train_encoded)
        y_pred_prob = logreg_L1.predict_proba(X_train_encoded)[:, 1]
        
        ## Compute metrics
        mean_auc = roc_auc_score(y_train, y_pred_prob)
        mean_recall = recall_score(y_train, y_pred)
        mean_accuracy = accuracy_score(y_train, y_pred)
    
    ## Count non-zero coefficients (how many features retained)
    non_zero = np.count_nonzero(logreg_L1.coef_)
    
    ## Append and store all results for current C
    l1_metrics.append([c, non_zero, mean_auc, mean_recall, mean_accuracy])

## Notify of completion
print('Run complete. Results are ready.')

# Summarize results and select best C
# -------------------------------------------------------------------
l1_results = pd.DataFrame(l1_metrics, columns=['C', 'Non-Zero Coeffs', 'Mean ROC-AUC', 'Recall', 'Accuracy'])

## Identify best C based on best Recall rate (Sensitivity)
best_row = l1_results.loc[l1_results['Recall'].idxmax()]
best_C = best_row['C']

## Display summary
print("\nğŸ“Š L1 Regularization Performance Summary:")
print(l1_results.to_string(index=False))
print(f"\nğŸ�† Best C Value: {best_C} (Recall = {best_row['Recall']:.4f}, "
      f"Mean ROC-AUC = {best_row['Mean ROC-AUC']:.4f}, Accuracy = {best_row['Accuracy']:.4f})")

# Train final L1 model on full training set
# -------------------------------------------------------------------
final_L1 = LogisticRegression(
    penalty='l1', C=best_C, solver='liblinear',
    max_iter=1000, random_state=42)
final_L1.fit(X_train_encoded, np.ravel(y_train))

print(f"\nâœ… Final L1 Model trained successfully using C={best_C}.")

# Feature importance extraction and visualization
# -------------------------------------------------------------------
feature_importance_L1 = pd.DataFrame({
    'Feature': X_train_encoded.columns,
    'Coefficient': final_L1.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

## Optional: Print sorted feature importance table
# print("\nğŸ“Š L1 Logistic Regression Coefficients (sorted):")
# print(feature_importance_L1.to_string(index=False))

# Visualization - Coefficients from Final Model
# -------------------------------------------------------------------
plt.figure(figsize=(10, max(6, len(feature_importance_L1) * 0.25))) # Dynamic height based on feature count
sns.barplot(data=feature_importance_L1, x='Coefficient', y='Feature', palette='coolwarm')
plt.title(f'L1 Coefficients (C={best_C}, Original Training Set)', fontsize=14)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.show()


# Feature Importance Analysis: Top Positive & Negative Coefficients
# -------------------------------------------------------------------

## Sort by coefficient value (descending = strongest positive influence first)
feature_importance_sorted = feature_importance_L1.sort_values(by='Coefficient', ascending=False)

## Combined Table: Top 10 Positive + Top 10 Negative Features
top10_positive = feature_importance_sorted.head(10).copy()
top10_negative = feature_importance_sorted.tail(10).sort_values(by='Coefficient', ascending=True).copy()

## Add a new column indicating coefficient direction
top10_positive["Direction"] = "Positive"
top10_negative["Direction"] = "Negative"

## Combine both into one DataFrame
top_combined = pd.concat([top10_positive, top10_negative], axis=0).reset_index(drop=True)
print("\nğŸ“Š Combined Top 10 Positive and Top 10 Negative Feature Coefficients:")
print(top_combined.to_string(index=False))


# Print the number of data in the training and testing feature sets
print(f"Length of X_train_encoded, y_train and X_test_encoded: {len(X_train_encoded)}, {len(y_train)}, {len(X_test_encoded)}")


# Define the final L1-regularized logistic regression model using the best C value
final_L1 = LogisticRegression(penalty='l1', C=best_C, solver='liblinear', max_iter=1000, random_state=42)
# Fit (train) the model on the training data
final_L1.fit(X_train_encoded, np.ravel(y_train))

# Generate predicted probabilities for the test dataset for target variables
y_pred = final_L1.predict_proba(X_test_encoded)[:, 1]  # Optional: Use .predict

# Create DataFrame with IDs and predicted probabilities
y_test_pred = pd.DataFrame({
    'id': test_set['id'],
    'probability': y_pred})

# Add binary prediction column using lambda (based on the threshold (at what point a probability is a loan_paid_back)
y_test_pred['loan_paid_back'] = y_test_pred['probability'].apply(lambda x: 1 if x >= 0.5 else 0)
# If false: y_test_pred['loan_paid_back'] = (y_test_pred['probability'] >= 0.5).astype(int)

# Reorder columns to match desired layout
y_test_pred = y_test_pred[['id', 'loan_paid_back', 'probability']]
# Print the result
print(y_test_pred.head(10))

# Optionally save the predictions to a CSV file
# y_test_pred.to_csv("y_test_pred.csv", index=False)


# Print the row counts for each loan payment status and the percentage
print(y_test_pred['loan_paid_back'].value_counts(normalize=False))
print(y_test_pred['loan_paid_back'].value_counts(normalize=True))


# Predict on the training set
y_train_pred = final_L1.predict(X_train_encoded)

# Print the confusion matrix
print(confusion_matrix(y_train, y_train_pred))

# Visualize the matrix
cm = confusion_matrix(y_train, y_train_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='YlGnBu')
plt.title('Confusion Matrix - Logistic Regression (Threshold = 0.5)', 
          loc='center', fontweight='bold', fontsize=16)
plt.xlabel('Predicted Values', fontweight='bold', fontsize=12)
plt.ylabel('True Values', fontweight='bold', fontsize=12)
plt.show()


# Print classification report for training performance
print(classification_report(y_train, y_train_pred))


# Install necessary library
!pip install xgboost
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier


print('Processing... Please wait.')
# Setup: Define candidate reg_alpha values (analogous to C in Logistic Regression)
# -------------------------------------------------------------------
alpha_values = [0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 5, 10]
xgb_metrics = []  # Store [reg_alpha, ROC-AUC, Recall, Accuracy] for each value
use_cv = False  # Set True for 5-fold CV, False for training-set evaluation

## Stratified Cross-Validation setup
if use_cv:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop over each L1 regularization value
# -------------------------------------------------------------------
for alpha in alpha_values:
    ## Define XGBoost Classifier with L1 regularization
    xgb_model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.1,n_estimators=100,
        reg_alpha=alpha, # L1 regularization, if want to use L2 then reg_lambda
        use_label_encoder=False, eval_metric='logloss',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # Handle class imbalance
        random_state=42)

    if use_cv:
        ## Cross-validation metrics
        cv_auc = cross_val_score(xgb_model, X_train_encoded, y_train, cv=cv, scoring='roc_auc')
        mean_auc = np.mean(cv_auc)

        cv_recall = cross_val_score(xgb_model, X_train_encoded, y_train, cv=cv, scoring='recall')
        mean_recall = np.mean(cv_recall)

        cv_accuracy = cross_val_score(xgb_model, X_train_encoded, y_train, cv=cv, scoring='accuracy')
        mean_accuracy = np.mean(cv_accuracy)

        ## Fit on full data for feature importance
        xgb_model.fit(X_train_encoded, y_train)

    else:
        ## Fit model on full training data
        xgb_model.fit(X_train_encoded, y_train)

        ## Predict labels and probabilities
        y_pred = xgb_model.predict(X_train_encoded)
        y_pred_prob = xgb_model.predict_proba(X_train_encoded)[:, 1]

        ## Compute metrics
        mean_auc = roc_auc_score(y_train, y_pred_prob)
        mean_recall = recall_score(y_train, y_pred)
        mean_accuracy = accuracy_score(y_train, y_pred)

    ## Append results
    xgb_metrics.append([alpha, mean_auc, mean_recall, mean_accuracy])

## Notify completion
print('Run complete. Results are ready.')

# Summarize results and select best reg_alpha
# -------------------------------------------------------------------
xgb_results = pd.DataFrame(xgb_metrics, columns=['reg_alpha', 'Mean ROC-AUC', 'Recall', 'Accuracy'])
best_row = xgb_results.loc[xgb_results['Recall'].idxmax()]
best_alpha = best_row['reg_alpha']

## Display summary
print("\n XGBoost L1 (reg_alpha) Performance Summary:")
print(xgb_results.to_string(index=False))
print(f"\nğŸ�† Best reg_alpha Value: {best_alpha} (Recall = {best_row['Recall']:.4f}, "
      f"Mean ROC-AUC = {best_row['Mean ROC-AUC']:.4f}, Accuracy = {best_row['Accuracy']:.4f})")

# Train final XGBoost model on full training set
# -------------------------------------------------------------------
max_dept=5
final_xgb = xgb.XGBClassifier(max_depth=max_dept, learning_rate=0.1,n_estimators=100,
    reg_alpha=best_alpha, use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # Handle class imbalance
    random_state=42)
final_xgb.fit(X_train_encoded, y_train)
print(f"\nâœ… Final XGBoost Model trained successfully using reg_alpha={best_alpha}.")


import shap

# Initialize SHAP Explainer for Tree-based Models
explainer = shap.TreeExplainer(final_xgb)

# Compute SHAP values for full training data
print("\nğŸ”� Computing SHAP values for all training samples (this may take some time)...")
shap_values = explainer(X_train_encoded)

# Compute Mean Absolute SHAP Values for ranking feature importance
plt.rcParams['font.family'] = 'DejaVu Sans'  # or 'Arial', 'Helvetica'
shap_importance = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "Feature": X_train_encoded.columns,
    "Mean |SHAP value|": shap_importance
}).sort_values(by="Mean |SHAP value|", ascending=False)

# Bar Chart of Top Features
top_n = 15
top_features = shap_importance_df.head(top_n)
plt.figure(figsize=(12, max(6, len(top_features) * 0.4)))
sns.barplot(data=top_features, x="Mean |SHAP value|", y="Feature", palette="coolwarm")
plt.title(f" Mean Absolute SHAP Values (Top {top_n} Features, reg_alpha={best_alpha})", fontsize=14, fontweight="bold",  color='#1E3A8A', loc='center', pad=20)
plt.xlabel("Mean |SHAP value|", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()

# Beeswarm Plot (Directional Impact of Each Feature)
print("\nğŸ“Š Generating SHAP Beeswarm Plot...")
plt.rcParams['font.family'] = 'DejaVu Sans'  # or 'Arial', 'Helvetica'
plt.figure(figsize=(12, 6))
shap.summary_plot(
    shap_values.values,
    X_train_encoded,
    feature_names=X_train_encoded.columns,
    show=False,
    plot_size=(10, 6),
    color=plt.cm.coolwarm)
plt.title(f"SHAP Summary Plot â€“ Directional Impact (reg_alpha={best_alpha})", fontsize=14, fontweight="bold", color="#1E3A8A", loc="center", pad=20)
plt.xlabel("SHAP value (impact on model output)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()


final_xgb = xgb.XGBClassifier(max_depth=max_dept, learning_rate=0.1,n_estimators=100,
    reg_alpha=best_alpha, use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # Handle class imbalance
    random_state=42)
final_xgb.fit(X_train_encoded, np.ravel(y_train))

# Generate predicted probabilities for the test dataset for target variables
y_predxgb = final_xgb.predict_proba(X_test_encoded)[:, 1]  # Optional: Use .predict

# Create DataFrame with IDs and predicted probabilities
y_test_predxgb = pd.DataFrame({
    'id': test_set['id'],
    'probability': y_predxgb})

# Add binary prediction column using lambda (based on the threshold (at what point a probability is a loan_paid_back)
y_test_predxgb['loan_paid_back'] = y_test_predxgb['probability'].apply(lambda x: 1 if x >= 0.5 else 0)
# If false: y_test_pred['loan_paid_back'] = (y_test_pred['probability'] >= 0.5).astype(int)

# Reorder columns to match desired layout
y_test_predxgb = y_test_predxgb[['id', 'loan_paid_back', 'probability']]
# Print the result
print(y_test_predxgb.head(10))

# Optionally save the predictions to a CSV file
# y_test_predxgb.to_csv("y_test_predxgb.csv", index=False)


# Print the row counts for each loan payment status and the percentage
print(y_test_predxgb['loan_paid_back'].value_counts(normalize=False))
print(y_test_predxgb['loan_paid_back'].value_counts(normalize=True))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------------------------------------------
# Predict on the training set
y_train_predxgb = final_xgb.predict(X_train_encoded)

# Compute confusion matrix
cm_xgb = confusion_matrix(y_train, y_train_predxgb)

# Print numerical matrix
print(cm_xgb)

# -------------------------------------------------------------------
# Visualize Confusion Matrix
# -------------------------------------------------------------------
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb)
disp_xgb.plot(cmap='YlGnBu')

plt.title(f'Confusion Matrix - XGBoost (Î±={best_alpha}, Threshold = 0.5)',
          loc='center', fontweight='bold', fontsize=16)
plt.xlabel('Predicted Values', fontweight='bold', fontsize=12)
plt.ylabel('True Values', fontweight='bold', fontsize=12)
plt.show()


# Print classification report for training performance
print(classification_report(y_train, y_train_predxgb))


# Create side-by-side sub-plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Setting the threshold
thresh_lr  = np.quantile(y_test_pred['probability'], 0.95)
thresh_gbt = np.quantile(y_test_predxgb['probability'], 0.95)

# Logistic Regression (left)
ax1.hist(y_test_pred['probability'], bins=20,
         color='blue', edgecolor='black', alpha=0.75, label='Logistic Regression')
ax1.axvline(thresh_lr, color='red', linestyle='--', linewidth=2, label=f'95% thresh = {thresh_lr:.4f}')
ax1.set_title('Logistic Regression', fontsize=16, fontweight='bold')
ax1.set_xlabel('Predicted Probability of Loan Paid back', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11, loc='upper center')
ax1.grid(True, linestyle='--', alpha=0.4)

# Gradient Boosting Trees (right)
ax2.hist(y_test_predxgb['probability'], bins=20,
         color='lightcoral', edgecolor='black', alpha=0.75, label='Gradient Boosting')
ax2.axvline(thresh_gbt, color='red', linestyle='--', linewidth=2, label=f'95% thresh = {thresh_gbt:.4f}')
ax2.set_title('Gradient Boosting Tree', fontsize=16, fontweight='bold')
ax2.set_xlabel('Predicted Probability of Loan Paid back', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11, loc='upper center')
ax2.grid(True, linestyle='--', alpha=0.4)

# Global figure title
fig.suptitle('Comparison of Predicted Loan Repayment Probabilities: Logistic Regression vs. Gradient Boosting', fontsize=18, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()


# Merge Logistic Regression and XGBoost predictions by 'id'
# -------------------------------------------------------------------
#combined_predictions = (
#    y_test_pred.rename(
#        columns={
#            'probability': 'LogReg_probability', 
#            'loan_paid_back': 'LogReg_loan_paid_back'})
#    .merge(
#        y_test_predxgb.rename(
#            columns={
#                'probability': 'XGB_probability', 
#                'loan_paid_back': 'XGB_loan_paid_back'}),
#        on='id', how='inner'))


# Export combined table
# -------------------------------------------------------------------
#combined_predictions.to_csv("combined_predictions.csv", index=False)

#print("âœ… Combined predictions saved successfully as 'combined_predictions.csv'")
#display(combined_predictions.head())

# os.remove("/kaggle/working/combined_predictions.csv")


# Save result of XGB prediction
submission = y_test_predxgb[['id', 'loan_paid_back']]
submission.to_csv("submission.csv", index=False) 

