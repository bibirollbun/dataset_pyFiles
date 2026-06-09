import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

from sklearn.linear_model import LogisticRegression, RANSACRegressor
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import scipy.stats as stats

import warnings
warnings.filterwarnings('ignore')

test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv", index_col='id')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')



def plot_hexbin(df,features,target):
    # Create hexbin plot
    plt.figure(figsize=(10, 6))
    plt.hexbin(df[features], df[target], gridsize=30, cmap='coolwarm', mincnt=1)
    plt.colorbar(label='Counts')

    plt.xlabel(f"{features}")
    plt.ylabel(f"{TARGET}")
    plt.title(f"Hexbin Plot of {features} vs {TARGET}")
    plt.show()



df.head()


print(df.columns)


print(df.dtypes)


CAT = df.select_dtypes(include=['object']).columns.tolist()
print(CAT)

TARGET = 'loan_paid_back'

NUM = df.select_dtypes(exclude=['object']).columns.tolist()
NUM = [col for col in NUM if col != TARGET]
print(NUM)


# Show uniques values for Field
for feature in CAT:
    unique_values = df[feature].unique()
    print(f'Unique Values in {feature}: ')
    print(f'{unique_values} \n')


encoder = TargetEncoder(cols=CAT)
df[CAT] = encoder.fit_transform(df[CAT], df[TARGET])


features = df.drop(columns=[TARGET]) 
target = df[TARGET]

# Standardize only the features
scaler = StandardScaler()
standardized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Combine the standardized features with the target variable
standardized_df = pd.concat([standardized_features, target.reset_index(drop=True)], axis=1)


def plot_joint_relationships(dataframe, features_array, target):
    """
    Generates a grid of joint plots for each feature against the target variable.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    features_array (list): A list of feature names (independent variables).
    target (str): The name of the target variable.
    """
    num_features = len(features_array)
    num_cols = 3  # Number of columns for the plots
    num_rows = (num_features * 3 + num_cols - 1) // num_cols  # Calculate the number of rows needed

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Loop through the features and create joint plots
    for i, feature in enumerate(features_array):
        # Joint plot with hexbin
        sns.histplot(data=dataframe, x=feature, y=target, ax=axes[i * 3], cmap='coolwarm', cbar=True)
        axes[i * 3].set_title(f'Hexbin: {feature} vs {target}')
        
        # Joint plot with regression
        sns.regplot(data=dataframe, x=feature, y=target, ax=axes[i * 3 + 1])
        axes[i * 3 + 1].set_title(f'Regression')
        
        # Joint plot with residuals
        sns.residplot(data=dataframe, x=feature, y=target, ax=axes[i * 3 + 2])
        axes[i * 3 + 2].set_title(f'Residuals')

    # Hide any unused subplots
    for j in range(i * 3 + 3, num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout() # adjusts the spacing between subplots to minimize overlaps
    plt.savefig('joint_plots_grid.png', dpi=300, bbox_inches='tight')  
    
    plt.show()


draw_relations = True
if draw_relations == True:
    plot_joint_relationships(standardized_df, np.append(CAT, NUM), TARGET)


def plot_target_correlations(dataframe, target_col):
    # Calculate correlations with the TARGET column
    correlations = dataframe.corr()[target_col].sort_values(ascending=False)

    # Create horizontal bar chart
    plt.figure(figsize=(10, 10)) # width, height
    correlations.drop(target_col).plot(kind='barh', color='skyblue')
    
    # Set titles and labels
    plt.title(f'Correlations of {target_col} with the rest of the variables', fontsize=16)
    plt.xlabel('Correlation', fontsize=14)
    plt.ylabel('Variables', fontsize=14)
    
    # Show the plot
    plt.show()


def plot_correlation_matrix(dataframe):
    # Calculate the correlation matrix
    correlation_matrix = dataframe.corr()

    # Set up the figure size
    plt.figure(figsize=(12, 10))
    
    # Draw the heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, cbar_kws={"shrink": .8}, vmin=-1, vmax=1)

    # Set titles and labels
    plt.title('Correlation Matrix Heatmap', fontsize=16)
    plt.show()


plot_correlation_matrix(standardized_df)


def calculate_vif(dataframe):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = dataframe.columns
    vif_data["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    vif_data["Tolerance"] = 1 / vif_data["VIF"]
    return vif_data


vif_data=calculate_vif(standardized_df)
vif_data


def plot_vif(vif_data):
    # Create a grid layout with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Horizontal bar chart for VIF
    axes[0].barh(vif_data["Variable"], vif_data["VIF"], color='skyblue')
    axes[0].axvline(x=10, color='red', linestyle='--', label='Critical VIF Threshold (10)')
    axes[0].set_title('Variance Inflation Factor (VIF)')
    axes[0].set_xlabel('VIF Value')
    axes[0].legend()
    
    # Horizontal bar chart for Tolerance
    axes[1].barh(vif_data["Variable"], vif_data["Tolerance"], color='lightgreen')
    axes[1].axvline(x=0.1, color='red', linestyle='--', label='Critical Tolerance Threshold (0.1)')
    axes[1].set_title('Tolerance')
    axes[1].set_xlabel('Tolerance Value')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


plot_vif(vif_data)


FEATURES_1  = [col for col in np.append(CAT,NUM) if col != 'credit_score' ]


def remove_outliers_ransac(df,features,target, n_trials=10): # si no se expecifica valor de iteracciones ejecuta 10 por defecto
    df['inlier'] = False
    for i in range(n_trials):
        base_model = LogisticRegression()
        model = RANSACRegressor(base_model,random_state=42 + i, min_samples=0.5) #min 50% must be inliers)
        model.fit(df[features], df[target].astype(int)) #Int because is logisticRegresion
        inlier_mask = model.inlier_mask_

        inlier_mask = model.inlier_mask_

        df.loc[inlier_mask, 'inlier'] = True

    return df 


# Run outlier removal multiple times
n_trials = 1
inliers_df = remove_outliers_ransac(standardized_df,np.append(CAT,NUM),TARGET, n_trials=n_trials)


outlier_per= 1 - (inliers_df[inliers_df['inlier']==True].shape[0] / standardized_df.shape[0])
print(f"Volume of outliers = {outlier_per*100:.2f}%")


def plot_outliers(df, features, target):
    # Create a color mapping based on the inlier status
    df['color'] = df['inlier'].map({True: 'blue', False: 'gray'})

    # Set up the grid for plots (2 columns, up to 5 rows)
    num_plots = min(len(features), 10)  # Handle at most 10 features
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 25))
    
    # Flatten the axes array for easy indexing
    axs = axs.flatten()

    for i in range(num_plots):
        axs[i].scatter(df[features[i]], df[target], c=df['color'], alpha=0.6)
        axs[i].set_title(f'Scatter Plot of {features[i]} vs {target}')
        axs[i].set_xlabel(features[i])
        axs[i].set_ylabel(target)
        axs[i].grid()

        # Create a custom legend for each plot
        blue_patch = plt.Line2D([], [], marker='o', color='w', label='Inlier', markerfacecolor='blue', markersize=10)
        gray_patch = plt.Line2D([], [], marker='o', color='w', label='Outlier', markerfacecolor='gray', markersize=10)
        axs[i].legend(handles=[blue_patch, gray_patch], loc='upper right')

    # Hide any unused subplots
    for j in range(num_plots, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


draw_outlier = True
if draw_outlier == True:
    plot_outliers(inliers_df, FEATURES_1, TARGET)


plot_target_correlations(inliers_df[np.append(FEATURES_1,TARGET)],TARGET)



X = inliers_df[inliers_df['inlier']==True][FEATURES_1]
y = inliers_df[inliers_df['inlier']==True][TARGET].astype(int)  # AsegÃºrate de que TARGET sea binaria


# Add a constant for the model
X = sm.add_constant(X)

# Fit the logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Get p-values
# p_values = result.pvalues
# print("P-values:\n", p_values)

# Optional: Print the model summary
print(result.summary())


def plot_t_distribution_with_pvalue(lineal_model, feature_index, num_observations):
    t_statistic = lineal_model.tvalues[feature_index]  # t-statistic for the slope
    p_value_statistic = lineal_model.pvalues[feature_index]  # p-value for the slope

    feature_names = lineal_model.params.index.tolist()  # Convert the index to a list
    feature_name = feature_names[feature_index]  # Get the feature name

    # Create a range of t values for the plot
    t_value_range = np.linspace(-4, 4, 100)
    t_probability_density = stats.t.pdf(t_value_range, df=num_observations - 2)  # df = n - k

    # Determine significance
    hypothesis_significance = "Reject H0: p-value < 0.05 (significant)" if p_value_statistic < 0.05 else "Fail to Reject H0: p-value >= 0.05 (not significant)"

    # Plot the t distribution
    plt.figure(figsize=(12, 6))

    # t distribution
    plt.subplot(1, 2, 1)
    plt.plot(t_value_range, t_probability_density, label='t-distribution (df={})'.format(num_observations - 2))
    plt.axvline(t_statistic, color='red', linestyle='--', label='t-statistic = {:.2f}'.format(t_statistic))
    plt.fill_between(t_value_range, t_probability_density, where=(t_value_range > t_statistic), alpha=0.5, color='lightcoral', label='p-value area')
    plt.title('T-Distribution with p-value for {}'.format(feature_name))
    plt.xlabel('t-value')
    plt.ylabel('Probability Density')
    plt.legend()

    # Plot the p-value
    plt.subplot(1, 2, 2)
    plt.bar(['p-value'], [p_value_statistic], color='skyblue')
    plt.title('P-value for {} from Regression\n{}'.format(feature_name, hypothesis_significance))
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()


n=X.shape[0]
plot_t_distribution_with_pvalue(result,2,n)
plot_t_distribution_with_pvalue(result,7,n)


#Significant features for the model p < 0.05
FEATURES_2 = [
    'gender',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade',
    'annual_income',
    'debt_to_income_ratio',
    'interest_rate'
]


X_2 = inliers_df[inliers_df['inlier']==True][FEATURES_2]
y_2 = inliers_df[inliers_df['inlier']==True][TARGET].astype(int)  # AsegÃºrate de que TARGET sea binaria

model_2 = LogisticRegression(max_iter=10000, C=1, class_weight={0: 1, 1: 1}, penalty='l2', solver='newton-cg')
model_2.fit(X_2, y_2)


def plot_roc_and_confusion_matrix(model, X_test, y_test):
    # Predict probabilities
    y_scores = model.predict_proba(X_test.values)[:, 1]

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Calculate the confusion matrix
    y_pred = (y_scores >= 0.5).astype(int)  # Threshold of 0.5 for classification
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate cross-validation for ROC AUC
    cv_auc_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
    mean_cv_auc = np.mean(cv_auc_scores)

    # Plot ROC curve and confusion matrix
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # ROC curve
    axs[0].plot(fpr, tpr, color='blue', label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
    axs[0].plot([0, 1], [0, 1], color='red', linestyle='--')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.0])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axs[0].legend(loc='lower right')

    # Confusion matrix
    cmd = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    cmd.plot(ax=axs[1], cmap='Blues', values_format='d')
    axs[1].set_title('Confusion Matrix')

    plt.tight_layout()
    plt.show()

    # Print cross-validation results
    print("ROC AUC scores from cross-validation (CV=5):", cv_auc_scores)
    print("Mean ROC AUC from cross-validation:", mean_cv_auc)


plot_roc_and_confusion_matrix(model_2, X_2, y_2)


# Transform the new data using the fitted encoder
test[CAT] = encoder.transform(test[CAT])

features = df.drop(columns=[TARGET]) 
# Reorder the columns in 'test' to match 'features'
test = test[features.columns]

# Standardize the new dataset
standardized_test = pd.DataFrame(scaler.transform(test), columns=test.columns)


print(test.shape)
print(standardized_test.shape)


X_submit = standardized_test[FEATURES_2]
#y_pred_submit = model_2.predict(X_submit) #--> Return 0 or 1
y_pred_submit = model_2.predict_proba(X_submit) #--> Return a probability between 0 and 1

# y_pred_submit will have two columns, since I use predict_proba: 
# - The first column is the probability of class 0
# - The second column is the probability of class 1

# Get only the probabilities of the positive class
df_sub[TARGET] = y_pred_submit[:, 1]
df_sub.to_csv('test_logistic_2.csv', index=False)
df_sub.head()

