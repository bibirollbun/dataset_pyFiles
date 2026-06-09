# ------------------------------
# Standard Libraries
# ------------------------------
import math
import warnings
from datetime import timedelta
from functools import partial
from time import time
import numpy as np
import pandas as pd

# ------------------------------
# Data Visualization
# ------------------------------
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import seaborn as sns

# ------------------------------
# Scientific Computing & Statistics
# ------------------------------
import scipy as sp
from scipy.stats import ks_2samp, zscore

# ------------------------------
# Scikit-Learn Ecosystem
# ------------------------------
# Model Selection & Evaluation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Feature Selection & Preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# ------------------------------
# LightGBM
# ------------------------------
from lightgbm import LGBMClassifier, LGBMRegressor

# ------------------------------
# Utilities & Configuration
# ------------------------------
from colorama import Fore, Style

# Configure warnings
warnings.filterwarnings("ignore")



original_data = pd.read_csv('/kaggle/input/wine-quality-dataset/WineQT.csv')

column_order = original_data.columns.tolist()
print("Column Order in Original Data:", column_order)

# Reorder columns to place 'Id' first
new_column_order = ['Id'] + [col for col in column_order if col != 'Id']
original_data = original_data[new_column_order]

new_column_order_check = original_data.columns.tolist()
print("\nNew Column Order in Original Data:", new_column_order_check)


# Reload and set 'Id' as the index
original_data = pd.read_csv('/kaggle/input/wine-quality-dataset/WineQT.csv')
original_data = original_data[new_column_order] 
original_data.set_index('Id', inplace=True)

# Verify shapes after setting index
train_data = pd.read_csv('/kaggle/input/chydv-hackathon-2025/train.csv', index_col=[0])
test_data = pd.read_csv('/kaggle/input/chydv-hackathon-2025/test.csv', index_col=[0])
sample_data = pd.read_csv('/kaggle/input/chydv-hackathon-2025/sample_submission.csv')

print("Original Data Shape:", original_data.shape)
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)


# Display few rows of each dataset
print("Original Data Preview:")
display(original_data.tail())

print("\nTrain Data Preview:")
display(train_data.tail())

print("\nTest Data Preview:")
display(test_data.head())


# Display information for the original dataset
print("Original Dataset Information: \n")
original_info = original_data.info()
display(original_info)
print('\n')

# Display information for the training dataset
print("Training Dataset Information: \n")
train_info = train_data.info()
display(train_info)
print('\n')

# Display information for the test dataset
print("Test Dataset Information: \n")
test_info = test_data.info()
display(test_info)


# Descriptive statistics for numerical columns
print("\nOriginal Data Describe:")
display(original_data.describe().T.style.background_gradient(cmap='YlOrRd'))

print("\nTrain Data Describe:")
display(train_data.describe().T.style.background_gradient(cmap='YlOrRd'))

print("\nTest Data Describe:")
display(test_data.describe().T.style.background_gradient(cmap='YlOrRd'))



# Calculate and print Kolmogorov-Smirnov statistics
print("ğŸ”� Kolmogorov-Smirnov Test Results:")
print("-"*35)
print(f"{'Feature':<20} | {'KS-statistic':<12}")
print("-"*35)

for col in train_data.columns[:-1]:
    original = original_data[col]
    train = train_data[col]
    ks_stat = ks_2samp(original, train)[0]
    print(f"{col:<20} | {ks_stat:.3f}")

print("\n")



# Function to create a summary table for missing values and data types
def missing_values_summary(df):
    missing_count = df.isnull().sum()
    missing_percentage = 100 * missing_count / len(df)
    data_types = df.dtypes
    return pd.DataFrame({
        'Missing Values Count': missing_count,
        'Percentage (%)': missing_percentage,
        'Data Type': data_types
    })

original_summary = missing_values_summary(original_data)
train_summary = missing_values_summary(train_data)
test_summary = missing_values_summary(test_data)

print("Original Dataset Summary:")
display(original_summary)

print("\nTrain Dataset Summary:")
display(train_summary)

print("\nTest Dataset Summary:")
display(test_summary)


# Check for duplicated rows
print("Duplicate Rows in Original Data:", original_data.duplicated().sum())

print("\nDuplicate Rows in Train Data:", train_data.duplicated().sum())

print("\nDuplicate Rows in Test Data:", test_data.duplicated().sum())


# Display some duplicate rows
duplicate_rows = original_data[original_data.duplicated()]
display(duplicate_rows.head(10))



# Class distribution 
print("Class distribution in original dataset:")
print("-"*40)
print(original_data['quality'].value_counts())

# Class distribution in duplicate rows
print("\nClass distribution in duplicate rows:")
print("-"*40)
print(duplicate_rows['quality'].value_counts())

# Class distribution after removing duplicates
original_data = original_data.drop_duplicates()
print("\nClass distribution after removing duplicates:")
print(original_data['quality'].value_counts())



# duplicate_details = original_data[original_data.duplicated(keep=False)]
# display(duplicate_details.sort_values(by='quality').head(20))


# Set target variable
target_variable = 'quality'

train_data = train_data.sort_values(target_variable)
original_data = original_data.sort_values(target_variable)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
datasets = [('Train Data', train_data), ('Original Data', original_data)]

for i, (title, data) in enumerate(datasets):

    sns.countplot(x=data[target_variable], ax=axes[i, 0], palette='YlOrRd')
    axes[i, 0].set_title(f'Count Plot of Quality in {title}', pad=20)
    axes[i, 0].set_xlabel('Quality Score')
    axes[i, 0].set_ylabel('Count')
    axes[i, 0].set_facecolor("lightgray")
    axes[i, 0].grid(axis='y', color='gray', linestyle='--', linewidth=0.7)
    axes[i, 0].set_xticklabels(axes[i, 0].get_xticklabels(), rotation=45)

    quality_counts = data[target_variable].value_counts().sort_index()
    wedges, texts, autotexts = axes[i, 1].pie(
        quality_counts,
        labels=quality_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("YlOrRd", len(quality_counts)),
        wedgeprops=dict(width=0.4, edgecolor='w'),
        radius=1.2
    )
    
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    axes[i, 1].add_artist(centre_circle)
    
    axes[i, 1].set_title(f'Quality Distribution in {title}', pad=25)
    axes[i, 1].set_facecolor("lightgray")
    axes[i, 1].axis('equal')

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.show()



def plot_numerical_features(train_data, test_data, original_data, numerical_features):

    colors = sns.color_palette('YlOrRd', 3)

    fig, axes = plt.subplots(len(numerical_features), 2, figsize=(12, len(numerical_features) * 4))

    for i, feature in enumerate(numerical_features):
        # Histogram
        sns.histplot(train_data[feature], color=colors[0], label='Train Data', bins=20, kde=True, ax=axes[i, 0])
        sns.histplot(test_data[feature], color=colors[1], label='Test Data', bins=20, kde=True, ax=axes[i, 0])
        sns.histplot(original_data[feature], color=colors[2], label='Original Data', bins=20, kde=True, ax=axes[i, 0])

        axes[i, 0].set_title(f'Histogram of {feature}')
        axes[i, 0].legend()
        axes[i, 0].set_facecolor("lightgray")
        axes[i, 0].grid(color='gray', linestyle='--', linewidth=0.7)

        # Horizontal Boxplot
        sns.boxplot(data=[train_data[feature], test_data[feature], original_data[feature]], 
                    palette=colors, orient='h', ax=axes[i, 1])

        axes[i, 1].set_title(f'Horizontal Boxplot of {feature}')
        axes[i, 1].set_yticklabels(['Train Data', 'Test Data', 'Original Data'])
        axes[i, 1].set_facecolor("lightgray")
        axes[i, 1].grid(axis='x', color='gray', linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.show()

# Define the numerical features
numerical_features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 
    'residual sugar', 'chlorides', 'free sulfur dioxide', 
    'total sulfur dioxide', 'density', 'pH', 
    'sulphates', 'alcohol'
]

plot_numerical_features(train_data, test_data, original_data, numerical_features)



def visualize_feature(train_data, original_data, feature, target_variable):

    fig = plt.figure(figsize=(12, 10))
    spec = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1.2, 1])

    # Density Plot (Spans full width in first row)
    ax1 = fig.add_subplot(spec[0, :])  
    sns.kdeplot(data=train_data, x=feature, hue=target_variable, fill=True, palette='YlOrRd', ax=ax1)
    ax1.set_title(f'Density Plot of {feature} by {target_variable} (Train Data)')
    ax1.set_facecolor("lightgray")
    ax1.grid(color='gray', linestyle='--', linewidth=0.7)

    # Box Plot (Bottom-left)
    ax2 = fig.add_subplot(spec[1, 0])  
    sns.boxplot(x=train_data[target_variable], y=train_data[feature], palette='YlOrRd', ax=ax2)
    ax2.set_title(f'Box Plot of {feature} by {target_variable} (Train Data)')
    ax2.set_xlabel(target_variable)
    ax2.set_ylabel(feature)
    ax2.set_facecolor("lightgray")
    ax2.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)

    # Strip Plot (Bottom-right)
    ax3 = fig.add_subplot(spec[1, 1])  
    sns.stripplot(x=original_data[target_variable], y=original_data[feature], jitter=True, palette='YlOrRd', ax=ax3)
    ax3.set_title(f'Strip Plot of {feature} by {target_variable} (Original Data)')
    ax3.set_xlabel(target_variable)
    ax3.set_ylabel(feature)
    ax3.set_facecolor("lightgray")
    ax3.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.show()



visualize_feature(train_data, original_data, feature='fixed acidity', target_variable='quality')


visualize_feature(train_data, original_data, feature='volatile acidity', target_variable='quality')


visualize_feature(train_data, original_data, feature='citric acid', target_variable='quality')


visualize_feature(train_data, original_data, feature='residual sugar', target_variable='quality')


visualize_feature(train_data, original_data, feature='chlorides', target_variable='quality')


visualize_feature(train_data, original_data, feature='free sulfur dioxide', target_variable='quality')


visualize_feature(train_data, original_data, feature='total sulfur dioxide', target_variable='quality')


visualize_feature(train_data, original_data, feature='density', target_variable='quality')


visualize_feature(train_data, original_data, feature='pH', target_variable='quality')


visualize_feature(train_data, original_data, feature='sulphates', target_variable='quality')


visualize_feature(train_data, original_data, feature='alcohol', target_variable='quality')


def plot_acidity_pairplot(data, target_variable):
    acidity_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'pH', target_variable]
    pair_plot = sns.pairplot(data[acidity_features], hue=target_variable, palette="YlOrRd", diag_kind="kde")
    plt.suptitle("Acidity Features vs. Quality", y=1.02, fontsize=14)
    plt.show()

plot_acidity_pairplot(train_data, target_variable='quality')


def plot_sugar_alcohol_pairplot(data, target_variable):
    features = ['residual sugar', 'alcohol', 'density', target_variable]
    pair_plot = sns.pairplot(
        data[features], hue=target_variable, palette="YlOrRd", diag_kind="kde"
    )
    plt.subplots_adjust(top=0.95)
    pair_plot.fig.suptitle("Residual Sugar, Alcohol & Density vs. Wine Quality", fontsize=12)
    plt.show()

plot_sugar_alcohol_pairplot(train_data, target_variable='quality')



def plot_quality_vs_features(data, target_variable):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    quality_colors = data[target_variable].map(lambda x: plt.cm.YlOrRd(x / data[target_variable].max()))

    sns.regplot(x=target_variable, y="alcohol", data=data, ax=axes[0], scatter=False, line_kws={'color': 'red'})
    axes[0].scatter(data[target_variable], data["alcohol"], c=quality_colors, s=10)
    axes[0].set_xlabel("Quality")
    axes[0].set_ylabel("Alcohol")
    axes[0].set_title("Quality vs Alcohol")
    axes[0].grid(axis="y", color="gray", linestyle="--", linewidth=0.7)

    sns.regplot(x=target_variable, y="density", data=data, ax=axes[1], scatter=False, line_kws={'color': 'red'})
    axes[1].scatter(data[target_variable], data["density"], c=quality_colors, s=10)
    axes[1].set_xlabel("Quality")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Quality vs Density")
    axes[1].grid(axis="y", color="gray", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    plt.show()

plot_quality_vs_features(train_data, target_variable="quality")



# Define a function to create correlation heatmap
def plot_correlation_heatmap(data, title, ax):
    corr_matrix = data.corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='YlOrRd', ax=ax, square=True, cbar_kws={"shrink": .8})
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

plot_correlation_heatmap(original_data, 'Correlation Heatmap - Original Data', axes[0])
plot_correlation_heatmap(train_data, 'Correlation Heatmap - Train Data', axes[1])
plot_correlation_heatmap(test_data, 'Correlation Heatmap - Test Data', axes[2])
plt.tight_layout()
plt.show()



train_data.columns = train_data.columns.str.lower().str.replace(" ", "_")
original_data.columns = original_data.columns.str.lower().str.replace(" ", "_")
test_data.columns = test_data.columns.str.lower().str.replace(" ", "_")

print(train_data.columns) 


# Function to calculate and print skewness for numerical features
def check_skewness(data, dataset_name):
    print(f"ğŸ”� Skewness for {dataset_name}:")
    print("-"*35)
    print(f"{'Feature':<20} | {'Skewness':<10}")
    print("-"*35)
    
    for feature in data.select_dtypes(include=[np.number]).columns:
        skewness = data[feature].skew()
        print(f"{feature:<20} | {skewness:.4f}")
    
    print("\n")

check_skewness(original_data, "Original Data")
check_skewness(train_data, "Train Data")
check_skewness(test_data, "Test Data")



features = train_data.columns[:-1]  # Exclude the target variable

# Compute Z-Scores for features grouped by 'quality'
z_scores = train_data.groupby("quality")[features].transform(zscore)

# Identify outliers (Z-score threshold â‰¥ 2)
outliers = (z_scores.abs() >= 2).groupby(train_data["quality"]).sum()


outlier_summary = pd.DataFrame({
    'Total Outliers': outliers.sum(axis=1),
    'Features with Most Outliers': outliers.idxmax(axis=1),
    'Max Outliers per Feature': outliers.max(axis=1)
})

detailed_breakdown = pd.melt(outliers.reset_index(), 
                           id_vars='quality', 
                           var_name='feature',
                           value_name='outlier_count')

total_samples = len(train_data)
quality_counts = train_data['quality'].value_counts().to_dict()

detailed_breakdown['percentage_of_quality'] = detailed_breakdown.apply(
    lambda x: f"{(x['outlier_count']/quality_counts[x['quality']]*100):.1f}%", 
    axis=1
)

detailed_breakdown['percentage_of_total'] = detailed_breakdown['outlier_count'].apply(
    lambda x: f"{(x/total_samples*100):.2f}%"
)

print("="*55)
print(f"ğŸ“Š Global Outlier Summary (Z â‰¥ 2)")
print("="*55)
print(f"Total outlier instances: {outliers.sum().sum():,}")
print(f"Features with most outliers: {outliers.sum().idxmax()} ({outliers.sum().max():,})")
print(f"Quality category with most outliers: {outlier_summary['Total Outliers'].idxmax()}")
print("\n")

print("="*55)
print("ğŸ”� Per-Feature & Quality Outlier Breakdown")
print("="*55)
for feature in features:
    feature_outliers = detailed_breakdown[detailed_breakdown['feature'] == feature]
    total = feature_outliers['outlier_count'].sum()
    
    print(f"\nğŸ“Œ {feature.upper()}")
    print(f"Total outliers: {total:,} ({total/total_samples*100:.1f}% of dataset)")
    print("-"*50)
    print(f"{'Quality':<8} | {'Count':<10} | {'% of Quality':<12} | {'% of Total':<10}")
    print("-"*50)
    
    for _, row in feature_outliers.iterrows():
        print(f"{row['quality']:<8} | {row['outlier_count']:<10,} | {row['percentage_of_quality']:<12} | {row['percentage_of_total']:<10}")



n_cols = 4
n_rows = math.ceil(len(features) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten()
palette = sns.color_palette("YlOrRd", as_cmap=True)

for i, feature in enumerate(features):
    sns.barplot(x=outliers.index, y=outliers[feature], ax=axes[i], palette="YlOrRd")
    axes[i].set_ylabel('Count', fontsize=12)
    axes[i].set_xlabel('Quality', fontsize=12)
    axes[i].set_title(f"Outliers in {feature}", fontsize=12)
    axes[i].set_facecolor("lightgray")
    axes[i].grid(axis="y", linestyle="--", alpha=0.7)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


plt.suptitle("Outliers per Quality Category (Train Data)", fontsize=14, y=1.02)


plt.tight_layout()
plt.show()


def analyze_mi_scores(data, dataset_name):
    X = data.drop(columns=['quality'])
    y = data['quality'].astype('category').cat.codes
    
    # Calculate MI scores
    mi = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    print(f"ğŸ”� Mutual Information Scores ({dataset_name}):")
    print("-"*45)
    print(f"{'Feature':<22} | {'MI Score':<10}")
    print("-"*45)
    for feature, score in mi_scores.items():
        print(f"{feature:<22} | {score:.4f}")
    print("\n")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=mi_scores.index, y=mi_scores.values, palette="YlOrRd_r")
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('MI Score', fontsize=12)
    plt.title(f'Feature Importance: {dataset_name}', fontsize=14)
    ax.set_facecolor("lightgray")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

analyze_mi_scores(train_data, "Training Data")
analyze_mi_scores(original_data, "Original Data")



# Feature Engineering function
def FE_density(X):
    X = X.copy()  
    
    new_features = {
        'alcohol*density': X['alcohol'] * X['density'],
        'alcohol/density': X['alcohol'] / (X['density'] + 1e-6),
        'acid/density': (X['fixed_acidity'] + X['volatile_acidity'] + X['citric_acid']) / (X['density'] + 1e-6),
        'density/residual_sugar': X['density'] / (X['residual_sugar'] + 1e-6),

    }
    
    for col, values in new_features.items():
        if col not in X.columns:
            X[col] = values
    
    return X


# Apply Feature Engineering to train data
X = train_data.copy()
X = FE_density(X)  # Apply feature engineering
new_cols = X.columns.difference(train_data.columns)

# Plot feature distributions
columns = new_cols
n_cols = 4
n_rows = math.ceil(len(columns) / n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
ax = ax.flatten()

for i, column in enumerate(columns):
    sns.boxplot(y=X[column], x=X.quality, ax=ax[i], palette="YlOrRd")
    ax[i].set_title(f'{column} Distribution')
    ax[i].set_xlabel(None)
    ax[i].set_facecolor("lightgray")
    ax[i].grid(axis="y", linestyle="--", alpha=0.7)

# Hide unused subplots
for j in range(i + 1, len(ax)):
    ax[j].axis('off')

plt.tight_layout()
plt.show()


# Features and target variable
X = train_data.copy().drop(columns=['quality'])  
X = FE_density(X) 
y_target = train_data['quality'].astype('category').cat.codes 

# Compute Mutual Information
mi_scores = mutual_info_classif(X, y_target, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Print MI Scores
print("ğŸ”� Mutual Information Scores:")
print("-"*35)
print(f"{'Feature':<20} | {'MI Score':<10}")
print("-"*35)
for feature, score in mi_scores.items():
    print(f"{feature:<20} | {score:.4f}")
print("\n")


# Cross-validation setup
cv = StratifiedKFold(5, shuffle=True, random_state=42)
X = train_data.drop(columns=["quality"])  # Features
y = train_data["quality"]  # Target

kappas = []
models = []
oof_preds = pd.Series(0, index=train_data.index)
start = time()

for fold, (tr_ix, vl_ix) in enumerate(cv.split(train_data, y)):
    start_fold = time()
    
    X_tr, y_tr = X.loc[tr_ix].copy(), y.loc[tr_ix]
    X_vl, y_vl = X.loc[vl_ix].copy(), y.loc[vl_ix]
    
    # Concatenate original df for training (avoiding target column)
    X_tr = pd.concat([X_tr, original_data.drop(columns=["quality"])])
    y_tr = pd.concat([y_tr, original_data["quality"]])

    # Apply feature engineering
    X_tr = FE_density(X_tr)
    X_vl = FE_density(X_vl)

    # Train LightGBM model
    model = LGBMClassifier(
        max_depth=4,
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=20,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    model.fit(X_tr, y_tr)

    # Make predictions
    y_pred = model.predict(X_vl)
    oof_preds.iloc[vl_ix] = y_pred
    kappas.append(cohen_kappa_score(y_vl, y_pred, weights="quadratic"))
    models.append(model)

    print("-"*30)
    print(f'Fold: {fold} - {timedelta(seconds=int(time() - start))}')
    print(f'Quadratic Kappa: {Fore.BLUE}{kappas[-1]:.4f}{Style.RESET_ALL}')
    print(f'Train Time taken: {timedelta(seconds=int(time() - start_fold))}')
    print()

# Output Mean Quadratic Kappa
print(f'Mean Quadratic Kappa: {Fore.BLUE}{np.mean(kappas):.4f}{Style.RESET_ALL}')


# Confusion Matrix Visualization
fig, ax = plt.subplots(figsize=(8, 6), facecolor="lightgray")

sns.heatmap(
    confusion_matrix(train_data.quality, oof_preds),
    annot=True,
    cmap='YlOrRd',
    fmt='',
    ax=ax
);

qualities = np.sort(train_data.quality.unique())
ax.set_xticklabels(qualities)
ax.set_yticklabels(qualities)
ax.set_ylabel('True Label')
ax.set_xlabel('Prediction Label')
ax.set_title('Confusion Matrix of OOF Predictions')
plt.tight_layout()
plt.show()


# Quadratic Weighted Kappa function
qwk = partial(cohen_kappa_score, weights='quadratic')

class LGBMRegressorWithRounder(LGBMRegressor):

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 3
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 4
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 5
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 6
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 7
            else:
                X_p[i] = 8

        ll = qwk(y, X_p)
        return -ll
    
    def fit(self, X, y, **params):
        super().fit(X, y, **params)
        X_pred = super().predict(X)
        loss_partial = partial(self._kappa_loss, X=X_pred, y=y)
        initial_coef = list(np.array([3.5, 4.5, 5.5, 6.5, 7.5]))
        self.round_coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        return self
    
    def set_params(self, **params):
        self.round_coef_ = None
        
    def predict_discrete(self, X):
        coef = self.coefficients()
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 3
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 4
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 5
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 6
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 7
            else:
                X_p[i] = 8
        return X_p.astype('int')

    def coefficients(self):
        return self.round_coef_['x']
    
    def predict(self, X):
        X_pred = super().predict(X)
        return self.predict_discrete(X_pred)



# Cross-validation setup
cv = StratifiedKFold(5, shuffle=True, random_state=42)
X = train_data[features]  # Features
y = train_data.quality  # Target

kappas = []
test_preds = []
models = []
oof_preds = pd.Series(0, index=train_data.index)
start = time()

# Cross-validation loop
for fold, (tr_ix, vl_ix) in enumerate(cv.split(train_data, train_data.quality)):
    start_fold = time()
    X_tr, y_tr = X.loc[tr_ix], y.loc[tr_ix]
    X_vl, y_vl = X.loc[vl_ix], y.loc[vl_ix]
    
    # Concatenate original df
    X_tr = pd.concat([X_tr, original_data[features]])
    y_tr = pd.concat([y_tr, original_data.quality])
    
    X_tr = FE_density(X_tr)
    X_vl = FE_density(X_vl)
    
    # Initialize model
    model = LGBMRegressorWithRounder(max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    
    # Predictions and evaluation
    y_pred = model.predict(X_vl)
    oof_preds.iloc[vl_ix] = y_pred
    kappas.append(cohen_kappa_score(y_vl, y_pred, weights='quadratic'))
    models.append(model)
    
    print("-"*30)
    print(f'Fold: {fold} - {timedelta(seconds=int(time() - start))}')
    print(f'Quadratic Kappa: {Fore.BLUE}{kappas[-1]:.4f}{Style.RESET_ALL}')
    print(f'Train Time taken: {timedelta(seconds=int(time() - start_fold))}')
    print()

# Final output
print(f'Mean Quadratic Kappa: {Fore.BLUE}{np.mean(kappas):.4f}{Style.RESET_ALL}')


# Confusion Matrix Visualization
fig, ax = plt.subplots(figsize=(8, 6), facecolor="lightgray")

sns.heatmap(
    confusion_matrix(train_data.quality, oof_preds),
    annot=True,
    cmap='YlOrRd',
    fmt='',
    ax=ax
);

qualities = np.sort(train_data.quality.unique())
ax.set_xticklabels(qualities)
ax.set_yticklabels(qualities)
ax.set_ylabel('True Label')
ax.set_xlabel('Prediction Label')
ax.set_title('Confusion Matrix of OOF Predictions')
plt.tight_layout()
plt.show()


test_preds = {i: model.predict(FE_density(test_data)) for i, model in enumerate(models)}

test_preds = pd.DataFrame(test_preds)

test_preds.set_index(test_data.index, inplace=True)


test_preds = test_preds.mode(axis=1)[0].astype(int)
test_preds.rename('quality', inplace=True)


test_preds.to_csv('submission.csv')
print("Submission file 'submission.csv' has been successfully saved.")


print(test_preds.head(10))


# Descriptive statistics of the test_preds
print("Descriptive statistics of test_preds:")
print(test_preds.describe())

