# === IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Configure visual settings
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.titlepad'] = 15
%matplotlib inline


# === LOAD AND PREPARE DATA ===
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]

# Clean column names
iris_df.columns = [col.replace(' (cm)', '').title() for col in iris_df.columns]

# Display data sample
print(f"Dataset shape: {iris_df.shape}")
iris_df.sample(5, random_state=42)


# === ENHANCED PAIR PLOT ===
g = sns.pairplot(iris_df, 
                 hue='Species',
                 palette='viridis',
                 height=1.8,
                 plot_kws={'s': 30, 'alpha': 0.8, 'edgecolor': 'k', 'linewidth': 0.5},
                 diag_kind='kde',
                 corner=True)

g.fig.suptitle('Pairwise Feature Relationships', y=1.02, fontsize=18)
plt.tight_layout()
plt.show()


# Load dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]

# Display first 5 rows
iris_df.head()


# Group by species and show statistics
iris_df.groupby('species').describe().transpose()


# Create pair plot
import warnings
with warnings.catch_warnings():
    # Suppress multiple warning types
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)  # Broaden suppression
    
    # Create pair plot
    g = sns.pairplot(iris_df, hue='species', palette='viridis', height=2)
    
    # Adjust title using the grid object instead of plt.suptitle
    g.fig.suptitle('Feature Relationships by Species', y=1.02)
    
    # Remove explicit tight_layout call
    # plt.tight_layout()  # This is redundant and causes warning
    plt.show()


# === BOXPLOTS WITH SWARM PLOTS ===
plt.figure(figsize=(13, 9))
features = iris_df.columns[:-1]

for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    
    # Create boxplot
    ax = sns.boxplot(x='Species', y=feature, data=iris_df, width=0.5, 
                    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'none'})
    
    # Add data points
    sns.swarmplot(x='Species', y=feature, data=iris_df, color='.2', size=3.5, alpha=0.7)
    
    # Add mean markers
    species_means = iris_df.groupby('Species')[feature].mean()
    for j, mean_val in enumerate(species_means):
        plt.scatter(j, mean_val, marker='D', s=80, color='red', 
                   edgecolor='k', zorder=5, label='Mean' if j==0 else "")
    
    # Formatting
    plt.title(f'{feature} Distribution', fontsize=14, fontweight='bold')
    plt.ylabel(f'{feature} (cm)', labelpad=10)
    plt.xlabel('')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if i == 0:
        plt.legend(loc='upper right')

plt.suptitle('Feature Distribution by Species', y=0.99, fontsize=18)
plt.tight_layout(pad=2.0)
plt.show()


# Create feature distribution plots with warning suppression
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features = iris_df.columns[:-1]  # Exclude species column

    for i, feature in enumerate(features):
        row, col = i//2, i%2
        
        # Box plot
        sns.boxplot(x='species', y=feature, data=iris_df, ax=axes[row, col], palette='viridis')
        
        # Distribution plot
        ax2 = axes[row, col].twinx()
        sns.kdeplot(data=iris_df, x=feature, hue='species', palette='viridis', 
                    fill=True, alpha=0.3, ax=ax2)
        
        axes[row, col].set_title(f'{feature.capitalize()} Distribution', fontsize=12)
        axes[row, col].set_xlabel('')
        ax2.set_ylabel('Density', fontsize=9)
        
        # Corrected mean markers calculation
        species_means = iris_df.groupby('species')[feature].mean()  # Fixed indexing
        for j, sp in enumerate(iris_df['species'].unique()):  # Fixed colon
            axes[row, col].axhline(y=species_means[sp], color='darkred', 
                                  linestyle='--', alpha=0.7, 
                                  xmin=0.05 + j*0.3,  # Fixed: multiplication not addition
                                  xmax=0.25 + j*0.3)  # Fixed: multiplication not addition

    plt.suptitle('Feature Distributions by Iris Species', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


# LDA Projection and Visualization
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Prepare data
X = iris_df.iloc[:, :-1]
y = iris_df['species']

# Fit LDA model
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Create LDA dataframe
lda_df = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
lda_df['species'] = y

# Plot LDA projection
plt.figure(figsize=(10, 8))
sns.scatterplot(x='LD1', y='LD2', data=lda_df, 
                hue='species', palette='viridis',
                s=100, alpha=0.8, edgecolor='k')

# Add separation metrics
lda_ratio = lda.explained_variance_ratio_.sum() * 100
plt.title(f'LDA Projection (Captures {lda_ratio:.1f}% of Separation)', 
          fontsize=14, pad=15)
plt.xlabel('Linear Discriminant 1', fontsize=12)
plt.ylabel('Linear Discriminant 2', fontsize=12)

# Add decision boundaries (visual estimate)
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
plt.fill_between([x_min, 0], y_min, y_max, color='#440154', alpha=0.05)
plt.fill_between([0, x_max], y_min, y_max, color='#21918c', alpha=0.05)

plt.legend(title='Species', loc='best')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# Build and evaluate classification model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Split data (80% train, 20% test)
X = iris_df.iloc[:, :-1]
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted Species', fontsize=12)
plt.ylabel('True Species', fontsize=12)
plt.title('Classification Confusion Matrix', fontsize=14, pad=12)
plt.show()

# Classification report
print("\n" + "="*55)
print("Classification Report:")
print("="*55)
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("="*55)

# Feature importance
plt.figure(figsize=(10, 5))
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Feature Importance for Species Classification', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()


# === VIOLIN PLOTS ===
plt.figure(figsize=(13, 9))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='Species', y=feature, data=iris_df, 
                  inner='quartile',  # Show quartile lines
                  cut=0,             # Don't trim density tails
                  saturation=0.8,
                  linewidth=1.5)
    
    # Add mean markers
    species_means = iris_df.groupby('Species')[feature].mean()
    for j, mean_val in enumerate(species_means):
        plt.scatter(j, mean_val, marker='s', s=70, color='yellow', 
                   edgecolor='k', zorder=5, label='Mean' if j==0 else "")
    
    plt.title(f'{feature} Distribution', fontsize=14, fontweight='bold')
    plt.ylabel(f'{feature} (cm)', labelpad=10)
    plt.xlabel('')
    
    if i == 0:
        plt.legend()

plt.suptitle('Feature Density Distributions', y=0.99, fontsize=18)
plt.tight_layout(pad=2.0)
plt.show()


# === CORRELATION MATRIX ===
plt.figure(figsize=(9, 7))
corr_matrix = iris_df.iloc[:, :4].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Create heatmap
ax = sns.heatmap(corr_matrix,
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1,
            linewidths=0.5,
            cbar_kws={'shrink': 0.7, 'label': 'Correlation Coefficient'},
            annot_kws={'size': 12})

# Formatting
plt.title('Feature Correlation Matrix', pad=20, fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()


# PROPER submission code template
import pandas as pd

# 1. Generate predictions (example - adapt to your model)
predictions = model.predict(test_data)  # Your actual prediction code here

# 2. Create submission DataFrame
submission = pd.DataFrame({
    'Id': test_ids,  # Make sure these match competition requirements
    'Target': predictions  # Column name may vary (check competition)
})

# 3. Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created!")


# 1. Load required data (ADAPT THESE TO YOUR NOTEBOOK)
# ---------------------------------------------------
from sklearn.datasets import load_iris
import pandas as pd

# Load your test data - THIS MUST MATCH YOUR COMPETITION DATA
# Replace this with however you loaded data earlier in your notebook
iris = load_iris()
test_data = pd.DataFrame(iris.data, columns=iris.feature_names)  # Example - use your actual test data
test_ids = range(len(test_data))  # Or use actual IDs from competition data

# 2. Load your trained model (ADAPT TO YOUR NOTEBOOK)
# ---------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()  # Replace with your actual trained model
model.fit(test_data, iris.target)  # Example - remove if you already trained

# 3. Generate predictions
# ---------------------------------------------------
predictions = model.predict(test_data)

# 4. Create submission DataFrame (CHECK COMPETITION COLUMN NAMES!)
# ---------------------------------------------------
submission = pd.DataFrame({
    'Id': test_ids,  # Must match competition requirements
    'Species': predictions  # Column name may vary - check competition!
})

# 5. Save submission file
# ---------------------------------------------------
submission.to_csv('submission.csv', index=False)
print("✔ Submission file created successfully!")
print(submission.head())  # Verify the output


# 1. Verify your test data
print("Test data sample:")
print(test_data.head())  # Should match competition features

# 2. Verify your model is trained
if 'model' not in locals():
    print("⚠️ Model not found - retrain your model!")
else:
    print("Model score:", model.score(X_train, y_train))  # Should be >0.9 for iris

# 3. Generate proper predictions
predictions = model.predict(test_data)

# 4. Map numeric predictions to species names (if needed)
if isinstance(predictions[0], int):
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    predictions = [species_map[p] for p in predictions]

# 5. Create submission
submission = pd.DataFrame({
    'Id': test_ids,
    'Species': predictions  # Now with actual species names
})

# 6. Save and verify
submission.to_csv('submission.csv', index=False)
print("\n✔ Final Submission:")
print(submission.head(10))  # Show more samples
print("\nValue counts:")
print(submission['Species'].value_counts())  # Check prediction distribution

