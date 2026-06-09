# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Statistical functions
from scipy.stats import skew

# Display utilities for cleaner notebook output
from IPython.display import display

# Preprocessing and modeling tools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, accuracy_score

# XGBoost model
import xgboost as xgb

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")


# Load competition datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_data = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# Load the original fertilizer dataset (for validation/EDA)
original_data = pd.read_csv('/kaggle/input/fertilizers-original-data/Fertilizer-Prediction.csv')



# Print dataset shapes to confirm successful loading
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)
print("Original Data Shape:", original_data.shape)
print("Sample Submission Shape:", sample_data.shape)


# Preview a few rows from the bottom of the training data
print("ğŸ”� Train Data Preview:")
display(train_data.tail())

# Preview a few rows from the top of the original fertilizer dataset
print("ğŸ”� Original Data Preview:")
display(original_data.head())

# Preview a few rows from the test dataset
print("ğŸ”� Test Data Preview:")
display(test_data.head())


# Function to report missing values in a dataset
def report_missing_values(df, name):
    total_missing = df.isnull().sum().sum()
    total_rows = len(df)

    print("=" * 50)
    print(f"ğŸ“‹ Missing Value Report for: {name}")
    print("=" * 50)

    if total_missing == 0:
        print(f"âœ… No missing values found in {total_rows:,} rows.")
    else:
        print(f"âš ï¸� Found {total_missing:,} missing values across {total_rows:,} rows.")




# Dictionary of datasets to evaluate
all_datasets = {
    "Train Set": train_data,
    "Test Set": test_data,
    "Original Fertilizer Set": original_data
}


# Execute missing value check for each dataset
for dataset_name, df in all_datasets.items():
    report_missing_values(df, dataset_name)
    print()
    


# Function to report duplicate entries
def report_duplicate_records(df, name):
    num_duplicates = df.duplicated().sum()
    total = len(df)

    print("=" * 50)
    print(f"ğŸ”� Duplicate Check for: {name}")
    print("=" * 50)

    if num_duplicates == 0:
        print(f"âœ… No duplicate records found in {total:,} rows.")
    else:
        print(f"âš ï¸� Found {num_duplicates:,} duplicate records ({num_duplicates/total:.2%} of total).")


# Execute duplicate check for each dataset
for dataset_name, df in all_datasets.items():
    report_duplicate_records(df, dataset_name)
    print()


# Define categorical columns to encode
categorical_columns = ['Soil Type', 'Crop Type']

# Create a dictionary to store fitted encoders
label_encoders = {}

# Fit encoders on training data and apply to all datasets
for column in categorical_columns:
    encoder = LabelEncoder()
    train_data[column + '_enc'] = encoder.fit_transform(train_data[column])
    test_data[column + '_enc'] = encoder.transform(test_data[column])
    original_data[column + '_enc'] = encoder.transform(original_data[column])

    label_encoders[column] = encoder  # Save for future use


# Display label mapping for reference
for col, le in label_encoders.items():
    print(f"\nEncoding for '{col}':")
    for i, label in enumerate(le.classes_):
        print(f"  {i}: {label}")


target_variable = 'Fertilizer Name'

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
datasets = [('Train Data', train_data), ('Original Data', original_data)]

for i, (title, data) in enumerate(datasets):
    ax = axes[i, 0]
    sns.countplot(y=target_variable, data=data, ax=ax, palette='viridis')
    ax.set_title(f'Count Plot of Fertilizer Names in {title}', pad=20)
    ax.set_ylabel('Fertilizer Name')
    ax.set_xlabel('Count')
    ax.grid(axis='x', color='gray', linestyle=':', linewidth=0.7)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    for p in ax.patches:
        width = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(width + max(data[target_variable].value_counts())*0.01, y,
                f'{int(width)}', 
                ha='left', va='center', fontsize=10, fontweight='bold', color='black')

    fertilizer_counts = data[target_variable].value_counts().sort_index()
    wedges, texts, autotexts = axes[i, 1].pie(
        fertilizer_counts,
        labels=fertilizer_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("viridis", len(fertilizer_counts)),
        wedgeprops=dict(width=0.4, edgecolor='w'),
        radius=1.2
    )

    for text in texts + autotexts:
        text.set_fontsize(10)
        text.set_fontweight('bold')

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    axes[i, 1].add_artist(centre_circle)
    axes[i, 1].set_title(f'Fertilizer Distribution in {title}', pad=25)
    axes[i, 1].axis('equal')

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.show()



numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
colors = sns.color_palette('viridis', 3)
fig, axes = plt.subplots(len(numerical_features), 2, figsize=(12, len(numerical_features)*4))
axes = np.atleast_2d(axes)

for i, feature in enumerate(numerical_features):
    sns.histplot(train_data[feature], color=colors[0], label='Train Data', bins=20, kde=True, ax=axes[i, 0])
    sns.histplot(test_data[feature], color=colors[1], label='Test Data', bins=20, kde=True, ax=axes[i, 0])
    sns.histplot(original_data[feature], color=colors[2], label='Original Data', bins=20, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {feature}')
    axes[i, 0].legend()
    axes[i, 0].set_facecolor("lightgray")
    axes[i, 0].grid(color='gray', linestyle=':', linewidth=0.7)

    sns.boxplot(data=[train_data[feature], test_data[feature], original_data[feature]],
                palette=colors, orient='h', ax=axes[i, 1])
    axes[i, 1].set_title(f'Horizontal Boxplot of {feature}')
    axes[i, 1].set_yticklabels(['Train Data', 'Test Data', 'Original Data'])
    axes[i, 1].set_xlabel(feature)
    axes[i, 1].set_facecolor("lightgray")
    axes[i, 1].grid(axis='x', color='gray', linestyle=':', linewidth=0.7)

plt.tight_layout()
plt.show()



numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
correlation_datasets = {
    "Train Data": train_data,
    "Original Data": original_data,
    "Test Data": test_data
}

fig, axes = plt.subplots(ncols=3, figsize=(20, 6))
axes = axes.ravel()

for i, (label, df) in enumerate(correlation_datasets.items()):
    corr_matrix = df[numerical_features].corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".4f",
        cmap="viridis",  # you can change to 'viridis', 'Blues', etc.
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.7},
        ax=axes[i]
    )
    axes[i].set_title(f'{label} - Correlation Heatmap', fontsize=14)

plt.tight_layout()
plt.show()



def plot_categorical_distribution_across_datasets(train_df, original_df, test_df, feature_name):
    dataset_labels = ['Train', 'Original', 'Test']
    dataset_sources = [train_df, original_df, test_df]
    color_palette = sns.color_palette("Set2", n_colors=train_df[feature_name].nunique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, (df, label) in enumerate(zip(dataset_sources, dataset_labels)):
        order = df[feature_name].value_counts().index
        sns.countplot(y=feature_name, data=df, order=order, ax=axes[0, i], palette=color_palette)
        axes[0, i].set_title(f'{label} Data - {feature_name} Counts')
        axes[0, i].set_xlabel('Count')
        axes[0, i].set_ylabel(feature_name)

        for p in axes[0, i].patches:
            axes[0, i].annotate(f'{int(p.get_width())}',
                                (p.get_width(), p.get_y() + p.get_height() / 2),
                                ha='left', va='center', fontsize=10)

        wedges, texts, autotexts = axes[1, i].pie(
            df[feature_name].value_counts(),
            labels=order,
            autopct='%1.1f%%',
            startangle=90,
            colors=color_palette,
            textprops={'fontsize': 12}
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[1, i].add_artist(centre_circle)
        axes[1, i].set_title(f'{label} Data - {feature_name} (%)')
        axes[1, i].axis('equal')

    plt.tight_layout()
    plt.show()



# Soil Type Distribution
plot_categorical_distribution_across_datasets(train_data, original_data, test_data, 'Soil Type')

# Crop Type Distribution
plot_categorical_distribution_across_datasets(train_data, original_data, test_data, 'Crop Type')



# Define logical feature groups
environmental_features = ['Temparature', 'Humidity', 'Moisture']
soil_feature = 'Soil Type'
crop_feature = 'Crop Type'
nutrient_features = ['Nitrogen', 'Phosphorous', 'Potassium']



mean_env = train_data.groupby('Fertilizer Name')[environmental_features].mean().reset_index()
mean_env_melted = mean_env.melt(id_vars='Fertilizer Name', var_name='Environmental Feature', value_name='Mean Value')

plt.figure(figsize=(14, 6))
sns.barplot(data=mean_env_melted, x='Fertilizer Name', y='Mean Value', hue='Environmental Feature', palette='Set2')
plt.title('Average Environmental Feature Levels by Fertilizer')
plt.xlabel('Fertilizer Name')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()



def summarize_environment_by_fertilizer(df, env_features):
    for feature in env_features:
        stats = df.groupby('Fertilizer Name')[feature].describe().round(2)
        print(f"\nğŸ”� Summary of {feature} by Fertilizer:")
        display(stats)

summarize_environment_by_fertilizer(train_data, environmental_features)



for feature in environmental_features:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=train_data, x='Fertilizer Name', y=feature, palette='Set2', ax=axes[0])
    axes[0].set_title(f'{feature} by Fertilizer Class')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle=':', alpha=0.7)

    sns.histplot(data=train_data, x=feature, hue='Fertilizer Name', kde=True, palette='Set2', ax=axes[1])
    axes[1].set_title(f'{feature} Distribution across Fertilizers')
    axes[1].grid(axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()



def plot_soil_type_by_fertilizer(data, fertilizer_name):
    subset = data[data['Fertilizer Name'] == fertilizer_name]
    soil_counts = subset['Soil Type'].value_counts()
    order = soil_counts.index.tolist()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("Set2", len(order))

    sns.countplot(y='Soil Type', data=subset, order=order, palette=palette, ax=ax[0])
    ax[0].set_title(f'Soil Type Count - {fertilizer_name}')
    for p in ax[0].patches:
        ax[0].annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height()/2),
                       ha='left', va='center', fontsize=10)

    wedges, texts, autotexts = ax[1].pie(
        soil_counts,
        labels=order,
        autopct='%1.1f%%',
        startangle=90,
        colors=palette,
        textprops={'fontsize': 12}
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax[1].add_artist(centre_circle)
    ax[1].set_title(f'Soil Type (%) - {fertilizer_name}')
    ax[1].axis('equal')

    plt.tight_layout()
    plt.show()

# Loop through all fertilizer types
for fert in train_data['Fertilizer Name'].unique():
    plot_soil_type_by_fertilizer(train_data, fert)



soil_fert_crosstab = pd.crosstab(train_data['Soil Type'], train_data['Fertilizer Name'])

plt.figure(figsize=(14, 7))
sns.heatmap(soil_fert_crosstab, annot=True, fmt='d', cmap='Set2', linewidths=0.5, linecolor='gray')
plt.title('Heatmap of Soil Type vs. Fertilizer Classes', fontsize=16)
plt.xlabel('Fertilizer Name')
plt.ylabel('Soil Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



def plot_crop_type_by_fertilizer(data, fertilizer_name):
    subset = data[data['Fertilizer Name'] == fertilizer_name]
    crop_counts = subset['Crop Type'].value_counts()
    order = crop_counts.index.tolist()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("Set2", len(order))

    sns.countplot(y='Crop Type', data=subset, order=order, palette=palette, ax=ax[0])
    ax[0].set_title(f'Crop Type Count - {fertilizer_name}')
    for p in ax[0].patches:
        ax[0].annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height()/2),
                       ha='left', va='center', fontsize=10)

    wedges, texts, autotexts = ax[1].pie(
        crop_counts,
        labels=order,
        autopct='%1.1f%%',
        startangle=90,
        colors=palette,
        textprops={'fontsize': 12}
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax[1].add_artist(centre_circle)
    ax[1].set_title(f'Crop Type (%) - {fertilizer_name}')
    ax[1].axis('equal')

    plt.tight_layout()
    plt.show()

# Loop through all fertilizer classes
for fert in train_data['Fertilizer Name'].unique():
    plot_crop_type_by_fertilizer(train_data, fert)



crop_fert_crosstab = pd.crosstab(train_data['Crop Type'], train_data['Fertilizer Name'])

plt.figure(figsize=(14, 10))
sns.heatmap(crop_fert_crosstab, annot=True, fmt='d', cmap='Set2', linewidths=0.5, linecolor='gray')
plt.title('Heatmap of Crop Type vs. Fertilizer Classes', fontsize=16)
plt.xlabel('Fertilizer Name')
plt.ylabel('Crop Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



mean_nutrients = train_data.groupby('Fertilizer Name')[nutrient_features].mean().reset_index()
mean_nutrients_melted = mean_nutrients.melt(id_vars='Fertilizer Name', var_name='Nutrient', value_name='Mean Level')

plt.figure(figsize=(14, 6))
sns.barplot(data=mean_nutrients_melted, x='Fertilizer Name', y='Mean Level', hue='Nutrient', palette='Set2')
plt.title('Average Nutrient Levels by Fertilizer Class')
plt.xlabel('Fertilizer Name')
plt.ylabel('Mean Nutrient Level')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()



def summarize_nutrients_by_fertilizer(df, nutrient_list):
    for nutrient in nutrient_list:
        stats = df.groupby('Fertilizer Name')[nutrient].describe().round(2)
        print(f"\nğŸ“Š {nutrient} Summary by Fertilizer:")
        display(stats)

summarize_nutrients_by_fertilizer(train_data, nutrient_features)



# Define the target and encode it
target_column = 'Fertilizer Name'
label_encoder_target = LabelEncoder()
train_data['target_label'] = label_encoder_target.fit_transform(train_data[target_column])

# Define numerical + encoded categorical features
num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
cat_encoded_features = ['Soil Type_enc', 'Crop Type_enc']

# Final input feature set
input_features = num_features + cat_encoded_features

# Prepare training input matrix and target
X_full = train_data[input_features]
y_full = train_data['target_label']

# Save class mapping for future inverse transform
class_mapping = dict(zip(label_encoder_target.classes_, label_encoder_target.transform(label_encoder_target.classes_)))


for nutrient in nutrient_features:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=train_data, x='Fertilizer Name', y=nutrient, palette='Set2', ax=axes[0])
    axes[0].set_title(f'{nutrient} Distribution by Fertilizer Class')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle=':', alpha=0.7)

    sns.histplot(data=train_data, x=nutrient, hue='Fertilizer Name', kde=True, bins=20, palette='Set2', ax=axes[1])
    axes[1].set_title(f'{nutrient} Histogram by Fertilizer Class')
    axes[1].grid(axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()



mean_nutrients_by_soil = train_data.groupby('Soil Type')[nutrient_features].mean().reset_index()
melted_soil_nutrients = mean_nutrients_by_soil.melt(id_vars='Soil Type', var_name='Nutrient', value_name='Mean Level')

plt.figure(figsize=(14, 6))
sns.barplot(data=melted_soil_nutrients, x='Soil Type', y='Mean Level', hue='Nutrient', palette='Set2')
plt.title('Average Nutrient Levels by Soil Type')
plt.xlabel('Soil Type')
plt.ylabel('Mean Nutrient Level')
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()



def summarize_nutrients_by_soil(df, nutrient_list):
    for nutrient in nutrient_list:
        stats = df.groupby('Soil Type')[nutrient].describe().round(2)
        print(f"\nğŸ“Š {nutrient} Summary by Soil Type:")
        display(stats)

summarize_nutrients_by_soil(train_data, nutrient_features)



for nutrient in nutrient_features:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=train_data, x='Soil Type', y=nutrient, palette='Set2', ax=axes[0])
    axes[0].set_title(f'{nutrient} by Soil Type')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle=':', alpha=0.7)

    sns.histplot(data=train_data, x=nutrient, hue='Soil Type', kde=True, bins=20, palette='Set2', ax=axes[1])
    axes[1].set_title(f'{nutrient} Histogram by Soil Type')
    axes[1].grid(axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()



mean_nutrients_by_crop = train_data.groupby('Crop Type')[nutrient_features].mean().reset_index()
melted_crop_nutrients = mean_nutrients_by_crop.melt(id_vars='Crop Type', var_name='Nutrient', value_name='Mean Level')

plt.figure(figsize=(16, 7))
sns.barplot(data=melted_crop_nutrients, x='Crop Type', y='Mean Level', hue='Nutrient', palette='Set2')
plt.title('Average Nutrient Levels by Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Mean Nutrient Level')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()



def summarize_nutrients_by_crop(df, nutrient_list):
    for nutrient in nutrient_list:
        stats = df.groupby('Crop Type')[nutrient].describe().round(2)
        print(f"\nğŸ“Š {nutrient} Summary by Crop Type:")
        display(stats)

summarize_nutrients_by_crop(train_data, nutrient_features)



for nutrient in nutrient_features:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=train_data, x='Crop Type', y=nutrient, palette='Set2', ax=axes[0])
    axes[0].set_title(f'{nutrient} by Crop Type')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle=':', alpha=0.7)

    sns.histplot(data=train_data, x=nutrient, hue='Crop Type', kde=True, bins=20, palette='Set2', ax=axes[1])
    axes[1].set_title(f'{nutrient} Histogram by Crop Type')
    axes[1].grid(axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()



# Import again just in case
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import xgboost as xgb

# Setup: number of folds
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Prepare placeholders
oof_preds = np.zeros((X_full.shape[0], len(class_mapping)))  # probabilities
fold_log_losses = []
models = []


# Loop through each fold
for fold, (train_idx, valid_idx) in enumerate(skf.split(X_full, y_full)):
    print(f"\nğŸ”� Fold {fold+1}/{n_folds}")
    
    X_train, X_valid = X_full.iloc[train_idx], X_full.iloc[valid_idx]
    y_train, y_valid = y_full.iloc[train_idx], y_full.iloc[valid_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(class_mapping),
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
    
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dvalid, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Save model
    models.append(booster)
    
    # Predict on validation fold
    oof_preds[valid_idx] = booster.predict(dvalid)
    
    # Log loss for fold
    loss = log_loss(y_valid, oof_preds[valid_idx])
    fold_log_losses.append(loss)
    print(f"âœ… Fold {fold+1} Log Loss: {loss:.5f}")



# Evaluate overall OOF Log Loss
overall_loss = log_loss(y_full, oof_preds)
print("\nğŸ“Œ OOF Log Loss (All Folds):", round(overall_loss, 5))


def mapk(actual, predicted, k=3):
    """
    Computes mean average precision at k.
    
    Parameters:
    actual : list or array of true labels
    predicted : list of predicted label lists or arrays
    """
    def apk(a, p, k):
        p = list(p[:k])  # Convert to list to use .index()
        try:
            return 1.0 / (p.index(a) + 1)
        except ValueError:
            return 0.0

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



# Get top 3 predicted classes for each OOF row
oof_top3_preds = np.argsort(oof_preds, axis=1)[:, ::-1][:, :3]

# True labels
true_labels = y_full.values

# Compute MAP@3
map3_score = mapk(true_labels, oof_top3_preds, k=3)
print(f"ğŸ“ˆ Out-of-Fold MAP@3 Score: {map3_score:.5f}")



importances = []

for i, model in enumerate(models):
    fold_importance = model.get_score(importance_type='gain')
    fold_df = pd.DataFrame.from_dict(fold_importance, orient='index', columns=['Gain'])
    fold_df['Feature'] = fold_df.index
    fold_df['Fold'] = i + 1
    importances.append(fold_df)

# Combine all folds
importance_df = pd.concat(importances)
importance_df = importance_df.groupby('Feature')['Gain'].mean().sort_values(ascending=False).reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Gain', y='Feature', palette='viridis')
plt.title('Average Feature Importance (XGBoost Gain)')
plt.xlabel('Average Gain')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



# Prepare test data
X_test = test_data[input_features]
dtest = xgb.DMatrix(X_test)

# Average predictions from all models
test_preds = np.mean([model.predict(dtest) for model in models], axis=0)

# Get top 3 class indices
top3_indices = np.argsort(test_preds, axis=1)[:, ::-1][:, :3]

# Convert indices to original fertilizer names
top3_labels = np.vectorize(lambda x: label_encoder_target.inverse_transform([x])[0])(top3_indices)

# Format prediction as space-separated string
submission = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})

submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")
submission.head()


