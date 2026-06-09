import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier

import pickle


target = 'Fertilizer Name'


# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Extract target
X = train_df.drop(columns=[target])
y = train_df[target]


# Combine to apply transformation together
combined = pd.concat([X, test_df], axis=0)


# Basic stats
print(train_df.info(),"\n")
print(train_df.describe(),"\n")

# Count of unique labels
print(train_df['Fertilizer Name'].value_counts())


# Continuous features
numerical_cols = ['Temparature', 'Humidity', 'Moisture',
                  'Nitrogen', 'Potassium', 'Phosphorous']


cat_cols = ['Soil Type', 'Crop Type']


def plot_histogram_colored_by_mode(df, feature_col, target_col, bins=30):
    # Bin the data
    bin_edges = np.histogram_bin_edges(df[feature_col], bins=bins)
    df['bin'] = pd.cut(df[feature_col], bins=bin_edges, include_lowest=True)

    # Get mode of target_col per bin
    bin_modes = df.groupby('bin')[target_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

    # Count how many entries fall in each bin
    bin_counts = df['bin'].value_counts(sort=False)

    # Map each mode to a distinct color
    unique_modes = bin_modes.dropna().unique()
    cmap = cm.get_cmap('tab20', len(unique_modes))
    mode_to_color = {mode: cmap(i) for i, mode in enumerate(unique_modes)}

    # Get the centers of the bins for plotting on x-axis
    bin_centers = [interval.left + (interval.right - interval.left) / 2 for interval in bin_counts.index]

    # Determine the color of each bar
    bar_colors = [mode_to_color.get(bin_modes.get(b), 'lightgray') for b in bin_counts.index]

    # Plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(bin_centers, bin_counts.values, width=np.diff(bin_edges), align='center', color=bar_colors)

    plt.xlabel(feature_col)
    plt.ylabel('Count')
    plt.title(f'Histogram of {feature_col} colored by mode of {target_col}')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in mode_to_color.items()]
    labels = list(mode_to_color.keys())
    plt.legend(handles, labels, title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    df.drop(columns='bin', inplace=True)


for col in numerical_cols:
    plot_histogram_colored_by_mode(train_df, col, 'Fertilizer Name')


# Countplot of Soil Type vs Fertilizer
plt.figure(figsize=(20, 8))
sns.countplot(data=train_df, x='Soil Type', hue='Fertilizer Name')
plt.title("Fertilizer Distribution by Soil Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Crop Type vs Fertilizer
plt.figure(figsize=(20, 8))
sns.countplot(data=train_df, x='Crop Type', hue='Fertilizer Name')
plt.title("Fertilizer Distribution by Crop Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Boxplots of each numerical feature grouped by Fertilizer
for col in numerical_cols:
    plt.figure(figsize=(20, 8))
    sns.boxplot(data=train_df, x='Fertilizer Name', y=col)
    plt.title(f'{col} by Fertilizer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def cramers_v(x, y):
    """Compute CramÃ©râ€™s V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))  # Bias correction
    r_corr = r - ((r-1)**2) / (n-1)
    k_corr = k - ((k-1)**2) / (n-1)
    return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))


# ========== Label encode target ==========
df_corr = train_df.copy()
le_target = LabelEncoder()
df_corr['Fertilizer Name Encoded'] = le_target.fit_transform(df_corr['Fertilizer Name'])



# ========== CramÃ©râ€™s V for categorical features ==========
cramers_results = {}
for col in cat_cols:
    cramers_results[col] = cramers_v(df_corr[col], df_corr['Fertilizer Name'])


# ========== ANOVA F-test for numerical features ==========
X_num = df_corr[numerical_cols]
y_cat = df_corr['Fertilizer Name Encoded']
f_scores, p_vals = f_classif(X_num, y_cat)
anova_results = dict(zip(numerical_cols, f_scores))


anova_df = pd.DataFrame({
    'Feature': list(anova_results.keys()),
    'Score': list(anova_results.values())
}).sort_values(by='Score', ascending=True)

cramer_df = pd.DataFrame({
    'Feature': list(cramers_results.keys()),
    'Score': list(cramers_results.values())
}).sort_values(by='Score', ascending=True)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# Plot CramÃ©râ€™s V
sns.barplot(data=cramer_df, x='Score', y='Feature', hue='Feature', ax=axes[0], palette='Blues_d', legend=False)
axes[0].set_title("CramÃ©râ€™s V (Categorical Features)")
axes[0].set_xlabel("Score")
axes[0].set_ylabel("Feature")

# Plot ANOVA F
sns.barplot(data=anova_df, x='Score', y='Feature', hue='Feature', ax=axes[1], palette='Greens_d', legend=False)
axes[1].set_title("ANOVA F-test (Numerical Features)")
axes[1].set_xlabel("Score")
axes[1].set_ylabel("")

plt.suptitle("Feature Relevance to Fertilizer Name", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


def add_interaction_features(df):
    df['N+P'] = df['Nitrogen'] + df['Phosphorous']
    df['K_ratio'] = df['Potassium'] / (df['Nitrogen'] + 1)
    df['Humidity_Temp'] = df['Humidity'] * df['Temparature']
    return df

combined = add_interaction_features(combined)
df_corr = add_interaction_features(df_corr)  # to see if new feature is good


numerical_cols.append('N+P')
numerical_cols.append('K_ratio')
numerical_cols.append('Humidity_Temp')


# Recompute ANOVA after feature engineering
X_num = df_corr[numerical_cols]
y_cat = df_corr['Fertilizer Name Encoded']
f_scores, p_vals = f_classif(X_num, y_cat)
anova_results = dict(zip(numerical_cols, f_scores))

anova_df = pd.DataFrame({
    'Feature': list(anova_results.keys()),
    'Score': list(anova_results.values())
}).sort_values(by='Score', ascending=True)

# ðŸ”§ FIXED: Create a NEW figure instead of reusing axes[1]
plt.figure(figsize=(8, 6))
sns.barplot(data=anova_df, x='Score', y='Feature', hue='Feature', palette='Greens_d', legend=False)
plt.title("ANOVA F-test (Numerical Features) - After Feature Engineering")
plt.xlabel("Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(
    df_corr[numerical_cols].corr(method='pearson'),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    vmin=-1,
    vmax=1
)
plt.title("Pearson Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.show()


# Drop from DataFrame
combined.drop(columns=['Temparature', 'Nitrogen'], inplace=True)

# Drop from numerical_cols list
numerical_cols = [col for col in numerical_cols if col not in ['Temparature', 'Nitrogen']]



combined_encoded = pd.get_dummies(combined, columns=cat_cols)


scaler = StandardScaler()
combined_encoded[numerical_cols] = scaler.fit_transform(combined_encoded[numerical_cols])


# Split back into train/test
X_encoded = combined_encoded.iloc[:len(X)]
X_test_encoded = combined_encoded.iloc[len(X):]

# Encode the target column
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Store target label mappings
target_label_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))


# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=123, stratify=y_encoded
)


target_label_mapping


# Identify one-hot encoded categorical columns
categorical_dummies = pd.get_dummies(combined[cat_cols]).columns

# Drop categorical columns from existing train/val/test splits
X_train_nocat = X_train.drop(columns=categorical_dummies)
X_val_nocat = X_val.drop(columns=categorical_dummies)
X_test_encoded_nocat = X_test_encoded.drop(columns=categorical_dummies)


rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)


rf_model_nocat = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf_model_nocat.fit(X_train_nocat, y_train)


lgbm_model = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_encoded)),
    random_state=42
)
lgbm_model.fit(X_train, y_train)


lgbm_model_nocat = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_encoded)),
    random_state=42
)
lgbm_model_nocat.fit(X_train_nocat, y_train)


xgb_model = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(np.unique(y_encoded)),
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train, y_train)


xgb_model_nocat = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(np.unique(y_encoded)),
    use_label_encoder=False,
    random_state=42
)
xgb_model_nocat.fit(X_train_nocat, y_train)


def mapk(y_true, y_pred, k=3):
    def apk(actual, predicted, k):
        predicted = list(predicted)  # âœ… Convert to list
        if actual in predicted[:k]:
            return 1 / (predicted[:k].index(actual) + 1)
        return 0

    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)])


def generate_top3_submission(model, X_test, test_ids, label_encoder, filename,
                             X_train=None, y_train=None, 
                             X_val=None, y_val=None):
    """
    Predicts top 3 classes using model.predict_proba and saves the submission file.
    Optionally computes MAP@3 on train and validation sets.

    Parameters:
    - model: trained classifier with predict_proba
    - X_test: test features
    - test_ids: array-like, IDs from test set
    - label_encoder: fitted LabelEncoder for the target
    - filename: output CSV filename
    - X_train, y_train: optional, for MAP@3 evaluation
    - X_val, y_val: optional, for MAP@3 evaluation
    """
    # ======= Top-3 prediction for submission =======
    probs = model.predict_proba(X_test)
    top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    top3_labels = [' '.join(label_encoder.inverse_transform(row)) for row in top3_indices]

    submission_df = pd.DataFrame({
        'id': test_ids,
        'Fertilizer Name': top3_labels
    })
    submission_df.to_csv(filename, index=False)
    print(f"âœ… Saved: {filename}")

    # ======= MAP@3 for training set =======
    if X_train is not None and y_train is not None:
        probs_train = model.predict_proba(X_train)
        top3_train = np.argsort(probs_train, axis=1)[:, -3:][:, ::-1]
        map3_train = mapk(y_train, top3_train, k=3)
        acc_train = accuracy_score(y_train, np.argmax(probs_train, axis=1))
        print(f"ðŸ”¹ Train Accuracy: {acc_train:.4f} | MAP@3: {map3_train:.4f}")

    # ======= MAP@3 for validation set =======
    if X_val is not None and y_val is not None:
        probs_val = model.predict_proba(X_val)
        top3_val = np.argsort(probs_val, axis=1)[:, -3:][:, ::-1]
        map3_val = mapk(y_val, top3_val, k=3)
        acc_val = accuracy_score(y_val, np.argmax(probs_val, axis=1))
        print(f"ðŸ”¹ Val Accuracy: {acc_val:.4f} | MAP@3: {map3_val:.4f}")


generate_top3_submission(
    model=rf_model,
    X_test=X_test_encoded,
    test_ids=test_df['id'],
    label_encoder=target_encoder,
    filename='submission_rf.csv',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)



generate_top3_submission(
    model=rf_model_nocat,
    X_test=X_test_encoded_nocat,
    test_ids=test_df['id'],
    label_encoder=target_encoder,
    filename='submission_rf_nocat.csv',
    X_train=X_train_nocat,
    y_train=y_train,
    X_val=X_val_nocat,
    y_val=y_val
)



generate_top3_submission(
    model=lgbm_model,
    X_test=X_test_encoded,
    test_ids=test_df['id'],
    label_encoder=target_encoder,
    filename='submission_lgbmboost.csv',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)



generate_top3_submission(
    model=lgbm_model_nocat,
    X_test=X_test_encoded_nocat,
    test_ids=test_df['id'],
    label_encoder=target_encoder,
    filename='submission_lgbmboost_nocat.csv',
    X_train=X_train_nocat,
    y_train=y_train,
    X_val=X_val_nocat,
    y_val=y_val
)



generate_top3_submission(
    model=xgb_model,
    X_test=X_test_encoded,
    test_ids=test_df['id'],
    label_encoder=target_encoder,
    filename='submission_xgboost.csv',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)



generate_top3_submission(
    model=xgb_model_nocat,
    X_test=X_test_encoded_nocat,
    test_ids=test_df['id'],
    label_encoder=target_encoder,
    filename='submission_xgboost_nocat.csv',
    X_train=X_train_nocat,
    y_train=y_train,
    X_val=X_val_nocat,
    y_val=y_val
)



ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model), ('lgb', lgbm_model), ('xgb', xgb_model)
], voting='soft')

ensemble_model.fit(X_train, y_train)


ensemble_model_nocat = VotingClassifier(estimators=[
    ('rf', rf_model_nocat), ('lgb', lgbm_model_nocat), ('xgb', xgb_model_nocat)
], voting='soft')

ensemble_model_nocat.fit(X_train_nocat, y_train)


specialist_model = OneVsRestClassifier(LGBMClassifier())
specialist_model.fit(X_train, y_train)


specialist_model_nocat = OneVsRestClassifier(LGBMClassifier())
specialist_model_nocat.fit(X_train_nocat, y_train)


generate_top3_submission(
    model=ensemble_model,
    X_test=X_test_encoded,
    test_ids=test_df['id'],
    label_encoder=target_encoder,
    filename='submission_ensemble.csv',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)



generate_top3_submission(
    model=ensemble_model_nocat,
    X_test=X_test_encoded_nocat,
    test_ids=test_df['id'],
    label_encoder=target_encoder,
    filename='submission_ensemble_nocat.csv',
    X_train=X_train_nocat,
    y_train=y_train,
    X_val=X_val_nocat,
    y_val=y_val
)



generate_top3_submission(
    model=specialist_model,
    X_test=X_test_encoded,
    test_ids=test_df['id'],
    label_encoder=target_encoder,
    filename='submission_specialist.csv',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)



generate_top3_submission(
    model=specialist_model_nocat,
    X_test=X_test_encoded_nocat,
    test_ids=test_df['id'],
    label_encoder=target_encoder,
    filename='submission_specialist_nocat.csv',
    X_train=X_train_nocat,
    y_train=y_train,
    X_val=X_val_nocat,
    y_val=y_val
)



# Save all relevant datasets
with open("fertilizer_data_splits.pkl", "wb") as f:
    pickle.dump({
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_test": X_test_encoded,
        "X_train_nocat": X_train_nocat,
        "X_val_nocat": X_val_nocat,
        "X_test_nocat": X_test_encoded_nocat,
        "target_encoder": target_encoder,
        "target_label_mapping": target_label_mapping
    }, f)

print("âœ… All data splits (including _nocat) saved to fertilizer_data_splits.pkl")

