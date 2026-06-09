# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# âœ… Section 1: Imports and Data Loadingã€€ã‚°ãƒ©ãƒ•ã�ªã�©ã‚’è¡¨ç¤ºã�™ã‚‹ã�Ÿã‚�ã�«å¿…è¦�ã�ªã‚³ãƒ¼ãƒ‰

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random, time

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')
sns.set_palette("husl")

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

train_clean = train.copy()
test_clean = test.copy()

print(f"âœ… Train shape: {train.shape}, Test shape: {test.shape}")


# âœ… Section 2: Preprocessing (cleaning, encoding, scaling)ãƒ‡ãƒ¼ã‚¿ã�®å‰�å‡¦ç�†ã€�æ¬ æ��å€¤ã�®å‡¦ç�†ã�ªã�©

# Replace infinities and NaNs
train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
test_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

common_numeric_cols = list(set(train_clean.select_dtypes(include=[np.number]).columns) &
                           set(test_clean.select_dtypes(include=[np.number]).columns))
common_categorical_cols = list(set(train_clean.select_dtypes(include=['object']).columns) &
                               set(test_clean.select_dtypes(include=['object']).columns))

if 'id' in common_numeric_cols: common_numeric_cols.remove('id')
if 'Personality' in common_numeric_cols: common_numeric_cols.remove('Personality')

# Impute missing values
numeric_imputer = SimpleImputer(strategy='median')
train_clean[common_numeric_cols] = numeric_imputer.fit_transform(train_clean[common_numeric_cols])
test_clean[common_numeric_cols] = numeric_imputer.transform(test_clean[common_numeric_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')
train_clean[common_categorical_cols] = categorical_imputer.fit_transform(train_clean[common_categorical_cols])
test_clean[common_categorical_cols] = categorical_imputer.transform(test_clean[common_categorical_cols])

# Encode categorical features
for col in common_categorical_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train_clean[col], test_clean[col]], axis=0).astype(str)
    le.fit(all_vals)
    train_clean[col] = le.transform(train_clean[col].astype(str))
    test_clean[col] = le.transform(test_clean[col].astype(str))


# âœ… Section 3: Target Encoding and Feature Scalingãƒ¢ãƒ‡ãƒ«å­¦ç¿’ãƒ»è©•ä¾¡

# Encode 'Personality' target
target_le = LabelEncoder()
train_clean['Personality'] = target_le.fit_transform(train_clean['Personality'])
y_binary = train_clean['Personality']
target_names = target_le.classes_

# Final features
features = common_numeric_cols + common_categorical_cols
X_train = train_clean[features]
X_test = test_clean[features]
test_ids = test_clean['id']

# Scale
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)


# âœ… Section 4: Model Training and Evaluation

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_binary, test_size=0.2, stratify=y_binary, random_state=42
)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
trained = {}
all_feature_importance = {}
for name, model in models.items():
    print(f"ğŸ”§ Training {name}...")
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_val, model.predict(X_val))
    cv = cross_val_score(model, X_tr, y_tr, cv=5).mean()
    results[name] = {"accuracy": acc, "cv": cv}
    trained[name] = model

    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        all_feature_importance[name] = fi

    print(f"âœ… Val Accuracy: {acc:.4f}, CV: {cv:.4f}")

best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = trained[best_model_name]
best_accuracy = results[best_model_name]['accuracy']
print(f"ğŸ�† Best Model: {best_model_name}")



# âœ… Section 5: Visual Analysis of the Models

# ğŸ“Š Compare Validation Accuracy
comparison_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Val_Accuracy": [results[m]["accuracy"] for m in results],
    "CV_Score": [results[m]["cv"] for m in results]
})

plt.figure(figsize=(14, 5))

# Accuracy bar plot
plt.subplot(1, 2, 1)
sns.barplot(x="Model", y="Val_Accuracy", data=comparison_df, palette="Blues_d")
plt.ylim(0.0, 1.0)
plt.title("ğŸ“Š Validation Accuracy by Model")
plt.ylabel("Accuracy")
plt.xlabel("Model")

# CV score bar plot
plt.subplot(1, 2, 2)
sns.barplot(x="Model", y="CV_Score", data=comparison_df, palette="Greens_d")
plt.ylim(0.0, 1.0)
plt.title("ğŸ“ˆ Cross-Validation Score by Model")
plt.ylabel("CV Score")
plt.xlabel("Model")

plt.suptitle("ğŸ”� Model Performance Comparison", fontsize=14)
plt.tight_layout()
plt.show()

# ğŸ“Œ Feature Importances for Tree-Based Models

for model_name, importance_df in all_feature_importance.items():
    top_n = importance_df.head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='importance', y='feature', data=top_n, palette='viridis')
    plt.title(f"ğŸ”¥ Top 10 Feature Importances - {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# ğŸ“‰ Logistic Regression Feature Coefficients (Top 10 by magnitude)

if "Logistic Regression" in trained:
    logreg_model = trained["Logistic Regression"]
    if hasattr(logreg_model, "coef_"):
        coef = logreg_model.coef_[0]
        coef_df = pd.DataFrame({
            "feature": features,
            "coefficient": coef
        })

        # Sort by absolute value (magnitude) and take top 10
        top_coef_df = coef_df.reindex(coef_df.coefficient.abs().sort_values(ascending=False).index).head(10)

        plt.figure(figsize=(8, 5))
        sns.barplot(x="coefficient", y="feature", data=top_coef_df,
                    palette=["#2E8B57" if c > 0 else "#DC143C" for c in top_coef_df["coefficient"]])
        plt.axvline(0, color="black", linestyle="--")
        plt.title("ğŸ“‰ Top Logistic Regression Feature Coefficients")
        plt.xlabel("Coefficient Value")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
    else:
        print("âš ï¸� Logistic Regression has no coefficients.")
else:
    print("âš ï¸� Logistic Regression model not found.")


# âœ… Section 6: Heatmap and Feature Distributions

# ğŸ”¥ Correlation Heatmap
plt.figure(figsize=(14, 10))
corr = X_train[features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("ğŸ”— Full Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ğŸ“Š Feature Distributions
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
n_features = min(16, len(numeric_features))

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(numeric_features[:n_features]):
    X_train[col].hist(bins=20, ax=axes[i], color='lightblue', alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    mean_val = X_train[col].mean()
    axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
    axes[i].legend(fontsize=8)

for i in range(n_features, 16):
    axes[i].set_visible(False)

plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# âœ… Section 7: Classification Report and Confusion Matrix

val_pred_best = best_model.predict(X_val)
print(f"ğŸ“‹ Classification Report - {best_model_name}")
print(classification_report(y_val, val_pred_best, target_names=target_names))

cm = confusion_matrix(y_val, val_pred_best)
ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()


# âœ… Section 8: Feature Importance (Top 10)

if hasattr(best_model, 'feature_importances_'):
    feat_imp_df = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp_df, palette='crest')
    plt.title(f'Top 10 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.show()


# âœ… Section 9: Random ID Personality Test (Robust for all models)

used_ids = set()

def create_personality_pie_chart(user_id, personality_type, confidence, feature_contributions, model_name):
    try:
        labels = [f"{feat}   ({pct:.1f}%)" for feat, pct in feature_contributions.items()]
        sizes = list(feature_contributions.values())
        colors = sns.color_palette('husl', len(sizes))

        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90, pctdistance=0.85)

        plt.setp(autotexts, size=8, weight="bold")
        plt.setp(texts, size=7)
        title_color = '#2E8B57' if personality_type == 'Extrovert' else '#4169E1'
        plt.title(f"ID {user_id}: {personality_type} ({confidence:.1%})  Model: {model_name}", fontsize=14, color=title_color)

        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    except Exception as e:
            print(f"âš ï¸� Error creating visualization: {str(e)}")

def analyze_individual_id(user_id, X_data, model, model_name):
    if user_id not in X_data.index:
        print(f"â�Œ ID {user_id} not found")
        return

    individual = X_data.loc[[user_id]]
    prediction = model.predict(individual)[0]
    proba = model.predict_proba(individual)[0]
    predicted_label = target_names[prediction]
    confidence = max(proba)

    # ğŸ”� Handle feature importance fallback
    if hasattr(model, 'feature_importances_'):
        importance_dict = dict(zip(X_data.columns, model.feature_importances_))
    else:
        importance_dict = dict.fromkeys(X_data.columns, 1.0)

    # Compute contributions
    contributions = {}
    for feature, importance in importance_dict.items():
        val = individual.iloc[0][feature]
        contributions[feature] = abs(importance * val)

    total = sum(contributions.values())
    full_contributions = {k: (v / total) * 100 if total > 0 else 0 for k, v in contributions.items()}
    create_personality_pie_chart(user_id, predicted_label, confidence, full_contributions, model_name)
    return predicted_label, confidence

def quick_personality_test():
    available = list(set(X_train_scaled.index.tolist()) - used_ids)
    if not available:
        print("âœ… All IDs tested.")
        return
    test_id = random.choice(available)
    used_ids.add(test_id)
    print(f"ğŸ�² Analyzing ID: {test_id}")
    analyze_individual_id(test_id, X_train_scaled, best_model, best_model_name)

quick_personality_test()


# âœ… FINAL Section 10: Robust Submission Creation with Invalid Handling

# Re-check the raw (preprocessed) test data for original NaNs
original_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# Predict only for rows that had no missing values originally
valid_mask = original_test.drop(columns=['id']).notna().all(axis=1)
valid_ids = test_ids[valid_mask]
invalid_ids = test_ids[~valid_mask]

# Make predictions only for valid rows
X_test_valid = X_test_scaled.loc[valid_ids.index]
test_probs = best_model.predict_proba(X_test_valid)
valid_predictions = [target_names[np.argmax(p)] for p in test_probs]

# Build result list
final_results = []

for id_ in test_ids:
    if id_ in valid_ids.values:
        pred = valid_predictions[list(valid_ids.values).index(id_)]
    else:
        pred = "Invalid data"
    final_results.append({"id": id_, "Personality": pred})

# Create final dataframe
submission_df = pd.DataFrame(final_results)
submission_df.to_csv("submission.csv", index=False)
print("ğŸ“� Robust submission.csv created with invalid ID handling.")

