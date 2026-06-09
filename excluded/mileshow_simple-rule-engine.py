import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# --- 1. Data Loading and Preparation ---
print("--- Loading Data ---")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

print("Data loaded successfully.")

# --- FEATURE ENGINEERING: Bin the Top 3 Numerical Features ---
print("\n--- Binning Top 3 Nutrient Features ---")
# Based on our model importances, N, P, and K are the most critical.
top_numerical_features = ['Nitrogen', 'Phosphorous', 'Potassium']
for df in [train_df, test_df]:
    for col in top_numerical_features:
        df[f'{col}_binned'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')

print("Top 3 nutrient features converted to categorical bins.")


# --- Define the features that make up a "rule" ---
# MODIFIED: Using the core categoricals + top 3 binned numericals
rule_features = ['Soil Type', 'Crop Type'] + [f'{col}_binned' for col in top_numerical_features]


# --- Custom MAP@3 Metric ---
def mapk(actual, predicted, k=3):
    actual_wrapped = [[a] for a in actual]
    apk_scores = []
    for a, p in zip(actual_wrapped, predicted):
        p = p[:k]
        score = 0.0
        hits = 0.0
        for i, pred_item in enumerate(p):
            if pred_item in a:
                hits += 1.0
                score += hits / (i + 1.0)
        apk_scores.append(score / min(len(a), k))
    return np.mean(apk_scores)

# --- 2. K-Fold Cross-Validation for Score Estimation ---
print("\n--- Starting 3-Fold Cross-Validation to Estimate MAP@3 Score ---")
FOLDS = 3
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
map3_scores = []

y_encoded = LabelEncoder().fit_transform(train_df['Fertilizer Name'])

for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df, y_encoded), 1):
    print(f"\n===== FOLD {fold} =====")
    
    train_fold_df = train_df.iloc[train_idx]
    valid_fold_df = train_df.iloc[valid_idx]

    # Build Lookup Table on the training portion of the fold using the new rule definition
    rules_lookup = {}
    grouped = train_fold_df.groupby(rule_features)
    for group_name, group_df in grouped:
        fertilizer_counts = Counter(group_df['Fertilizer Name'])
        top_3_fertilizers = [item[0] for item in fertilizer_counts.most_common(3)]
        rules_lookup[group_name] = top_3_fertilizers

    overall_top_3_fold = [item[0] for item in Counter(train_fold_df['Fertilizer Name']).most_common(3)]

    # Apply Rules and Evaluate on the validation portion
    predictions_fold = []
    for index, row in valid_fold_df.iterrows():
        # Create the complex rule key from the row's values
        rule_key = tuple(row[rule_features])
        prediction = rules_lookup.get(rule_key, overall_top_3_fold)
        while len(prediction) < 3:
            prediction.append(overall_top_3_fold[len(prediction)])
        predictions_fold.append(prediction)
    
    score = mapk(valid_fold_df['Fertilizer Name'].values, predictions_fold)
    map3_scores.append(score)
    print(f"Fold {fold} MAP@3 Score: {score:.5f}")

print("\n----------------------------------------------------")
print(f"Average Estimated MAP@3 Score from 3 Folds: {np.mean(map3_scores):.5f}")
print("----------------------------------------------------")


# --- 3. Build Final Lookup Table on FULL Training Data for Submission ---
print("\n--- Building Final Rules Lookup Table on All Training Data ---")

final_rules_lookup = {}
rule_counts = {}
grouped_full = train_df.groupby(rule_features)

for group_name, group_df in grouped_full:
    fertilizer_counts = Counter(group_df['Fertilizer Name'])
    top_3_fertilizers = [item[0] for item in fertilizer_counts.most_common(3)]
    final_rules_lookup[group_name] = top_3_fertilizers
    if fertilizer_counts:
        rule_counts[group_name] = len(group_df) # Count of total occurrences of this rule

print(f"Successfully created {len(final_rules_lookup)} unique rules for final model.")
overall_top_3_final = [item[0] for item in Counter(train_df['Fertilizer Name']).most_common(3)]


# --- 4. Apply Final Rules to the Test Set ---
print("\n--- Applying Final Rules to Test Set for Submission ---")
final_predictions = []
for index, row in test_df.iterrows():
    rule_key = tuple(row[rule_features])
    prediction = final_rules_lookup.get(rule_key, overall_top_3_final)
    while len(prediction) < 3:
        prediction.append(overall_top_3_final[len(prediction)])
    final_predictions.append(prediction)

# --- 5. Analyze Effectiveness ---
print("\n--- Top 50 Most Powerful Rules ---")
top_50_rules = sorted(rule_counts.items(), key=lambda item: item[1], reverse=True)[:50]
for i, (rule, count) in enumerate(top_50_rules, 1):
    top_fertilizer = final_rules_lookup.get(rule, ["N/A"])[0]
    # The rule is now a tuple of all the feature values
    print(f"{i:3d}. IF Features are {rule}, THEN Top Fertilizer='{top_fertilizer}' (Observed {count} times)")


# --- 6. Create Submission File ---
print("\n--- Creating Submission File ---")
final_submission_df = pd.DataFrame()
final_submission_df['id'] = submission_df['id']
final_submission_df['Fertilizer Name'] = [' '.join(pred) for pred in final_predictions]
final_submission_df.to_csv('submission_top3_rules_engine.csv', index=False)

print("\n✅ Submission file 'submission_top3_rules_engine.csv' created successfully!")
display(final_submission_df.head())


