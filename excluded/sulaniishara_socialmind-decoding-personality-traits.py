# ================================
#            IMPORTS
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ================================
#         VISUAL SETTINGS
# ================================
sns.set_palette("PRGn")
sns.set_style("whitegrid", {
    'grid.color': '.7',
    'grid.linestyle': ':',
    'grid.linewidth': 0.7
})

# ================================
#         PREPROCESSING
# ================================
def preprocess_data(original, train, test):
    # Match records from the original dataset
    original = original.rename(columns={'Personality': 'match_p'})
    original = original.drop_duplicates(subset=original.drop(columns='match_p').columns.tolist())
    train = train.merge(original, how='left')
    test = test.merge(original, how='left')

    # Separate features and target
    X = train.drop(columns=['Personality'])
    y = train['Personality']

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y))

    return X, test, y_encoded, label_encoder

# ================================
#      FEATURE ENGINEERING
# ================================
def feature_engineering(X_train, X_val):
    X_train = X_train.copy()
    X_val = X_val.copy()

    for df in [X_train, X_val]:
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df.drop(columns=[col for col in ['id'] if col in df.columns], inplace=True)
        df['stage_fear'] = df['stage_fear'].fillna('unknown')
        df['drained_after_socializing'] = df['drained_after_socializing'].fillna('unknown')
        df['match_p_is_null'] = df['match_p'].isna().astype(int)
        df['match_p'] = df['match_p'].fillna('unknown')

        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].fillna(df[col].mean())

    # Create cross-term features (pairwise products of numerical variables)
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            feat_name = f"{num_cols[i]}_x_{num_cols[j]}"
            X_train[feat_name] = X_train[num_cols[i]] * X_train[num_cols[j]]
            X_val[feat_name] = X_val[num_cols[i]] * X_val[num_cols[j]]

    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = encoder.transform(X_val[cat_cols])

    return X_train, X_val

# ================================
#        MODEL TRAINING
# ================================
def fit_cross_val_model(X, y, model, n_splits=7):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models, scores = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train_prep, X_val_prep = feature_engineering(X_train, X_val)

        model.fit(
            X_train_prep, y_train,
            eval_set=[(X_val_prep, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict(X_val_prep)
        acc = accuracy_score(y_val, preds)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        scores.append(acc)
        models.append(model)

    print(f"\nMean Cross-Validation Accuracy: {np.mean(scores):.4f}")
    return models, np.mean(scores)

# ================================
#           EVALUATION
# ================================
def evaluate_model(model, X_val, y_val, label_encoder):
    preds = model.predict(X_val)
    report = classification_report(y_val, preds, target_names=label_encoder.classes_)
    cm = confusion_matrix(y_val, preds)

    print("\nClassification Report:\n", report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="PRGn", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Calibration Curve
    if len(np.unique(y_val)) == 2:
        prob_pos = model.predict_proba(X_val)[:, 1]
        prob_true, prob_pred = calibration_curve(y_val, prob_pos, n_bins=10)

        plt.figure(figsize=(6, 5))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("True Positive Rate")
        plt.title("Calibration Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Confidence Interval via Bootstrap
        bootstrap_acc = [
            accuracy_score(y_val.iloc[np.random.choice(len(y_val), len(y_val), replace=True)], preds)
            for _ in range(1000)
        ]
        ci_lower, ci_upper = np.percentile(bootstrap_acc, [2.5, 97.5])
        print(f"Accuracy 95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ================================
#         SUBMISSION
# ================================
def generate_submission(models, X_test, label_encoder, test_df, output_file='submission.csv'):
    print("\nGenerating submission file...")

    # Average predicted probabilities from all models
    probs = np.mean([model.predict_proba(X_test) for model in models], axis=0)
    preds = np.argmax(probs, axis=1)
    predicted_labels = label_encoder.inverse_transform(preds)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': predicted_labels
    })
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved!")

    print("\nğŸ”� Prediction Counts:")
    print(submission['Personality'].value_counts())

    plt.figure(figsize=(12, 5))
    sns.countplot(data=submission, x='Personality', palette='PRGn')
    plt.title("Test Set Personality Distribution")
    plt.xlabel("Predicted Personality")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    if probs.shape[1] == 2:
        proba_df = pd.DataFrame({
            'prob_class_1': probs[:, 1],
            'predicted_label': predicted_labels
        })

        plt.figure(figsize=(12, 5))
        sns.histplot(
            data=proba_df,
            x='prob_class_1',
            hue='predicted_label',
            bins=30,
            kde=True,
            palette='PRGn',
            alpha=0.6
        )
        plt.title("Distribution of Predicted Probabilities (Binary Classification)")
        plt.xlabel("Predicted Probability for Positive Class")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    display(submission.head())

# ================================
#     FEATURE IMPORTANCE
# ================================
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 5))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='PRGn')
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

# ================================
#        MAIN PIPELINE
# ================================
def run_pipeline():
    train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')

    X, test_processed, y, label_encoder = preprocess_data(original, train, test)
    _, X_test = feature_engineering(X, test_processed)

    print("\nğŸš« Skipping Optuna. Using Best Trial Parameters...")
    best_params = {
        'n_estimators':758,
        'max_depth': 5,
        'learning_rate': 0.033719260906216304,
        'min_child_weight': 2,
        'subsample': 0.9787957652855827,
        'colsample_bytree': 0.8393127138353787,
        'objective': "binary:logistic",
        'eval_metric': "logloss",
        'random_state': 42,
        'tree_method': "gpu_hist",
        'use_label_encoder': False
    }
    model = XGBClassifier(**best_params)

    print("\nğŸ§ª Cross-validation training...")
    models, _ = fit_cross_val_model(X, y, model)

    # Hold-out evaluation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_final, X_val_final = feature_engineering(X_train_split, X_val_split)
    model.fit(X_train_final, y_train_split)
    evaluate_model(model, X_val_final, y_val_split, label_encoder)

    print("\nğŸ“Š Plotting feature importances...")
    plot_feature_importance(model, X_train_final.columns)

    print("\nğŸ“¤ Generating final submission...")
    generate_submission(models, X_test, label_encoder, test_processed)

# Execute
if __name__ == "__main__":
    run_pipeline()


