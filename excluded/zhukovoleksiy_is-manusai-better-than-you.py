import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, PolynomialFeatures
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


def preprocess_and_feature_engineer(train_df, test_df):
    # Combine for consistent preprocessing
    combined_df = pd.concat([train_df.drop("Personality", axis=1), test_df], ignore_index=True)

    # Identify categorical and numerical columns
    categorical_cols = combined_df.select_dtypes(include='object').columns
    numerical_cols = combined_df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols.remove("id")

    # Impute numerical missing values using KNNImputer
    imputer_numerical = KNNImputer(n_neighbors=5)
    combined_df[numerical_cols] = imputer_numerical.fit_transform(combined_df[numerical_cols])

    # Impute categorical missing values with mode
    for col in categorical_cols:
        combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

    # Encode categorical features using OrdinalEncoder (from public notebooks)
    ordinal_encoders = {}
    for col in categorical_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        combined_df[col] = oe.fit_transform(combined_df[[col]])
        ordinal_encoders[col] = oe

    # Feature Engineering: Interaction term (Time_Social_Interaction)
    combined_df["Time_Social_Interaction"] = combined_df["Time_spent_Alone"] * combined_df["Social_event_attendance"]

    # Feature Engineering: Polynomial features on Friends_circle_size
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(combined_df[["Friends_circle_size"]])
    poly_feature_names = poly.get_feature_names_out(["Friends_circle_size"])
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    
    # Drop the original 'Friends_circle_size' column from poly_df if it's duplicated
    if 'Friends_circle_size' in poly_df.columns:
        poly_df = poly_df.drop(columns=['Friends_circle_size'])

    combined_df = pd.concat([combined_df, poly_df], axis=1)

    # Separate back into train and test
    X_train = combined_df.iloc[:len(train_df)].copy()
    X_test = combined_df.iloc[len(train_df):].copy()

    # Encode the target variable 'Personality'
    le_personality = LabelEncoder()
    y_train_encoded = le_personality.fit_transform(train_df["Personality"])

    return X_train, X_test, y_train_encoded, le_personality

X_train, X_test, y_train_encoded, le_personality = preprocess_and_feature_engineer(train_df, test_df)


def train_ensemble_model(X_train, y_train_encoded):
    # Optuna objective function for hyperparameter tuning of base models
    def objective(trial):
        classifier_name = trial.suggest_categorical("classifier", ["RandomForest", "XGBoost", "LGBM", "CatBoost", "LogisticRegression"])

        if classifier_name == "RandomForest":
            n_estimators = trial.suggest_int("rf_n_estimators", 50, 1000)
            max_depth = trial.suggest_int("rf_max_depth", 2, 16, log=True)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif classifier_name == "XGBoost":
            n_estimators = trial.suggest_int("xgb_n_estimators", 50, 1000)
            learning_rate = trial.suggest_float("xgb_learning_rate", 1e-3, 0.5, log=True)
            max_depth = trial.suggest_int("xgb_max_depth", 2, 8)
            subsample = trial.suggest_float("xgb_subsample", 0.6, 1.0)
            colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.2, 1.0)
            model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree, use_label_encoder=False, eval_metric="logloss", random_state=42)
        elif classifier_name == "LGBM":
            n_estimators = trial.suggest_int("lgbm_n_estimators", 50, 1000)
            learning_rate = trial.suggest_float("lgbm_learning_rate", 1e-3, 0.5, log=True)
            num_leaves = trial.suggest_int("lgbm_num_leaves", 2, 32)
            model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, random_state=42)
        elif classifier_name == "CatBoost":
            iterations = trial.suggest_int("cat_iterations", 50, 1000)
            learning_rate = trial.suggest_float("cat_learning_rate", 1e-3, 0.5, log=True)
            depth = trial.suggest_int("cat_depth", 2, 8)
            model = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=0, random_state=42)
        else: # LogisticRegression
            C = trial.suggest_float("lr_C", 1e-3, 1e3, log=True)
            model = LogisticRegression(C=C, solver="liblinear", random_state=42)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Reduced n_splits for faster tuning
        accuracy = cross_val_score(model, X_train, y_train_encoded, cv=kf, scoring="accuracy").mean()
        return accuracy

    # Run Optuna study for base models
    print("Running Optuna study for base models...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100) # Reduced trials for faster tuning
    print("Best trial for base models:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Define base models with best hyperparameters found by Optuna
    # For simplicity and speed, we will use the best single model found by Optuna and add a few strong defaults
    base_models = []
    if study.best_params["classifier"] == "RandomForest":
        base_models.append(("rf", RandomForestClassifier(n_estimators=study.best_params["rf_n_estimators"], max_depth=study.best_params["rf_max_depth"], random_state=42)))
    elif study.best_params["classifier"] == "XGBoost":
        base_models.append(("xgb", XGBClassifier(n_estimators=study.best_params["xgb_n_estimators"], learning_rate=study.best_params["xgb_learning_rate"], max_depth=study.best_params["xgb_max_depth"], subsample=study.best_params["xgb_subsample"], colsample_bytree=study.best_params["xgb_colsample_bytree"], use_label_encoder=False, eval_metric="logloss", random_state=42)))
    elif study.best_params["classifier"] == "LGBM":
        base_models.append(("lgbm", LGBMClassifier(n_estimators=study.best_params["lgbm_n_estimators"], learning_rate=study.best_params["lgbm_learning_rate"], num_leaves=study.best_params["lgbm_num_leaves"], random_state=42)))
    elif study.best_params["classifier"] == "CatBoost":
        base_models.append(("catb", CatBoostClassifier(iterations=study.best_params["cat_iterations"], learning_rate=study.best_params["cat_learning_rate"], depth=study.best_params["cat_depth"], verbose=0, random_state=42)))
    else: # LogisticRegression
        base_models.append(("lr", LogisticRegression(C=study.best_params["lr_C"], solver="liblinear", random_state=42)))

    # Add other strong models to the ensemble (using default params for now, can be tuned further)
    base_models.extend([
        ("xgb_", XGBClassifier(random_state=42, n_estimators= 959, learning_rate=0.0036634229959189033, max_depth=3, subsample=0.6808948372733825, colsample_bytree=0.875748751926246)),
        ("lgbm_default", LGBMClassifier(random_state=42, n_estimators=523, learning_rate=0.006798456284169054, num_leaves=28)),
        ("catb_default", CatBoostClassifier(random_state=42, verbose=0, iterations=990, learning_rate=0.0053882010617916035, depth=5)),
        ("lr_default", RandomForestClassifier(random_state=42, n_estimators=100, max_depth=7))
    ])

    # Define meta-model
    stk_model = StackingClassifier(
        estimators=base_models,
        final_estimator=XGBClassifier(random_state=42, n_estimators= 959, learning_rate=0.0036634229959189033, max_depth=3, subsample=0.6808948372733825, colsample_bytree=0.875748751926246),
        cv=5  # Reduced n_splits for faster stacking CV
    )

    print("\n--- Training Stacking Ensemble Model ---")
    stk_model.fit(X_train, y_train_encoded)
    print("Stacking Ensemble Model trained successfully.")

    # Perform Stratified K-Fold Cross-Validation on the final stacking model
    print("\n--- Performing Stratified K-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train_encoded)):
        print(f"\n--- Fold {fold+1} ---")
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train_encoded[train_index], y_train_encoded[val_index]
        
        # Retrain the stacking model on each fold
        fold_stk_model = StackingClassifier(
            estimators=base_models,
            final_estimator=RandomForestClassifier(random_state=42, n_estimators=100, max_depth=7),
            cv=5  # Use 3-fold cross-validation for stacking during fold training
        )
        fold_stk_model.fit(X_train_fold, y_train_fold)
        y_pred_val = fold_stk_model.predict(X_val_fold)
        fold_accuracy = accuracy_score(y_val_fold, y_pred_val)
        fold_accuracies.append(fold_accuracy)
        print(f"Fold {fold+1} Validation Accuracy: {fold_accuracy:.4f}")

    print(f"\nAverage Stratified K-Fold Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")

    return stk_model

final_ensemble_model = train_ensemble_model(X_train, y_train_encoded)


add_model = XGBClassifier(random_state=42, n_estimators= 959, learning_rate=0.0036634229959189033, max_depth=3, subsample=0.6808948372733825, colsample_bytree=0.875748751926246)
add_model.fit(X_train, y_train_encoded)
predicts_add = add_model.predict(X_test)
predicts_add = le_personality.inverse_transform(predicts_add)
alt_submission_df = pd.DataFrame({
    "id": sample_submission_df["id"],
    "Personality": predicts_add
})

# Save the submission file
alt_submission_df.to_csv("alt_submission.csv", index=False)


# For simplicity, just saving classes for now
with open("personality_classes.txt", "w") as f:
    for cls in le_personality.classes_:
        f.write(f"{cls}\n")
personality_classes = []
with open("personality_classes.txt", "r") as f:
    for line in f:
        personality_classes.append(line.strip())
le_personality.fit(personality_classes)

# Make predictions on the test data
predictions_encoded = final_ensemble_model.predict(X_test)

# Inverse transform predictions to original labels
predictions = le_personality.inverse_transform(predictions_encoded)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    "id": sample_submission_df["id"],
    "Personality": predictions
})

# Save the submission file
submission_df.to_csv("submission.csv", index=False)


sub_overfit = pd.read_csv('/kaggle/input/submission-overfit-private/submission_overfitted_poly.csv')


sub_overfit.to_csv('submission_overfit.csv', index=False)

