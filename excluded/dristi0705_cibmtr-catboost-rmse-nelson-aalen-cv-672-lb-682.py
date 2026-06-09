# Install required libraries
! pip install lifelines
! pip install scikit-survival

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lifelines for survival analysis
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter

# Preprocessing and feature transformation
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA

# Model selection and evaluation
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Survival models and metrics
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# Miscellaneous
from scipy.stats import rankdata

# Visualization settings (optional)
sns.set(style="whitegrid")


# Load datasets
train_data_path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
test_data_path = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

### Enhanced EDA for Survival Analysis ###
def survival_analysis_eda(data, duration_col="efs_time", event_col="efs"):
    """
    Perform preliminary and advanced EDA for survival analysis.

    Args:
        data (pd.DataFrame): The dataset containing survival data.
        duration_col (str): Column representing survival time.
        event_col (str): Column representing event occurrence (1 for event, 0 for censored).
    """
    if data.empty:
        print("The dataset is empty. Please check your input.")
        return

    # Dataset overview
    print("\n### Dataset Overview ###")
    print(data.info())
    print("\n### Missing Values ###")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])

    # Visualize missing values
    plt.figure(figsize=(8, 6))
    missing_values[missing_values > 0].sort_values(ascending=False).plot(kind="bar", color="skyblue")
    plt.title("Missing Values per Column")
    plt.ylabel("Count")
    plt.show()

    print("\n### Unique Values per Column ###")
    print(data.nunique())

    # Distribution of survival times
    plt.figure(figsize=(8, 6))
    sns.histplot(data[duration_col].dropna(), kde=True, bins=30, color="blue")
    plt.title(f"Distribution of {duration_col}")
    plt.xlabel("Survival Time")
    plt.ylabel("Frequency")
    plt.show()

    # Kaplan-Meier survival curve for the overall dataset
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(data[duration_col], event_observed=data[event_col])
        plt.figure(figsize=(8, 6))
        kmf.plot_survival_function()
        plt.title("Kaplan-Meier Survival Curve (Overall)")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.show()
    except Exception as e:
        print(f"Error in Kaplan-Meier Curve (Overall): {e}")

    # Kaplan-Meier survival curves by categorical features
    categorical_features = data.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_features:
        if data[col].nunique() <= 5:  # Focus on features with fewer categories
            plt.figure(figsize=(8, 6))
            for category in data[col].dropna().unique():
                mask = data[col] == category
                try:
                    kmf.fit(data[duration_col][mask], event_observed=data[event_col][mask], label=str(category))
                    kmf.plot_survival_function()
                except Exception as e:
                    print(f"Error plotting Kaplan-Meier curve for {col}={category}: {e}")
            plt.title(f"Kaplan-Meier Survival Curves by {col}")
            plt.xlabel("Time")
            plt.ylabel("Survival Probability")
            plt.legend(title=col)
            plt.show()

    # Correlation analysis for numerical features
    numerical_features = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if event_col in numerical_features:
        numerical_features.remove(event_col)
    if duration_col in numerical_features:
        numerical_features.remove(duration_col)

    if numerical_features:
        correlation_matrix = data[numerical_features].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={"shrink": 0.8})
        plt.title("Correlation Matrix (Numerical Features)")
        plt.show()
    else:
        print("\nNo numerical features available for correlation analysis.")

# Perform EDA
print("### Preliminary and Advanced EDA on Train Dataset ###")
survival_analysis_eda(train_data, duration_col="efs_time", event_col="efs")


assert "efs" in train_data.columns, "The event column ('efs') is missing."
assert "efs_time" in train_data.columns, "The duration column ('efs_time') is missing."


def feature_engineering(
    df, essential_columns=["race_group", "efs", "efs_time"], current_year=2024, is_train=True
):
    """
    Engineer features for survival analysis.
    Essential columns remain intact, transformations are minimal, and features are categorized for CatBoost.
    """
    # Set ID as index to avoid it being treated as a feature
    if "ID" in df.columns:
        df = df.set_index("ID")

    # Preserve essential columns
    preserved_data = df[essential_columns].copy() if is_train else df[["race_group"]].copy()

    # Dynamically identify categorical features (object or string types)
    cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Treat missing values in categorical features
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")  # Replace missing values with 'Unknown'

    # # Time-based features
    # if "year_hct" in df.columns:
    #     df["time_since_transplant"] = current_year - df["year_hct"]

    # # Interaction features
    # if "age_at_hct" in df.columns and "donor_age" in df.columns:
    #     df["age_difference"] = df["age_at_hct"] - df["donor_age"]

    # Dynamically identify numerical features (int64 and float64 types)
    num_features = df.select_dtypes(include=["int64", "float64"]).columns.difference(
        essential_columns + ["ID"]
    ).tolist()

    # Add preserved columns back
    for col in preserved_data.columns:
        df[col] = preserved_data[col]

    return df, cat_features, num_features


# Apply feature engineering to train and test datasets
engineered_train_data, train_cat_features, train_num_features = feature_engineering(train_data, is_train=True)
engineered_test_data, test_cat_features, test_num_features = feature_engineering(test_data, is_train=False)

# Check essential columns
for col in ["race_group", "efs", "efs_time"]:
    if col not in engineered_train_data.columns:
        raise KeyError(f"The column '{col}' is missing after feature engineering.")

print("Feature engineering completed successfully.")
print(f"Categorical features: {train_cat_features}")
print(f"Numerical features: {train_num_features}")


engineered_test_data["race_group"].unique()


# Encode 'race_group' in the train and test data
train_data = engineered_train_data.copy()
test_data = engineered_test_data.copy()


import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np


def derive_nelson_aalen_risk_index(data, event_label="efs", time_label="efs_time"):
    """
    Compute Nelson-Aalen-based cumulative hazard risk indices for the dataset.
    """
    naf = NelsonAalenFitter()
    data = data.reset_index(drop=True)  # Reset indices to ensure alignment

    # Ensure required columns exist
    assert event_label in data.columns, f"Column '{event_label}' not found in data."
    assert time_label in data.columns, f"Column '{time_label}' not found in data."

    # Fit Nelson-Aalen model
    naf.fit(durations=data[time_label].to_numpy(), event_observed=data[event_label].to_numpy())

    # Risk index is -cumulative hazard at each survival time
    data["risk_index"] = -naf.cumulative_hazard_at_times(data[time_label].to_numpy()).values
    return data


def group_based_c_index(solution, submission, race_column="race_group", prediction_label="prediction"):
    """
    Compute the group-based concordance index (mean - sqrt(variance)) across race groups.
    """
    # Merging solution and submission
    merged_df = pd.concat([solution.reset_index(drop=True), submission.reset_index(drop=True)], axis=1)
    group_indices = merged_df.groupby(race_column).groups

    c_indices = []
    for group, indices in group_indices.items():
        group_data = merged_df.loc[indices]
        c_index = concordance_index(
            event_times=group_data["efs_time"],
            predicted_scores=-group_data[prediction_label],
            event_observed=group_data["efs"]
        )
        c_indices.append(c_index)

    # Compute the mean - sqrt(variance)
    return float(np.mean(c_indices) - np.sqrt(np.var(c_indices)))


# def objective(trial, X_train, y_train, cat_features):
#     """
#     Objective function for Optuna optimization with extended parameters.
#     """
#     param = {
#         'iterations': trial.suggest_int('iterations', 100, 2000),
#         'depth': trial.suggest_int('depth', 4, 12),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0),
#         'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
#         'subsample': trial.suggest_float('subsample', 0.7, 1.0),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
#         'max_bin': trial.suggest_int('max_bin', 16, 256),
#         'loss_function': 'RMSE',
#         'task_type': 'CPU',
#         'verbose': 100
#     }

#     # Initialize CatBoostRegressor
#     model = CatBoostRegressor(**param)

#     # Cross-validation setup
#     kf = KFold(n_splits=10, shuffle=True, random_state=42)
#     c_index_scores = []

#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
#         # Split data
#         X_train_fold = X_train.iloc[train_idx].reset_index(drop=True)
#         X_val_fold = X_train.iloc[val_idx].reset_index(drop=True)

#         y_train_fold = y_train.iloc[train_idx].reset_index(drop=True)
#         y_val_fold = y_train.iloc[val_idx].reset_index(drop=True)

#         # Fit the model
#         model.fit(
#             X_train_fold, y_train_fold["risk_index"],
#             cat_features=cat_features, eval_set=[(X_val_fold, y_val_fold["risk_index"])],
#             use_best_model=True
#         )

#         # Prepare solution and submission DataFrames
#         val_solution = y_val_fold[["efs", "efs_time", "race_group"]].copy()
#         val_submission = pd.DataFrame({
#             "prediction": model.predict(X_val_fold)
#         })

#         # Compute the group-based C-index
#         c_index = group_based_c_index(
#             solution=val_solution,
#             submission=val_submission,
#             race_column="race_group",
#             prediction_label="prediction"
#         )

#         # Log fold-level results
#         print(f"Trial {trial.number}, Fold {fold + 1}: C-index = {c_index:.4f}")

#         c_index_scores.append(c_index)

#     # Log average C-index for the trial
#     mean_c_index = np.mean(c_index_scores)
#     print(f"Trial {trial.number}: Mean C-index = {mean_c_index:.4f}\n")
#     return mean_c_index


# def tune_catboost(train_data, event_label="efs", time_label="efs_time"):
#     """
#     Train and optimize CatBoost for the entire dataset.
#     """
#     # Derive Nelson-Aalen risk indices
#     train_data = derive_nelson_aalen_risk_index(train_data, event_label=event_label, time_label=time_label)

#     # Slice X_train and y_train
#     y_train = train_data[[event_label, time_label, "race_group"]].assign(risk_index=train_data["risk_index"])
#     X_train = train_data.drop(columns=[event_label, time_label, "risk_index"])

#     # Identify categorical features dynamically
#     cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

#     # Define Optuna study
#     study = optuna.create_study(direction="maximize")
#     study.optimize(lambda trial: objective(trial, X_train, y_train, cat_features), n_trials=50)

#     # Output best results
#     best_params = study.best_params
#     best_score = study.best_value

#     print(f"Best Parameters: {best_params}")
#     print(f"Best C-index Score: {best_score}")

#     return best_params, best_score


# # Example Usage
# best_params, best_score = tune_catboost(train_data)

# # Print Results
# print("Model training completed.")
# print(f"Best Parameters: {best_params}")
# print(f"Best C-index Score: {best_score}")


# from sklearn.model_selection import train_test_split
# from catboost import CatBoostRegressor
# from lifelines.utils import concordance_index
# import pandas as pd
# import numpy as np

# # Define best parameters
# best_params_final = {
#                      'iterations': 1225,
#                      'depth': 7,
#                      'learning_rate': 0.04990289157255559,
#                      'l2_leaf_reg': 16.951151720446834,
#                      'random_strength': 3.1791047479362367,
#                      'bagging_temperature': 1.4306949347002924,
#                      'subsample': 0.7447947581252647,
#                      'min_data_in_leaf': 1,
#                      'max_bin': 122
#                     }


# def evaluate_catboost(train_data, best_params, event_label="efs", time_label="efs_time", test_size=0.2, random_state=42):
#     """
#     Train CatBoost with the best parameters and evaluate its performance using a holdout validation set.
#     """
#     # Derive Kaplan-Meier risk indices
#     train_data = derive_nelson_aalen_risk_index(train_data, event_label=event_label, time_label=time_label)

#     # Split into features and target
#     y = train_data[[event_label, time_label, "race_group"]].assign(risk_index=train_data["risk_index"])
#     X = train_data.drop(columns=[event_label, time_label, "risk_index"])

#     # Identify categorical features dynamically
#     cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

#     # Train-validation split
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y["race_group"]
#     )

#     # Initialize CatBoostRegressor with the best parameters
#     model = CatBoostRegressor(
#         iterations=best_params['iterations'],
#         depth=best_params['depth'],
#         learning_rate=best_params['learning_rate'],
#         l2_leaf_reg=best_params['l2_leaf_reg'],
#         random_strength=best_params['random_strength'],
#         bagging_temperature=best_params['bagging_temperature'],
#         subsample=best_params['subsample'],
#         min_data_in_leaf=best_params['min_data_in_leaf'],
#         max_bin=best_params['max_bin'],
#         loss_function='RMSE',
#         task_type='CPU',
#         verbose=100
#     )

#     # Fit the model
#     model.fit(
#         X_train, y_train["risk_index"],
#         cat_features=cat_features,
#         eval_set=(X_val, y_val["risk_index"]),
#         use_best_model=True
#     )

#     # Create a new DataFrame for predictions to ensure no conflicts with y_val
#     val_predictions = pd.DataFrame({
#         "race_group": y_val["race_group"],
#         "efs_time": y_val["efs_time"],
#         "efs": y_val["efs"],
#         "prediction": model.predict(X_val)
#     }).reset_index(drop=True)

#     # Compute group-based C-index
#     final_c_index = group_based_c_index(
#         solution=val_predictions[["efs", "efs_time", "race_group"]],
#         submission=val_predictions[["prediction"]],
#         race_column="race_group",
#         prediction_label="prediction"
#     )

#     print(f"Final C-index Score: {final_c_index}")
#     return model, final_c_index


# # Train and evaluate the model using a holdout validation set
# final_model, final_score = evaluate_catboost(train_data, best_params_final)

# # Print final results
# print("Model Training and Evaluation Completed.")
# print(f"Final C-index Score: {final_score}")


# from sklearn.model_selection import KFold
# from catboost import CatBoostRegressor
# from lifelines.utils import concordance_index
# import pandas as pd
# import numpy as np

# # Define best parameters
# best_params_final = {
#                      'iterations': 1225,
#                      'depth': 7,
#                      'learning_rate': 0.04990289157255559,
#                      'l2_leaf_reg': 16.951151720446834,
#                      'random_strength': 3.1791047479362367,
#                      'bagging_temperature': 1.4306949347002924,
#                      'subsample': 0.7447947581252647,
#                      'min_data_in_leaf': 1,
#                      'max_bin': 122
#                     }


# def kfold_evaluate_catboost(train_data, best_params, event_label="efs", time_label="efs_time", n_splits=5):
#     """
#     Perform K-fold cross-validation for CatBoost and evaluate the mean C-index across folds.
#     """
#     # Derive Kaplan-Meier risk indices
#     train_data = derive_nelson_aalen_risk_index(train_data, event_label=event_label, time_label=time_label)

#     # Split into features and target
#     y = train_data[[event_label, time_label, "race_group"]].assign(risk_index=train_data["risk_index"])
#     X = train_data.drop(columns=[event_label, time_label, "risk_index"])

#     # Identify categorical features dynamically
#     cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

#     # Initialize KFold
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     c_index_scores = []

#     for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
#         print(f"\nStarting Fold {fold}/{n_splits}")

#         # Split data into training and validation sets
#         X_train, X_val = X.iloc[train_idx].reset_index(drop=True), X.iloc[val_idx].reset_index(drop=True)
#         y_train, y_val = y.iloc[train_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)

#         # Initialize CatBoostRegressor with the best parameters
#         model = CatBoostRegressor(
#                     iterations=best_params['iterations'],
#                     depth=best_params['depth'],
#                     learning_rate=best_params['learning_rate'],
#                     l2_leaf_reg=best_params['l2_leaf_reg'],
#                     random_strength=best_params['random_strength'],
#                     bagging_temperature=best_params['bagging_temperature'],
#                     subsample=best_params['subsample'],
#                     min_data_in_leaf=best_params['min_data_in_leaf'],
#                     max_bin=best_params['max_bin'],
#                     loss_function='RMSE',
#                     task_type='CPU',
#                     verbose=100
#                 )

#         # Fit the model
#         print(f"Training Fold {fold}...")
#         model.fit(
#             X_train, y_train["risk_index"],
#             cat_features=cat_features,
#             eval_set=(X_val, y_val["risk_index"]),
#             use_best_model=True
#         )

#         # Create a separate DataFrame for predictions
#         val_predictions = pd.DataFrame({
#             "race_group": y_val["race_group"],
#             "efs_time": y_val["efs_time"],
#             "efs": y_val["efs"],
#             "prediction": model.predict(X_val)
#         }).reset_index(drop=True)

#         # Compute group-based C-index for the validation fold
#         c_index = group_based_c_index(
#             solution=val_predictions[["efs", "efs_time", "race_group"]],
#             submission=val_predictions[["prediction"]],
#             race_column="race_group",
#             prediction_label="prediction"
#         )

#         print(f"Fold {fold} C-index: {c_index:.4f}")
#         c_index_scores.append(c_index)

#     # Compute mean and standard deviation of C-index across folds
#     mean_c_index = np.mean(c_index_scores)
#     std_c_index = np.std(c_index_scores)

#     print(f"\nMean C-index: {mean_c_index:.4f}")
#     print(f"Standard Deviation of C-index: {std_c_index:.4f}")
#     return mean_c_index, std_c_index


# # Perform K-fold cross-validation
# mean_c_index, std_c_index = kfold_evaluate_catboost(train_data, best_params_final)

# # Print final results
# print("\nK-Fold Cross-Validation Completed.")
# print(f"Mean C-index Score: {mean_c_index}")
# print(f"Standard Deviation of C-index: {std_c_index}")


from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np

# Define best parameters
best_params_final = {
                     'iterations': 1225,
                     'depth': 7,
                     'learning_rate': 0.04990289157255559,
                     'l2_leaf_reg': 16.951151720446834,
                     'random_strength': 3.1791047479362367,
                     'bagging_temperature': 1.4306949347002924,
                     'subsample': 0.7447947581252647,
                     'min_data_in_leaf': 1,
                     'max_bin': 122
                    }


def fit_catboost_and_predict(train_data, test_data, best_params, event_label="efs", time_label="efs_time", test_size=0.2, random_state=42):
    """
    Train CatBoost with the best parameters, evaluate using validation set, and predict on the full dataset.
    """
    # Derive Kaplan-Meier risk indices
    train_data = derive_nelson_aalen_risk_index(train_data, event_label=event_label, time_label=time_label)

    # Split into features and target
    y = train_data[[event_label, time_label, "race_group"]].assign(risk_index=train_data["risk_index"])
    X = train_data.drop(columns=[event_label, time_label, "risk_index"])

    # Identify categorical features dynamically
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y["race_group"]
    )

    # Initialize CatBoostRegressor with the best parameters
    model = CatBoostRegressor(
                    iterations=best_params['iterations'],
                    depth=best_params['depth'],
                    learning_rate=best_params['learning_rate'],
                    l2_leaf_reg=best_params['l2_leaf_reg'],
                    random_strength=best_params['random_strength'],
                    bagging_temperature=best_params['bagging_temperature'],
                    subsample=best_params['subsample'],
                    min_data_in_leaf=best_params['min_data_in_leaf'],
                    max_bin=best_params['max_bin'],
                    loss_function='RMSE',
                    task_type='CPU',
                    verbose=100
                )

    # Fit the model
    model.fit(
        X_train, y_train["risk_index"],
        cat_features=cat_features,
        eval_set=(X_val, y_val["risk_index"]),
        use_best_model=True
    )

    # Predict on the full train dataset
    train_predictions = model.predict(X)

    # Ensure `ID` is the index for predictions
    train_data_predictions = pd.Series(train_predictions, index=train_data.index, name="train_predictions")

    # Predict on the test dataset
    if "ID" in test_data.columns:
        test_data = test_data.set_index("ID")
    test_predictions = model.predict(test_data)
    test_data_predictions = pd.Series(test_predictions, index=test_data.index, name="test_predictions")

    print("Predictions completed on train and test datasets.")
    return model, train_data_predictions, test_data_predictions


# Example usage
# Replace 'train_data' and 'test_data' with your actual DataFrames
final_model, train_preds, test_preds = fit_catboost_and_predict(train_data, test_data, best_params_final)

# Print summary
print("Model Training and Predictions Completed.")
print("Train Predictions and Test Predictions saved.")


train_preds


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter
# from sksurv.ensemble import RandomSurvivalForest
# from sksurv.metrics import concordance_index_censored
# from sklearn.model_selection import KFold, GridSearchCV
# from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from scipy.stats import rankdata

# # Preprocess train and test data
# def preprocess_data(train_data, test_data, essential_columns=["efs", "efs_time", "race_group"]):
#     """
#     Preprocess the train and test data, ensuring that the essential columns are preserved and that
#     categorical and numerical columns are handled properly. The 'race_group' is not used in PCA.
#     """
#     # Check that essential columns exist in the train dataset
#     missing_columns = [col for col in essential_columns if col not in train_data.columns]
#     if missing_columns:
#         raise KeyError(f"Essential columns missing from the train dataset: {missing_columns}")
    
#     # Handle categorical columns (excluding essential columns like race_group, efs, efs_time)
#     categorical_columns = train_data.select_dtypes(include=["object"]).columns
#     categorical_columns = [col for col in categorical_columns if col not in essential_columns]  # Exclude essential columns
#     numerical_columns = train_data.select_dtypes(include=["float64", "int64"]).columns
#     numerical_columns = [col for col in numerical_columns if col not in essential_columns]  # Exclude essential columns

#     # Preprocess categorical data (Do not encode 'race_group')
#     for col in categorical_columns:
#         train_data[col] = train_data[col].astype(str).fillna("Missing")
#         if col in test_data.columns:
#             test_data[col] = test_data[col].astype(str).fillna("Missing")
#         train_data[col] = train_data[col].factorize()[0]  # Encode as integers
#         if col in test_data.columns:
#             test_data[col] = test_data[col].map(
#                 dict(zip(train_data[col].unique(), range(len(train_data[col].unique()))))
#             ).fillna(-1)

#     # Handle numerical columns (excluding the target columns)
#     train_data[numerical_columns] = train_data[numerical_columns].fillna(
#         train_data[numerical_columns].mean()
#     )
#     if set(numerical_columns).intersection(test_data.columns):
#         test_data[numerical_columns] = test_data[numerical_columns].fillna(
#             train_data[numerical_columns].mean()
#         )

#     # One-hot encoding for categorical features (excluding essential columns like race_group)
#     train_data = pd.get_dummies(train_data, drop_first=True)
#     test_data = pd.get_dummies(test_data, drop_first=True)

#     # After one-hot encoding, add the 'race_group' column back to both datasets
#     if 'race_group' in train_data.columns:
#         train_data = train_data.drop(columns='race_group', errors='ignore')  # Drop if it exists to prevent duplicates
#     train_data['race_group'] = engineered_train_data['race_group']  # Add 'race_group' back to the train data

#     if 'race_group' in test_data.columns:
#         test_data = test_data.drop(columns='race_group', errors='ignore')  # Drop if it exists to prevent duplicates
#     test_data['race_group'] = engineered_test_data['race_group']  # Add 'race_group' back to the test data
    
#     # Ensure both datasets have the same columns after one-hot encoding
#     missing_cols = set(train_data.columns) - set(test_data.columns)
#     for col in missing_cols:
#         test_data[col] = 0
#     test_data = test_data[train_data.columns]  # Ensure the order of columns matches

#     return train_data, test_data, numerical_columns, categorical_columns  # Return numerical columns as well

# # Apply preprocessing to train and test data
# train_data = engineered_train_data.copy()
# test_data = engineered_test_data.copy()

# # Apply preprocessing
# train_data, test_data, numerical_columns, categorical_columns = preprocess_data(train_data, test_data)
# print("Data preprocessing completed successfully.")


# train_data["race_group"].unique()


# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # Apply PCA to the numerical features excluding 'race_group' and target columns
# def apply_pca(X, numerical_columns, variance_threshold=0.95):
#     """
#     Apply PCA to reduce dimensionality for linear models, excluding 'race_group' and target columns.
#     """
#     # Use the numerical columns provided from the previous step
#     X_numerical = X[numerical_columns]
    
#     # Standardizing the data before PCA
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_numerical)
    
#     pca = PCA(n_components=variance_threshold)
#     X_pca = pca.fit_transform(X_scaled)
    
#     print(f"Number of PCA components retained: {pca.n_components_}")
    
#     # Convert PCA result to DataFrame with appropriate column names
#     pca_columns = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
#     X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    
#     return X_pca_df, pca

# # Apply PCA only to the features excluding 'race_group' and target columns
# X_train_pca_df, pca_model = apply_pca(train_data, numerical_columns)

# # Ensure that the test data has the same number of features as the training data before applying PCA
# test_data_no_target = test_data.drop(columns=["efs", "efs_time", "race_group"], errors="ignore")  # Drop target and race_group columns

# # Align columns of test data to the training data (handle missing columns in the test set)
# test_data_no_target = test_data_no_target[numerical_columns]  # Keep only numerical columns for PCA

# # Apply PCA transformation to test data
# X_test_pca = pca_model.transform(test_data_no_target)

# # Convert PCA result to DataFrame
# X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f"pca_{i+1}" for i in range(X_test_pca.shape[1])])

# # Remove target columns and race_group from train_data and combine with PCA results
# train_data_no_target = train_data.drop(columns=["efs", "efs_time", "race_group"])
# test_data_no_target = test_data.drop(columns=["efs", "efs_time", "race_group"])

# # Combine PCA features with other columns (excluding race_group and target columns)
# train_data_combined = pd.concat([train_data_no_target, X_train_pca_df, train_data[["efs", "efs_time", "race_group"]]], axis=1)
# test_data_combined = pd.concat([test_data_no_target, X_test_pca_df, test_data[["race_group"]]], axis=1)

# # Ensure that the essential columns are intact in the train data (test data will not have these columns)
# for col in ["efs", "efs_time"]:
#     if col not in train_data_combined.columns:
#         raise KeyError(f"The column '{col}' is missing after preprocessing.")

# print("PCA integration completed successfully.")


# train_data_combined["race_group"].unique()


# # After imputation, check if there are any remaining NaN values and print only the columns with missing values
# print("Missing values after imputation:")

# # For train data, print columns with missing values
# train_missing_columns = train_data_combined.columns[train_data_combined.isnull().any()]
# print(f"Train data missing values in columns:\n{train_missing_columns}")

# # For test data, print columns with missing values
# test_missing_columns = test_data_combined.columns[test_data_combined.isnull().any()]
# print(f"Test data missing values in columns:\n{test_missing_columns}")


# # For train data, print columns with missing values
# for c in train_data_combined.columns:
#     print(c,train_data_combined[c].isnull().sum())



# Save final predictions
final_predictions = pd.DataFrame(
    {
        "ID": test_preds.index,
        "prediction": test_preds,
    }
)
final_predictions.to_csv("submission.csv", index=False)
print("Final predictions saved to 'final_predictions.csv'.")





