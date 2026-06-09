# IMPORT BASE LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Notebook Settings
import warnings
warnings.filterwarnings('ignore')


# IMPORT DATASETS (TRAIN & TEST)
train_data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


# Considering the train dataset...
print("\n>>>>>>>>>>>>>>>>>>>>>THE FRIST FIVE INSTANCES IN TRAIN<<<<<<<<<<<<<<<<<<<<<\n")
display(train_data.head())
print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>TRAIN DATA INFORMATION<<<<<<<<<<<<<<<<<<<<<<<<<\n")
display(train_data.info())
print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>TRAIN DATA SUMMARY STATS<<<<<<<<<<<<<<<<<<<<<\n")
display(train_data.describe())



# Considering the test data...
print("\n>>>>>>>>>>>>>>>>>>>>>THE FRIST FIVE INSTANCES IN TEST<<<<<<<<<<<<<<<<<<<<<\n")
display(test_data.head())
print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>TEST DATA INFORMATION<<<<<<<<<<<<<<<<<<<<<<<<<\n")
display(test_data.info())
print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>TEST DATA SUMMARY STATS<<<<<<<<<<<<<<<<<<<<<\n")
display(test_data.describe())


# ALL THE EDA IS CARRIED OUT ON THE TRAINING DATASET

# Drop id variable
train_data_v2 = train_data.copy()
train_data_v2.drop(["id"], axis=1, inplace=True)


# Understanding the distribution of numerical variables

# Helper function to do that
def plot_hist_box(df, target_col):
    col_name_list = df.select_dtypes(include=np.number).columns.tolist()
    # Create plot canvas
    fig, axes = plt.subplots(int(len(col_name_list)), 2, figsize=(15, 20))
    # axes = axes.flatten()
    for i, col in tqdm(enumerate(col_name_list), total=len(col_name_list), desc="plotting"):
        sns.histplot(data=df, x=col, hue=target_col, kde=True, ax=axes[i, 0])
        sns.boxplot(data=df, y=col, x=target_col, ax=axes[i, 1])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)

plot_hist_box(train_data_v2, "Fertilizer Name")


# Correlation Matrix
sns.heatmap(
    train_data_v2.select_dtypes(include=np.number).corr(),
    annot=True,
    fmt=".4f"
)
plt.show()


# Barplots for categorical variables

# Helper function to do that
def plot_barplots(df):
    col_name_list = df.select_dtypes(exclude=np.number).columns.tolist()
    # Create plot canvas
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes = axes.flatten()
    for i, col in tqdm(enumerate(col_name_list), total=len(col_name_list), desc="plotting"):
        sns.countplot(data=df, x=col, ax=axes[i], palette="Spectral")
        axes[i].set_xticks(axes[i].get_xticks(), axes[i].get_xticklabels(), rotation=45, fontsize=14)

    plt.tight_layout()

plot_barplots(train_data_v2)


# Import necessary libraries for this section
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


X, y = train_data_v2.iloc[:, :8], train_data_v2.iloc[:, 8:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# numerical columns
num_vars = X_train.select_dtypes(include=np.number).columns.tolist()
# categorical columns
cat_vars = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Basic data transformation pipeline
data_pipeline = ColumnTransformer([
    ("num", StandardScaler(), num_vars),
    ("cat", OneHotEncoder(sparse=False), cat_vars)
])


X_train_v2 = data_pipeline.fit_transform(X_train)
X_test_v2 = data_pipeline.fit_transform(X_test)

y_train_v2 = OneHotEncoder(sparse=False).fit_transform(y_train)
y_test_v2 = OneHotEncoder(sparse=False).fit_transform(y_test)


# Transform the test dataset (for model inference)
test_data_v2 = data_pipeline.fit_transform(test_data)


# Importing libraries
import optuna
from catboost import CatBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold


def map3_score(predicted_top3: np.ndarray,   # shape = (n_val, 3), dtype = object or int
               y_true_fold: np.ndarray,      # shape = (n_val,)
              ) -> float:
    """
    predicted_top5[i] is a lengthâ€�3 array of labels (strings/ints) that your model thinks
    are most likely for sample i, ordered from most confident 3rd most confident.
    y_true_fold[i] is the single true label for sample i.
    We give credit = 1/rank if the true label is at position 'rank' in that topâ€�3 list;
    otherwise 0. Then we average over all i.
    """
    n_val = y_true_fold.shape[0]
    total_score = 0.0

    for i in range(n_val):
        true_label = y_true_fold[i]
        top3_preds = predicted_top3[i].tolist()  # convert row to a Python list

        try:
            # .index(...) returns 0-based position. Add +1 to get 1-based rank.
            rank = top3_preds.index(true_label) + 1
            if rank <= 3:
                total_score += 1.0 / rank
            # If rank > 3, that cannot happen here, because top3_preds has exactly 3 items.
        except ValueError:
            # true_label not in top-3  score += 0
            pass

    return total_score / n_val


def catboost_objective(trial):
    # Define the parameter space for catboost
    params = {
        'iterations': trial.suggest_int('iterations', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 5),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0, 2),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5),
        'auto_class_weights' : trial.suggest_categorical('auto_class_weights', ['Balanced', None]),
    }
    
    # DEFINE THE CROSS-VALIDATION LOOP
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    map3_scores = []

    # For each fold, slice X_train_v2 & y_train by the indices
    for train_index, val_index in cv.split(X_train_v2, y_train):
        x_train_fold = X_train_v2[train_index]
        x_val_fold = X_train_v2[val_index]

        y_train_fold = y_train.iloc[train_index].values
        y_val_fold = y_train.iloc[val_index].values

        model = CatBoostClassifier(
            **params,
            random_seed=42,
            thread_count=-1,
            verbose=False,
            task_type="GPU",
            devices="0"
        )
        
        # Fit the model
        model.fit(x_train_fold, y_train_fold, 
                  eval_set=(x_val_fold, y_val_fold),
                 early_stopping_rounds=150,
                 use_best_model=True,
                 verbose=False
                 )
        pred_proba = model.predict_proba(x_val_fold)
        
        top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1] # Align the first 3 proba indexes, starting from the highest proba
        class_labels = model.classes_ # Return class names in the target variable (types of fertilizers)
        top3_labels = class_labels[top3_index] # Return first-three predictions

        fold_map3 = map3_score(top3_labels, y_val_fold)
        map3_scores.append(fold_map3)
        mean_map3 = np.mean(map3_scores)

        return mean_map3


# study = optuna.create_study(direction="maximize", sampler=TPESampler(n_startup_trials=30, seed=42, multivariate=True))
# study.optimize(catboost_objective, n_trials=50, n_jobs=1)


# The best parameters

# catboost_best_params = study.best_params
# catboost_best_params


# Fit the model with the best parameters

# ğŸ§ª Best parameters obtained from the previous optuna job
catboost_best_params = {'iterations': 3820,
 'learning_rate': 0.2839466858475117,
 'depth': 3,
 'l2_leaf_reg': 4.245507965247879,
 'border_count': 169,
 'random_strength': 0.1416768708009795,
 'bagging_temperature': 0.10225652601468996,
 'auto_class_weights': None}

catboost_model = CatBoostClassifier(
     **catboost_best_params,
      random_seed=42,
      task_type="GPU",
      verbose=False
)

catboost_model.fit(X_train_v2, y_train) # Fit model to the training data
pred_proba = catboost_model.predict_proba(X_test_v2)

top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
class_labels = catboost_model.classes_
top3_labels = class_labels[top3_index]

map3 = map3_score(top3_labels, y_test.values)
print(f"MAP@3 Score: {map3}")


submission_probas = catboost_model.predict_proba(test_data_v2)

top3_index_subm = np.argsort(submission_probas, axis=1)[:, -3:][:, ::-1]
class_labels_subm = catboost_model.classes_
top3_labels_subm = class_labels_subm[top3_index_subm]

# Convert top 3 predictions to space-separated strings
preds_as_strings = [' '.join(labels) for labels in top3_labels_subm]

# Build the submission DataFrame
submission = pd.DataFrame({
    'id': test_data['id'].values,
    'Fertilizer Name': preds_as_strings
})
submission

submission.to_csv('/kaggle/working/submission.csv',index=False)


def xgboost_objective(trial):
    # Define xgboost's parameter space
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        'subsample': trial.suggest_float('subsample', 0, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 15_000),
    }

    # DEFINE A CROSS-VALIDATION LOOP
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []

    # Encord target labels to (1, 2, 3, 4, 5,...)
    lab_encoder = LabelEncoder()
    y_train_encoded = lab_encoder.fit_transform(y_train) 

    # For each fold, slice X_train_v2 & y_train_v2 by the indices
    for train_index, val_index in cv.split(X_train_v2, y_train_encoded):
        x_train_fold = X_train_v2[train_index]
        x_val_fold = X_train_v2[val_index]

        y_train_fold = y_train_encoded[train_index]
        y_val_fold = y_train_encoded[val_index]
        
        # The model
        model = xgb.XGBClassifier(
            **params,
            verbosity=0,
            objective='multi:softprob',
            enable_categorical=True,
            tree_method="gpu_hist", # Use the GPU
            predictor="gpu_predictor",
            n_jobs=-1,
            device="cuda",
            random_seed=42
        )
        
        # Fit the model
        model.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
              early_stopping_rounds=50, verbose=False)
        # Prediction probas
        pred_proba = model.predict_proba(x_val_fold)
        top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
        class_labs = model.classes_
        top3_labs = class_labs[top3_index] # Return the first-three predicted classes

        fold_map3 = map3_score(top3_labs, y_val_fold)
        map3_scores.append(fold_map3)
        mean_map3 = np.mean(map3_scores)

        return mean_map3  


# study_2 = optuna.create_study(direction="maximize", sampler=TPESampler(n_startup_trials=30, seed=42, multivariate=True))
# study_2.optimize(xgboost_objective, n_trials=50, n_jobs=-1)


# The best parameters for xgboost
# xgb_best_params = study_2.best_params
# xgb_best_params


# Encord target labels to (1, 2, 3, 4, 5,...)
lab_encoder_train = LabelEncoder() # for y_train
lab_encoder_test = LabelEncoder() # for y_test
y_train_encoded = lab_encoder_train.fit_transform(y_train) 
y_test_encoded = lab_encoder_test.fit_transform(y_test)

# ğŸ§ª Best parameters obtained from the previous optuna job
xgb_best_params = {'learning_rate': 0.0855138611826152,
 'max_depth': 305,
 'min_child_weight': 16,
 'gamma': 0.26409924765908566,
 'reg_alpha': 0.0021104826613006014,
 'reg_lambda': 0.08981634779333057,
 'subsample': 0.2653850247835302,
 'colsample_bytree': 0.1378307171917018,
 'n_estimators': 11930}

# XGB Model with the best params from optuna
xgb_model = xgb.XGBClassifier(
    **xgb_best_params,
    verbosity=0,
    objective='multi:softprob',
    enable_categorical=True,
    tree_method="gpu_hist",
    gpu_id=0, 
    n_jobs=-1,
    random_seed=42
)
xgb_model.fit(X_train_v2, y_train_encoded)
xgb_pred_proba = xgb_model.predict_proba(X_test_v2)

top3_index2 = np.argsort(xgb_pred_proba, axis=1)[:, -3:][:, ::-1]
class_labels2 = xgb_model.classes_
top3_labels2 = class_labels2[top3_index2]

xgb_map3 = map3_score(top3_labels2, y_test_encoded)
print(f"MAP@3 Score: {xgb_map3}")


# Returning prediction probabilies for each example in the test_data_v2
submission_probas_xgb = xgb_model.predict_proba(test_data_v2)
# Returning the indexes of top 3 probabilities for each class (from the largest)
top3_index_subm = np.argsort(submission_probas_xgb, axis=1)[:, -3:][:, ::-1]

decoded_flat_top3 = lab_encoder_train.inverse_transform(top3_index_subm.flatten())
decoded_flat_top3.reshape(top3_index_subm.shape)


# Convert top 3 predictions to space-separated strings
preds_as_strings = [' '.join(labels) for labels in decoded_flat_top3.reshape(top3_index_subm.shape)]
# Build the submission DataFrame
submission = pd.DataFrame({
    'id': test_data['id'].values,
    'Fertilizer Name': preds_as_strings
})
submission.to_csv('/kaggle/working/submission.csv',index=False)
display(submission)


from sklearn.ensemble import VotingClassifier

def ensemble_one(X_train, y_train, X_test, y_test, testing_data):
    # Best parameters for each model from previous jobs and other notebooks 
    catboost_params = {'iterations': 3820,
     'learning_rate': 0.2839466858475117,
     'depth': 4,
     'l2_leaf_reg': 4.245507965247879,
     'border_count': 169,
     'random_strength': 0.1416768708009795,
     'bagging_temperature': 0.10225652601468996,
     'auto_class_weights': None,
     'random_seed': 42,
     'thread_count': -1,
     'verbose': False,
     'task_type': "GPU",
     'devices': 'cuda'
    }
    lgbm_params = {
        'n_estimators': 1000, 
        'learning_rate': 0.05, 
        'max_depth': 16, 
        'device_type': 'gpu',
        'random_state': 42,
        'subsample': 0.686,
        'colsample_bytree': 0.25,
        'min_child_samples': 30,
        'num_leaves': 128,
        'boosting_type':'gbdt',
        'verbosity':-1
    }
    # Encord target labels to (1, 2, 3, 4, 5,...)
    lab_encoder_train = LabelEncoder() # for y_train
    lab_encoder_test = LabelEncoder() # for y_test
    y_train_encoded = lab_encoder_train.fit_transform(y_train) 
    y_test_encoded = lab_encoder_test.fit_transform(y_test)

    # Defining an ensemble of catboost and lgbm
    ens_model_1 = VotingClassifier(
        estimators = [
            ("CatBoost", CatBoostClassifier(**catboost_params,)),
            ("LightGBM", LGBMClassifier(**lgbm_params))
        ],
        voting="soft" # Return the average proba from CatBoost and LightGBM
    )
    
    ens_model_1.fit(X_train, y_train_encoded)
    ens_pred_proba = ens_model_1.predict_proba(X_test)
    
    # Returning the indexes of top 3 probabilities for each class (from the largest)
    top3_index2 = np.argsort(ens_pred_proba, axis=1)[:, -3:][:, ::-1]
    class_labels2 = ens_model_1.classes_
    top3_labels2 = class_labels2[top3_index2]

    # Compute the map@3 score for the test set
    ens_map3 = map3_score(top3_labels2, y_test_encoded)
    print(f"â–¶ï¸�MAP@3 Score: {ens_map3}\n")
    
    print("Inference on test data...")
    # Returning prediction probabilies for each example in the test_data_v2
    submission_probas = ens_model_1.predict_proba(testing_data)
    # Returning the indexes of top 3 probabilities for each class (from the largest)
    top3_index_subm = np.argsort(submission_probas, axis=1)[:, -3:][:, ::-1]
    decoded_flat_top3 = lab_encoder_train.inverse_transform(top3_index_subm.flatten())
    # print(f"Debug: {decoded_flat_top3}") <==== used it during debugging
    print("Inference completeâœ…")
    
    print("Submitting predictions...")
    # Convert top 3 predictions to space-separated strings
    preds_as_strings = [' '.join(labels) for labels in decoded_flat_top3.reshape(top3_index_subm.shape)]
    # Build the submission DataFrame
    submission = pd.DataFrame({
        'id': test_data['id'].values,
        'Fertilizer Name': preds_as_strings
    })
    submission.to_csv('/kaggle/working/submission.csv',index=False)
    print("Submitted ğŸ�‰ğŸ�‰")
    display(submission)



# Function call 
ensemble_one(
    X_train=X_train_v2,
    y_train=y_train,
    X_test=X_test_v2,
    y_test=y_test,
    testing_data=test_data_v2
)


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def stacked_ensemble(X_train, y_train, X_test, y_test, testing_data):
    # Best parameters for each model from previous jobs and other notebooks 
    catboost_params = {'iterations': 3820,
     'learning_rate': 0.2839466858475117,
     'depth': 4,
     'l2_leaf_reg': 4.245507965247879,
     'border_count': 169,
     'random_strength': 0.1416768708009795,
     'bagging_temperature': 0.10225652601468996,
     'auto_class_weights': None,
     'random_seed': 42,
     'thread_count': -1,
     'verbose': False,
     'task_type': "GPU",
     'devices': 'cuda'
    }
    lgbm_params = {
        'n_estimators': 4000, 
        'learning_rate': 0.05, 
        'max_depth': 16, 
        'device_type': 'gpu',
        'random_state': 42,
        'subsample': 0.686,
        'colsample_bytree': 0.25,
        'min_child_samples': 30,
        'num_leaves': 128,
        'boosting_type':'gbdt',
        'verbosity':-1
    }
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 7,
        'max_depth': 16,
        'learning_rate': 0.01,
        'n_estimators': 4000,
        'reg_alpha': 3,
        'reg_lambda': 1.4,
        'gamma': 0.26,
        'max_delta_step': 5,
        'subsample': 0.86,
        'colsample_bytree': 0.4,
        'min_child_weight': 5,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss',
        'enable_categorical': True,
        'device': "cuda",
        'verbosity': 0
    }

    base_models = [
        ("catboost", CatBoostClassifier(**catboost_params)),
        ("lightgbm", LGBMClassifier(**lgbm_params)),
        ("xgboost", xgb.XGBClassifier(**xgb_params))
    ]

    meta_learner = LogisticRegression()

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        # passthrough=True
    )

    
    # Encord target labels to (1, 2, 3, 4, 5,...)
    lab_encoder_train = LabelEncoder() # for y_train
    lab_encoder_test = LabelEncoder() # for y_test
    y_train_encoded = lab_encoder_train.fit_transform(y_train) 
    y_test_encoded = lab_encoder_test.fit_transform(y_test)

    # Fit the stacked model    
    stack.fit(X_train, y_train_encoded)
    pred_proba = stack.predict_proba(X_test)
    
    # Returning the indexes of top 3 probabilities for each class (from the largest)
    top3_index2 = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
    class_labels2 = ens_model_1.classes_
    top3_labels2 = class_labels2[top3_index2]

    # Compute the map@3 score for the test set
    ens_map3 = map3_score(top3_labels2, y_test_encoded)
    print(f"âš™ï¸� MAP@3 Score: {ens_map3}\n")
    
    print("Inference on test data...")
    # Returning prediction probabilies for each example in the test_data_v2
    submission_probas = stack.predict_proba(testing_data)
    # Returning the indexes of top 3 probabilities for each class (from the largest)
    top3_index_subm = np.argsort(submission_probas, axis=1)[:, -3:][:, ::-1]
    decoded_flat_top3 = lab_encoder_train.inverse_transform(top3_index_subm.flatten())
    # print(f"Debug: {decoded_flat_top3}") <==== used it during debugging
    print("Inference complete âœ…âœ…âœ…")
    
    print("Submitting predictions...")
    # Convert top 3 predictions to space-separated strings
    preds_as_strings = [' '.join(labels) for labels in decoded_flat_top3.reshape(top3_index_subm.shape)]
    # Build the submission DataFrame
    submission = pd.DataFrame({
        'id': test_data['id'].values,
        'Fertilizer Name': preds_as_strings
    })
    submission.to_csv('/kaggle/working/submission.csv',index=False)
    print("Submitted ğŸ�‰ğŸ�‰ğŸ�‰")
    display(submission)



# Function call 
ensemble_two(
    X_train=X_train_v2,
    y_train=y_train,
    X_test=X_test_v2,
    y_test=y_test,
    testing_data=test_data_v2
)




