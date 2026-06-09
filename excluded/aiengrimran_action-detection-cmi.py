import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold,StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import os
import optuna
import sys
from xgboost import XGBClassifier
import numpy as np
# import joblib
from joblib import Memory,dump,load
import os
# /kaggle/working/models/xgb_tunned.pkl


os.listdir('/kaggle/input/xgb_tunned/scikitlearn/default/1')


model = load('/kaggle/input/xgb_tunned/scikitlearn/default/1/xgb_tuned.pkl')
# model = load('/kaggle/input/brbf-model-features-cols/xgb_model.pkl')
model_features = load('/kaggle/input/brbf-model-features-cols/model_features.pkl')
# le = load('/kaggle/input/brbf-model-features-cols/label_encoder.pkl')


# df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')



# def drop_not_used_columns(df):
#     # df.drop('sequence_id', axis=1, inplace=True)
#     df.drop('row_id', axis=1, inplace=True)
#     df.drop('subject', axis=1, inplace=True)
#     # df.drop('gesture', axis=1)
#     # df.drop('sequence_type', axis=1, inplace=True)
#     df.drop('orientation', axis=1, inplace=True)
#     df.drop('behavior', axis=1, inplace=True)
#     df.drop('phase', axis=1, inplace=True)
#     df.drop('sequence_counter', axis=1, inplace=True)
    
#     return df


# df = drop_not_used_columns(df)

# labels = df[['sequence_id', 'gesture']].drop_duplicates().set_index('sequence_id')


# le = LabelEncoder()
# labels['gesture_enc'] = le.fit_transform(labels['gesture'])

# seq_type = df[['sequence_id', 'sequence_type']].drop_duplicates().set_index('sequence_id')
# labels['binary_target'] = (seq_type['sequence_type'] =='target').astype(int)



# # df.columns.drop()
# df.drop(columns=['gesture', 'sequence_type'], inplace=True, errors='ignore')

# # fill NAs (can also be done post-aggregation if preferred )
# df.fillna(-1, inplace=True)

# # Aggregate features: mean, std, min, max
# agg_df = df.groupby('sequence_id').agg(['mean', 'std', 'min', 'max'])

# # fill multi-index column 
# agg_df.columns = ['_'.join(col) for col in agg_df.columns]

# # merge features and target
# data = agg_df.join(labels)

# # define features and target
# X = data.drop(columns=['gesture', 'gesture_enc', 'binary_target'])
# # or binary_target for binary classification
# y = data['gesture_enc']

# X_train, X_test, y_train, y_test= train_test_split(X,y, stratify=y, random_state=42)


# def objective(trial):  # Fixed typo: 'trail' → 'trial'

#     params = {
#         "verbosity": 0,
#         "n_estimators": trial.suggest_int("n_estimators", 100, 500),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03),
#         "subsample": trial.suggest_float("subsample", 0.5, 1),  # Fixed: 'sub_sample' → 'subsample'
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
#         "tree_method": "gpu_hist",
#         "predictor": "gpu_predictor"
#     }

#     model2 = XGBClassifier(**params, random_state=42)

#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     scores = cross_val_score(model2, X_train, y_train, cv=kf, scoring="accuracy")  # Added: cv and scoring

#     return scores.mean()


#create optuna study case
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=15, n_jobs=4)



# study.trials


# os.makedirs('/kaggle/working/models/',exist_ok=True)


#Preprocessing function

def preprocess_function(sequence_df):
    
    if not isinstance(sequence_df, pd.DataFrame):
        sequence_df = sequence_df.to_pandas()

    sequence_df = sequence_df.reset_index(drop=True)

    # Drop unused columns
    sequence_df.drop(columns=[
        'sequence_id', 'row_id', 'subject', 'sequence_type',
        'orientation', 'behavior', 'phase', 'sequence_counter'
    ], inplace=True, errors='ignore')

    sequence_df.fillna(-1, inplace=True)

    # Calculate all stats
    stats = ['mean', 'std', 'min', 'max']
    stat_dfs = {
        stat: getattr(sequence_df, stat)(numeric_only=True).to_frame().T
        for stat in stats
    }

    # Rename columns with suffix
    for stat in stats:
        stat_dfs[stat].columns = [f"{col}_{stat}" for col in stat_dfs[stat].columns]

    # Interleave columns: col1_mean, col1_std, col1_min, col1_max, col2_mean, ...
    all_columns = sequence_df.select_dtypes(include='number').columns
    combined = pd.DataFrame()
    for col in all_columns:
        row = {
            f"{col}_mean": stat_dfs['mean'][f"{col}_mean"].iloc[0],
            f"{col}_std": stat_dfs['std'][f"{col}_std"].iloc[0],
            f"{col}_min": stat_dfs['min'][f"{col}_min"].iloc[0],
            f"{col}_max": stat_dfs['max'][f"{col}_max"].iloc[0],
        }
        combined = pd.concat([combined, pd.DataFrame([row])], axis=1)

    return combined

# predict function 
def predict(sequence_df, sequence_demos_df):
# def predict(sequence_df, sequence_demos_df):
    # print("sequence_df shape:", sequence_df.shape)
    # print("sequence_demos_df shape:", sequence_demos_df.shape)
    # print("sequence_id:", sequence_df['sequence_id'][0] if 'sequence_id' in sequence_df.columns else "N/A")
    # 
    try:
        features = preprocess_function(sequence_df)
        # features = preprocess_function(sequence_df, sequence_demos_df)
        print("Preprocessing successful. Feature shape:", features.shape)
    except Exception as e:
        print("Error during preprocessing:", e)
        raise

    try:
        features = features.reindex(columns=model_features, fill_value=0.0)
        y_pred = model.predict(features)
        
        gesture = le.inverse_transform([y_pred[0]])[0]
        print("Predicted gesture:", gesture)
    except Exception as e:
        print("Error during prediction:", e)
        raise

    return gesture


# features
# features.shape

# model_features = X_test.columns.to_list()
# features3 = features.reindex(columns=model_features, fill_value=0.0)
# y_predict_test2 = model.predict(features)


# === Run Evaluation Server ===
import kaggle_evaluation.cmi_inference_server
import os
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'
        )
    )




