RANDOM_SEED = 42 
RANDOM_SEED_TEST = 42
# why I choose random seed 42?
# Because it's a common convention in programming to use 42 as a placeholder for a random seed
# There are not any mathematical significance to this number, but it has become a popular cultural reference in programming communities.


import pandas as pd
train = pd.read_csv(r'/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
test = pd.read_csv(r'/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
train_demo = pd.read_csv(r'/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_demo = pd.read_csv(r'/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')

train_features, other_train_information,test_features, other_test_information = None, None, None, None
try:
    train_features = pd.read_csv(r'/kaggle/input/cmi-competition-2/train_features.csv')
    other_train_information = pd.read_csv(r'/kaggle/input/cmi-competition-2/other_train_information.csv')
    test_features = pd.read_csv(r'/kaggle/input/cmi-competition-2/test_features.csv')
    other_test_information = pd.read_csv(r'/kaggle/input/cmi-competition-2/other_test_information.csv')
    print(" read successfully")
except Exception as e:
    print(f"Error reading feature files: {e}")



print('Train shape:', train.shape)
print('Test shape:', test.shape)
print('Train demographics shape:', train_demo.shape)
print('Test demographics shape:', test_demo.shape)
print('Train features shape:', train_features.shape)    
print('Other train information shape:', other_train_information.shape)
print('Test features shape:', test_features.shape)  
print('Other test information shape:', other_test_information.shape)


# Find columns in train but not in test
train_cols = set(train.columns)
test_cols = set(test.columns)
columns_in_train_only = list(train_cols - test_cols)
print("\nColumns present in train but not in test:")
print(columns_in_train_only)



def number_missing_values(df:pd.DataFrame , name):
    if df is None:
        print(f"{name}: DataFrame is None, skipping missing value check.")
        return
    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        print(f"{name}: No missing values found!")
    else:
        print(f"{name}: Found {total_missing} missing values!")
    
    return total_missing

number_missing_values(train, "train.csv")
number_missing_values(test, "test.csv")
number_missing_values(train_demo, "train_demographics.csv")
number_missing_values(test_demo, "test_demographics.csv")
number_missing_values(train_features, "train_features.csv")
number_missing_values(other_train_information, "other_train_information.csv")    
number_missing_values(test_features, "test_features.csv")
number_missing_values(other_test_information, "other_test_information.csv")




def check_group_less_than_2_elements(df, logger_name):
    print(logger_name)
    grouped = df.groupby('sequence_id')
    print("num groups:", len(grouped))

    flag = True
    for name, group in grouped:
        if len(group) < 2:
            print(f"Group {name} has less than 2 elements.")
            flag = False

    if flag:
        print("All groups have at least 2 elements.")

check_group_less_than_2_elements(train,'train')
check_group_less_than_2_elements(test,'test')


#skip this cell because I have uploaded the results of this cell
from math import inf
import numpy as np
import pandas as pd
from scipy import stats

def extract_features(df, df_demo, drop_sequence_id=False, skip_info_list = False):
    continue_columns = {
        'sequence_id', 'row_id', 'sequence_type', 'sequence_counter',
        'subject', 'orientation', 'behavior', 'phase', 'gesture'
    }

    # Group by sequence_id
    grouped = df.groupby('sequence_id')
    feature_list = []
    info_list = []

    for seq_id, group in grouped:
        # Initialize feature and info dictionaries
        features = {'sequence_id': seq_id}
        info = {}

        # Count total missing values in this sequence
        missing_count = group.isna().sum().sum()
        features['number_missing_values'] = missing_count
        features['subject'] = group['subject'].values[0]
        # Collect constant info fields
        if skip_info_list == False:
            for col in continue_columns:
                if col in group.columns:
                    vals = group[col].values
                    if vals[0] == vals[-1] and (vals == vals[0]).all():
                        info[col] = vals[0]

        # Numeric feature extraction
        numeric_cols = [col for col in group.columns if col not in continue_columns]
        data = group[numeric_cols].values  # (n_rows, n_features)

        means = data.mean(axis=0)
        stds = data.std(axis=0)
        mins = data.min(axis=0)
        maxs = data.max(axis=0)
        medians = np.median(data, axis=0)
        no_responses = (data == -1).sum(axis=0)
        modes = stats.mode(data, axis=0, keepdims=False)[0]

        for i, col in enumerate(numeric_cols):
            features[f'{col}_mean'] = means[i]
            features[f'{col}_std'] = stds[i]
            features[f'{col}_min'] = mins[i]
            features[f'{col}_max'] = maxs[i]
            features[f'{col}_median'] = medians[i]
            features[f'{col}_no_response'] = no_responses[i]
            features[f'{col}_mode'] = modes[i]
            if stds[i] != 0:
                features[f'{col}_skew'] = stats.skew(data[:, i])
                features[f'{col}_kurtosis'] = stats.kurtosis(data[:, i])
            else:
                features[f'{col}_skew'] = -1
                features[f'{col}_kurtosis'] = -1

        feature_list.append(features)
        info_list.append(info)

    # Create DataFrames
    feature_df = pd.DataFrame(feature_list)
    info_df = pd.DataFrame(info_list)

    # Merge demographic info
    feature_df = feature_df.merge(df_demo, on='subject', how='inner')

    # Drop columns
    feature_df.drop(columns=['subject'], inplace=True, errors='ignore')
    if drop_sequence_id:
        feature_df.drop(columns=['sequence_id'], inplace=True, errors='ignore')

    # Return feature_df, info_df
    if skip_info_list:
        return feature_df,None 
    else:
        return feature_df, info_df
    


# train_features, other_train_information = extract_features(train,train_demo)
# test_features, other_test_information = extract_features(test,test_demo)

# display(train_features.head(1))
# display(test_features.head(1))



TARGET_GESTURES =  set(train[ train['sequence_type']=='Target' ]['gesture'])
NON_TARGET_GESTURES = set(train[ train['sequence_type']=='Non-Target' ]['gesture'])
ALL_GESTURES = TARGET_GESTURES.union(NON_TARGET_GESTURES)
NUM_GESTURES = len(ALL_GESTURES)
print("Target gestures:", TARGET_GESTURES)
print("Non-target gestures:", NON_TARGET_GESTURES)
print("Intersection: ", TARGET_GESTURES.intersection(NON_TARGET_GESTURES) )
print("all_gestures", ALL_GESTURES)

TARGET_GESTURES = list(TARGET_GESTURES)
NON_TARGET_GESTURES = list(NON_TARGET_GESTURES)
ALL_GESTURES = list(ALL_GESTURES)

GESTURE_TO_IDX = {gesture: i for i, gesture in enumerate(ALL_GESTURES)}
IDX_TO_GESTURE = {i: gesture for i, gesture in enumerate(ALL_GESTURES)} 
print("GESTURE_TO_IDX:", GESTURE_TO_IDX)
print("IDX_TO_GESTURE:", IDX_TO_GESTURE)

MAX_INPUT_FEATURES = train_features.drop(columns=['sequence_id']).shape[1]
print("MAX_INPUT_FEATURES:", MAX_INPUT_FEATURES)


from sklearn.model_selection import train_test_split


X = train_features.drop(columns=['sequence_id'])
y = other_train_information['gesture']
y = y.map(GESTURE_TO_IDX) 

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_SEED_TEST, stratify=y
)
print("Shape X_train:", X_train.shape)
print("Shape X_val:", X_val.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_val:", y_val.shape)



from sklearn.metrics import f1_score

def calculate_hierarchical_f1(y_true, y_pred):
    """
    Calculate hierarchical F1 score từ class indices.
    
    Args:
        y_true: numpy array of true class indices
        y_pred: numpy array of predicted class indices
        
    Returns:
        float: Hierarchical F1 score (0.5 * binary_f1 + 0.5 * macro_f1)
    """
    # Convert indices back to gesture names
    y_true_gestures = [IDX_TO_GESTURE[int(idx)] for idx in y_true]
    y_pred_gestures = [IDX_TO_GESTURE[int(idx)] for idx in y_pred]
    
    # Binary F1 (Target vs Non-Target)
    y_true_bin = [gesture in TARGET_GESTURES for gesture in y_true_gestures]
    y_pred_bin = [gesture in TARGET_GESTURES for gesture in y_pred_gestures]
    
    f1_binary = f1_score(y_true_bin, y_pred_bin, pos_label=True, zero_division=0, average='binary')
    
    # Macro F1 (collapse non-targets to single class)
    y_true_mc = [gesture if gesture in TARGET_GESTURES else 'non_target' for gesture in y_true_gestures]
    y_pred_mc = [gesture if gesture in TARGET_GESTURES else 'non_target' for gesture in y_pred_gestures]
    
    f1_macro = f1_score(y_true_mc, y_pred_mc, average='macro', zero_division=0)
    
    return 0.5 * f1_binary + 0.5 * f1_macro



import xgboost as xgb

XGBOOST_MAX_NUM_TREES = 2000
XGBOOST_EARLY_STOPPING_ROUNDS = 100 
XGBOOST_VERBOSE_EVAL = 50

def xgb_hierarchical_f1_eval(preds, dtrain):
    labels = dtrain.get_label()
    preds_class = np.argmax(preds.reshape(-1, len(ALL_GESTURES)), axis=1)
    score = calculate_hierarchical_f1(labels, preds_class)
    return 'hierarchical_f1', score

def xgboost_show_importance(bst, importance_type='gain', num_features_show=50):

    assert(num_features_show > 0 and num_features_show <= MAX_INPUT_FEATURES)
    import matplotlib.pyplot as plt

    xgb.plot_importance(bst, importance_type=importance_type, max_num_features=num_features_show)
    plt.show()
    importance_dict = bst.get_score( importance_type=importance_type )  
    importance_list = [ (key,importance_dict[key]) for key in importance_dict]
    importance_list_sorted = sorted(importance_list, key=lambda x: x[1], reverse=True)
    for feature, importance in importance_list_sorted[:num_features_show]:
        print(f"{feature}: {importance:.4f}")

    print( " num features:", len(importance_list_sorted) ) 
    return importance_list_sorted

def xgboost_get_top_k_important_features(bst, k=1000, importance_type='gain'):
    assert(k > 0 and k <= MAX_INPUT_FEATURES)
    important_dict = bst.get_score(importance_type=importance_type)
    topk = [feat for feat, _ in sorted(important_dict.items(), key=lambda x: x[1], reverse=True)[:k]]
    return topk

def xgboost_train( X_train, y_train, X_val, y_val , print_best = True, print_warning = True):
    import warnings
    if not print_warning:
        warnings.filterwarnings("ignore", category=UserWarning)

    params = {
    'objective': 'multi:softprob',
    'num_class': NUM_GESTURES,
    'eval_metric': 'mlogloss',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'seed': RANDOM_SEED,
}

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=XGBOOST_MAX_NUM_TREES,
        evals=watchlist,
        feval=xgb_hierarchical_f1_eval,
        maximize=True,
        early_stopping_rounds=XGBOOST_EARLY_STOPPING_ROUNDS,
        verbose_eval=XGBOOST_VERBOSE_EVAL,
    )
    if print_best:
        print("Best iteration:", bst.best_iteration, " Best score:", bst.best_score)
    return bst  


def xgboost_retrain_with_top_k_features(bst, X_train, y_train, X_val, y_val, k=1000, importance_type='gain', print_best=False): 
    assert(k > 0 and k <= MAX_INPUT_FEATURES)

    topk = xgboost_get_top_k_important_features(bst, k=k, importance_type=importance_type)
    X_train_topk = X_train[topk]
    X_val_topk   = X_val[topk]
    return xgboost_train(X_train_topk,y_train,X_val_topk,y_val,print_best,False)


def xgboost_predict(bst, sequence, demographics, type_return) -> str|int|list:
    """
    type_return = 0 -> return str.  
    type_return = 1 -> return int.  
    type_return = 2 -> return list of probality.
    """
    try:
        sequence = sequence.to_pandas()
        demographics = demographics.to_pandas()
    except Exception as e:
        print("Lỗi khi chuyển đổi sang pandas:", e)

    input_features, other_input_information = extract_features(sequence,demographics)
    expected_cols = bst.feature_names
    input_features = input_features[expected_cols]
    input_features = xgb.DMatrix(input_features)
    best_iter = bst.best_iteration
    val = bst.predict(input_features,iteration_range=(0, best_iter))
    
    if type_return == 0 :
        pred_indices = np.argmax(val, axis=1)
        pred_gestures = [IDX_TO_GESTURE[idx] for idx in pred_indices]
        return str(pred_gestures[0]) 
    elif type_return == 1: 
        pred_indices = np.argmax(val, axis=1)
        return int(pred_indices[0]) 
    else :
        return list(val)


# retrain with all data 
print(" start training")
xgboost_model =  xgboost_train(X_train, y_train, X_val, y_val)


best_xgboost_model = xgboost_model
best_iteration = xgboost_model.best_iteration 
best_score = xgboost_model.best_score 
best_topK_features = MAX_INPUT_FEATURES

# for num_features in range(100,2000+100,100):
for num_features in [500,1400,1500]:
    print(f"Retraining with top {num_features} features...", end="")
    xgboost_mode_topK_features=  xgboost_retrain_with_top_k_features(xgboost_model, X_train, y_train, X_val, y_val, k=num_features, importance_type='gain')
    iteration = xgboost_mode_topK_features.best_iteration
    score = xgboost_mode_topK_features.best_score 
    print(" iteration = ", iteration," score = ",score)
    if score > best_score:
        best_xgboost_model = xgboost_mode_topK_features
        best_iteration = iteration
        best_score = score
        best_topK_features = num_features
        print(f"New best model found with {num_features} features: Best iteration: {iteration}, Best score: {score:.4f}") 

print(f"Best model with {best_topK_features} features: Best iteration: {best_iteration}, Best score: {best_score:.4f}")

best_xgboost_model.save_model("best_xgboost_model.json")



final_model=  xgboost_retrain_with_top_k_features(xgboost_model, X, y, X_val, y_val, k=best_topK_features, importance_type='gain')
print(" FINAL MODEL = score", final_model.best_score)


import polars as pl
import os

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    return xgboost_predict(final_model,sequence,demographics,0) 
#predict( test, test_demo)


import kaggle_evaluation.cmi_inference_server

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

