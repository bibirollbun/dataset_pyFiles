import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import xgboost as xgb
import random
import os
import joblib


from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
from scipy.spatial.transform import Rotation
from sklearn.feature_selection import SelectFromModel


KAGGLE = True
TRAIN = False 
GRIDSEARCH = False
SHORTENSEQUENCE = False
VERSION = 31
MODELFILENAME = 'motion_only_simple_xgboost-v' + str(VERSION) + '.json'
CLASSFILENAME = 'gesture_classes-v' + str(VERSION) + '.npy'
PECENTLISTFILENAME = 'percent_list-v' + str(VERSION) + '.npy'
LENGTHLISTFILENAME = 'length_list-v' + str(VERSION) + '.npy'


random_seed = 42
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
seed_everything(seed=random_seed)


# Load the datasets
if KAGGLE:
    import polars as pl
    DATA_DIR = Path('/kaggle/input/cmi-detect-behavior-with-sensor-data')
    EXPORT_DIR = Path('/kaggle/input/motion-only-xgboost/other/default/15')
    test_df = pl.read_csv(DATA_DIR / "test.csv")
    test_dem_df = pl.read_csv(DATA_DIR / "test_demographics.csv")
else:
    DATA_DIR = Path('.')
    EXPORT_DIR = Path('.')
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_dem_df = pd.read_csv(DATA_DIR / "test_demographics.csv")

if TRAIN:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    train_dem_df = pd.read_csv(DATA_DIR / "train_demographics.csv")


numeric_columns_accel = ["acc_x", "acc_y", "acc_z"]
numeric_columns_accel_no_gravity = ["acc_x_no_gravity", "acc_y_no_gravity", "acc_z_no_gravity"]
numeric_columns_rotation = ["rot_w", "rot_x", "rot_y", "rot_z"]
numeric_columns_euler_rotation = ["rot_euler_x", "rot_euler_y", "rot_euler_z"]
numeric_columns_thermal = [f"thm_{i}" for i in range(1, 6)]
numeric_columns_tof = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]
numeric_columns_tof_mean = [f"tof_{i}_mean" for i in range(1, 6)]
train_categorical_columns = ['sequence_type', 'subject', 'gesture', 'orientation', 'behavior']
train_demographics_categorical_columns = [
    'subject',
    'adult_child',
    'sex',
    'handedness'
]
train_demographics_numerical_columns = [
    'age',
    'height_cm',
    'shoulder_to_wrist_cm',
    'elbow_to_wrist_cm'
]

train_demographics_columns_to_use = ['subject', 'handedness']


target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]





def remove_gravity_from_acc(df):

    acc_values = df[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)

    gravity = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue

        try:
            rotation = Rotation.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
             linear_accel[i, :] = acc_values[i, :]

    df['acc_x_no_gravity'] = linear_accel[:,0]
    df['acc_y_no_gravity'] = linear_accel[:,1]
    df['acc_z_no_gravity'] = linear_accel[:,2]

    """
    try:
        euler_angles = rotation.as_euler('xyz', degrees=True)
        df['rot_euler_x'] = euler_angles[0]
        df['rot_euler_y'] = euler_angles[1]
        df['rot_euler_z'] = euler_angles[2]
    except ValueError:
        df['rot_euler_x'] = np.nan
        df['rot_euler_y'] = np.nan
        df['rot_euler_z'] = np.nan
    """
    return df



def prepare_tof_data(df):
    #df[numeric_columns_tof] = df[numeric_columns_tof].ffill().bfill().fillna(-1)
    # 255 corresponds to the largest distance from the ToF sensor
    for col in numeric_columns_tof:
        df.loc[df[col] == -1, col] = 256
        
    added_cols = {}
    for i in range(1, 6):
        cols = [f"tof_{i}_v{j}"  for j in range(64)]
        added_cols[f"tof_{i}_mean"] = df[cols].mean(axis=1)
        added_cols[f"tof_{i}_max"] = df[cols].max(axis=1)
        added_cols[f"tof_{i}_min"] = df[cols].min(axis=1)
        added_cols[f"tof_{i}_median"] = df[cols].median(axis=1)
        added_cols[f"tof_{i}_std"] = df[cols].std(axis=1)
        #added_cols[f"tof_{i}_skew"] = df[cols].skew(axis=1)
        #added_cols[f"tof_{i}_iqr"] = df[cols].quantile(0.75) - df[cols].quantile(0.25)
    new_cols = pd.DataFrame(added_cols)
    return pd.concat([df, new_cols], axis=1) 


def create_features_seq(df, train_bool):
    if train_bool:
        gesture = df['gesture'].iloc[0]
    subject = df['subject'].iloc[0]
    sequence_id = df['sequence_id'].iloc[0]
    handedness = df['handedness'].iloc[0]

    #  components
    seq_df = df[numeric_columns_accel + numeric_columns_rotation + numeric_columns_accel_no_gravity + 
                numeric_columns_thermal]
    
    math_df = seq_df.apply([np.mean, np.max, np.min, np.median, np.std])
    iqr = seq_df.quantile(0.75) - seq_df.quantile(0.25)
    skew = seq_df.skew()
    kurtosis = seq_df.kurtosis()
    energy = seq_df.apply(lambda num: num**2).sum()/len(seq_df)

    abs_signal = seq_df.apply(lambda num: abs(num)).sum()/len(seq_df)
    sma_accel = abs_signal[numeric_columns_accel].sum()
    sma_accel_no_gravity = abs_signal[numeric_columns_accel_no_gravity].sum()
    sma_rotation = abs_signal[numeric_columns_rotation].sum()
    #sma_euler_rotation = abs_signal[numeric_columns_euler_rotation].sum()
    
    concat_df = pd.concat([iqr, skew, kurtosis, energy], axis=1)
    concat_df.columns = ['iqr', 'skew', 'kurtosis', 'energy']
    concat_df = concat_df.transpose()
    concat_df = pd.concat([math_df, concat_df], axis=0)

    concat_df['func'] = concat_df.index
    concat_df = concat_df.melt(id_vars=['func'])
    concat_df['var_name'] = concat_df['variable'] + '_' + concat_df['func']
    concat_df = concat_df.drop(columns=['func', 'variable'])
    
    new_row = pd.DataFrame({'var_name': ['sma_accel',
                                         'sma_accel_no_gravity',
                                         'sma_rotation',
                                         'subject',
                                         'sequence_id', 
                                        'handedness'], 
                            'value': [sma_accel,
                                      sma_accel_no_gravity,
                                      sma_rotation,
                                      subject,
                                      sequence_id,
                                     handedness]})
    

    # 255 corresponds to the largest distance from the ToF sensor
    """
    for col in numeric_columns_tof:
        seq_df[seq_df[col] == -1] = 256
    """
    
    
    # time of flight
    tof_cols = {}
    for i in range(1, 6):
        cols = [f"tof_{i}_v{j}"  for j in range(64)]
        tof_values = df[cols].to_numpy().flatten()
        tof_values[tof_values == -1] = 256
        if sum(np.isnan(tof_values)) <= 0.8 * len(tof_values):
            tof_cols[f"tof_{i}_mean"] = np.nanmean(tof_values)
            tof_cols[f"tof_{i}_max"] = np.nanmax(tof_values)
            tof_cols[f"tof_{i}_min"] = np.nanmin(tof_values)
            tof_cols[f"tof_{i}_median"] = np.nanmedian(tof_values)
            tof_cols[f"tof_{i}_std"] = np.nanstd(tof_values)
        else:
            tof_cols[f"tof_{i}_mean"] = np.nan
            tof_cols[f"tof_{i}_max"] = np.nan
            tof_cols[f"tof_{i}_min"] = np.nan
            tof_cols[f"tof_{i}_median"] = np.nan
            tof_cols[f"tof_{i}_std"] = np.nan
        #tof_cols[f"tof_{i}_skew"] = np.skew(df[cols].to_numpy())
        #tof_cols[f"tof_{i}_iqr"] = seq_df[cols].quantile(0.75) - seq_df[cols].quantile(0.25)
    
    tof_df = pd.DataFrame(tof_cols, index =['value']).transpose()
    tof_df.index.names = ['var_name']
    tof_df = tof_df.reset_index()

    #concat_df = pd.concat([concat_df, new_row, tof_cols], ignore_index=True)

    concat_df = pd.concat([concat_df, new_row, tof_df], ignore_index=True)
    
    if train_bool:
        new_row = pd.DataFrame({'var_name': ['gesture'],'value': [gesture]})
        concat_df = pd.concat([concat_df, new_row], ignore_index=True)
    
    concat_df = concat_df.set_index('var_name')

    return concat_df


def create_feature_df(df):
    cat_list = []
    for ind, sequence_id in enumerate(df['sequence_id'].unique()):
        if ind % 100 == 0:
            print(ind)
        seq = df[df['sequence_id'] == sequence_id]
        if SHORTENSEQUENCE:
            gesture_percent = predict_gesture_percent(len(seq))
            gesture_ind = round(len(seq) * gesture_percent)
            seq = seq.iloc[-gesture_ind:-1]
        cat_list.append(create_features_seq(seq, TRAIN))    
    ret_df = pd.concat(cat_list, axis=1)
    ret_df = ret_df.transpose().reset_index().drop(columns='index')
    return ret_df


def split_data(df):
    X, y = df.drop(['gesture', 'subject', 'sequence_id'], axis=1), df[['gesture']]
    X = X.apply(pd.to_numeric)
    
    # Encode y to numeric
    le = LabelEncoder()
    y_encoded = le.fit_transform(np.ravel(y))
    gesture_classes = le.classes_
    np.save(EXPORT_DIR / CLASSFILENAME, le.classes_)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, 
                                                        random_state=1, 
                                                        stratify=y_encoded)
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    
    params = {
        'objective': 'multi:softmax',
        'eval_metric': 'merror',
        'early_stopping_rounds': 10,
        'random_state': random_seed,
        'max_depth': 6, #depth of tree
        'min_child_weight': 5, # minimum number of instances needed to be in each node 
        'subsample': 1.0, #fraction of observations used for each tree
        'colsample_bytree': 0.8, #percentage of features ( columns ) will be used for building each tree
        'learning_rate': 0.3   
    }

    
    """
    params = {
        'objective': 'multi:softmax',
        'eval_metric': 'merror',
        'early_stopping_rounds': 10,
        'random_state': random_seed,  
    }
    """
    
    class_weights = dict(enumerate(len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))))
    sample_weight = np.array([class_weights[label] for label in y_train])
    
    # Instantiate XGBClassifier with the parameters
    model = xgb.XGBClassifier(**params)
    # Train the model with early stopping
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], sample_weight=sample_weight)

    # Analyze Model
    # Retrieve the merror values from the training process
    results = model.evals_result()
    epochs = len(results['validation_0']['merror'])
    x_axis = range(0, epochs)
    
    # Plot the merror values
    plt.figure()
    plt.plot(x_axis, results['validation_0']['merror'], label='Test')
    plt.legend()
    plt.xlabel('Number of Boosting Rounds')
    plt.ylabel('Multiclass Classification Error')
    plt.title('XGBoost merror Performance')
    plt.show()
    
    y_pred = model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    f1_score_val = f1_score(y_pred,y_test,average='macro')
    confusion = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1_score_val:.3f}")
    
    print(f"Confusion Matrix:\n{confusion}")
    
    print("\n Classification Report:\n",
          classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(4, 30))
    xgb.plot_importance(model, ax=ax)
    plt.show()
    
    return model


def param_search(df):
    param_grid = {
        'max_depth': [4, 6, 8], #depth of tree, default=6
        'min_child_weight': [1, 3, 5], #default=1
        'subsample': [0.6, 0.8, 1.0], #fraction of observations used for each tree. default=1 
        'colsample_bytree': [0.6, 0.8, 1], #percentage of features ( columns ) will be used for building each tree, default=1
        'learning_rate': [0.03, 0.1, 0.3] #default = 0.3
    }
    
    params = {
        'objective': 'multi:softmax',
        'eval_metric': 'merror',
        'random_state': random_seed
    }

    X_train, X_test, y_train, y_test = split_data(df)

    
    
    # Create XGBoost classifier
    model = xgb.XGBClassifier(**params)
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, n_jobs=-1, verbose=3, scoring='f1_macro')
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

#


def construct_gesture_percent_lists(df, window_size):
    gesture_percent_list = []
    length_seq = []
    concat_list = []
    for ind, sequence_id in enumerate(df['sequence_id'].unique()):
        if ind % 100 == 0:
            print(ind)
        seq = df[df['sequence_id'] == sequence_id]
        length_seq.append(len(seq))
        if 'Performs gesture' in seq.value_counts('behavior'):
            gesture_percent =seq.value_counts('behavior')['Performs gesture'] /len(seq)
        gesture_percent_list.append(gesture_percent)
    test_df = pd.DataFrame({'length': pd.Series(length_seq),  'percent':pd.Series(gesture_percent_list)}).sort_values(by='length')
    for ind in np.arange(window_size,len(test_df),window_size):
        concat_list.append(test_df.iloc[ind-window_size:ind].mean())
    ret_df = pd.concat(concat_list, axis=1)
    length_list = np.array(ret_df.loc['length', :].values.flatten().tolist())
    percent_list = np.array(ret_df.loc['percent', :].values.flatten().tolist())
    length_list = length_list + 1e-3*np.random.rand(len(length_list))
    
    np.save(EXPORT_DIR / PECENTLISTFILENAME, percent_list)
    np.save(EXPORT_DIR / LENGTHLISTFILENAME, length_list)
    
    return length_list, percent_list


def predict_gesture_percent(x):
    
    percent_list = np.load(EXPORT_DIR / PECENTLISTFILENAME, allow_pickle=True)
    length_list = np.load(EXPORT_DIR / LENGTHLISTFILENAME, allow_pickle=True)
    
    if x < length_list[0]:
        return percent_list[0]
    elif x > length_list[-1]:
        return percent_list[-1]
    else:
        itemindex = np.where(np.array(length_list) > x)[0][0]
        return 0.5*(percent_list[itemindex]+percent_list[itemindex-1])


def f1_eval(y_pred, y_true):
    """Custom F1 evaluation function for XGBoost"""
    y_true = y_true.get_label()
    y_pred = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    return 'f1', f1


def remove_thermal_outliers_constant(df):    
    for col in numeric_columns_thermal:
        df.loc[df[col] < 17.5, col] = np.nan
        df.loc[df[col] > 40, col] = np.nan
    return df


if TRAIN:
    train_df = train_df.merge(train_dem_df[train_demographics_columns_to_use], on='subject', how='left')
    train_df = remove_gravity_from_acc(train_df)
    #tof_data = prepare_tof_data(train_df)
    train_df = remove_thermal_outliers_constant(train_df)
    train_df[numeric_columns_tof].replace(-1, 256, inplace=True)
    if SHORTENSEQUENCE:
        length_list, percent_list = construct_gesture_percent_lists(train_df, 10)
    feature_df = create_feature_df(train_df)
    #feature_df = pd.concat([feature_df, tof_data], axis = 1)
    X_train, X_test, y_train, y_test = split_data(feature_df)
    model = train_model(X_train, X_test, y_train, y_test)
    model.save_model(EXPORT_DIR / MODELFILENAME)
    #joblib.dump(model,)
  





if GRIDSEARCH:
    param_search(feature_df)





def predict_pandas(sequence: pd.DataFrame, demographics: pd.DataFrame) -> str:
    gesture_classes = np.load(EXPORT_DIR / CLASSFILENAME, allow_pickle=True)
    model_reload = xgb.XGBClassifier()
    model_reload.load_model(EXPORT_DIR / MODELFILENAME)
    sequence = sequence.merge(demographics[train_demographics_columns_to_use], on='subject', how='left')
    sequence = remove_gravity_from_acc(sequence)
    #tof_data = prepare_tof_data(sequence)
    sequence = remove_thermal_outliers_constant(sequence)
    feature_df = create_features_seq(sequence, False)
    #feature_df = pd.concat([feature_df, tof_data], axis=1)
    feature_df = feature_df.drop(['sequence_id', 'subject'])
    feature_df = feature_df.apply(pd.to_numeric)
    feature_df = feature_df.transpose().reset_index().drop(columns='index')
    return gesture_classes[model_reload.predict(feature_df)[0]]

if KAGGLE:
    import polars as pl
    
    def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
        sequence = sequence.to_pandas()
        demographics = demographics.to_pandas()
        return predict_pandas(sequence, demographics)
    
    print(predict(test_df,test_dem_df))
else:
    print(predict_pandas(test_df,test_dem_df))


if KAGGLE:
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






