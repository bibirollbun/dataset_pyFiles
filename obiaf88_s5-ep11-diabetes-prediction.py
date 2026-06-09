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


pd.set_option('display.max_columns', 500)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
import keras
from keras import layers
from keras import ops
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.utils.class_weight import compute_class_weight


train = pd.read_csv(r'/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e12/test.csv')
sample_submission = pd.read_csv(r'/kaggle/input/playground-series-s5e12/sample_submission.csv')


test.head(2)


sample_submission.head()


train.head(2)


test.head()


for df in [train,test]:
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['map_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
    df['high_bp_flag'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85)).astype(int)
    df['obesity_flag'] = (df['bmi'] >= 30).astype(int)
    df['age_group'] = pd.cut(df['age'],bins = [0,20,43,51,60,80,150], include_lowest = True, duplicates = 'drop')
    df['cholesterol_ldl_ratio'] = df['ldl_cholesterol'] / df['cholesterol_total']
    df['alcohol_consumption_per_week_group'] = pd.cut(df['alcohol_consumption_per_week'],bins = [0,3,5,7,10,24,1000], include_lowest = True, duplicates = 'drop')
    df['physical_activity_minutes_per_week_group'] = pd.cut(df['physical_activity_minutes_per_week'],bins = [0,5,30,50,75,105,200,400,800,1500,10000], include_lowest = True, duplicates = 'drop')
    df['sleep_hours_per_day_group'] = pd.cut(df['sleep_hours_per_day'],bins = [0,3.5,5.5,7.5,8.5,15,100], include_lowest = True, duplicates = 'drop')
    df['screen_time_hours_per_day_group'] = pd.cut(df['screen_time_hours_per_day'],bins = [0,1,3,5,7,9,15,36,200], include_lowest = True, duplicates = 'drop')
    df['bmi_group'] = pd.cut(df['bmi'],bins = [0,16,19,24,26,29,34,39,55,200], include_lowest = True, duplicates = 'drop')
    df['waist_to_hip_ratio_group'] = pd.cut(df['waist_to_hip_ratio'],bins = [0,0.7,0.75,0.8,0.83,0.85,0.87,0.9,0.95,1,1.1,1.5,100], include_lowest = True, duplicates = 'drop')
    df['systolic_bp_group'] = pd.cut(df['systolic_bp'],bins = [0,91,95,102,110,117,120,125,130,140,150,160,190,250,2500], include_lowest = True, duplicates = 'drop')
    df['diastolic_bp_group'] = pd.cut(df['diastolic_bp'],bins = [0,52,60,65,73,78,82,85,90,95,100,130,500], include_lowest = True, duplicates = 'drop')
    df['heart_rate_group'] = pd.cut(df['heart_rate'],bins = [0,43,50,55,60,66,70,75,80,90,100,110,130,150,2000], include_lowest = True, duplicates = 'drop')
    df['cholesterol_total_group'] = pd.cut(df['cholesterol_total'],bins = [0,110,120,130,150,170,175,180,185,190,195,200,220,250,270,290,400,5000], include_lowest = True, duplicates = 'drop')
    df['hdl_cholesterol_group'] = pd.cut(df['hdl_cholesterol'],bins = [0,22,25,30,35,40,45,50,55,60,70,80,90,100,130,170,2500], include_lowest = True, duplicates = 'drop')
    df['ldl_cholesterol_group'] = pd.cut(df['ldl_cholesterol'],bins = [0,52,60,70,80,90,100,105,110,120,130,140,150,160,180,200,220,240,500,10000], include_lowest = True, duplicates = 'drop')
    df['triglycerides_group'] = pd.cut(df['triglycerides'],bins = [0,32,40,50,60,70,90,110,120,130,140,150,160,180,190,200,220,250,270,290,300,500,10000], include_lowest = True, duplicates = 'drop')    


X = train[[col for col in train.columns if col not in 'diagnosed_diabetes']]
y = train['diagnosed_diabetes']


num_columns = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides','pulse_pressure','map_pressure','cholesterol_ldl_ratio']
cat_columns = ['gender', 'ethnicity', 'education_level',
       'income_level', 'smoking_status', 'employment_status',
       'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history','high_bp_flag','obesity_flag',
        'age_group','alcohol_consumption_per_week_group','physical_activity_minutes_per_week_group',
              'sleep_hours_per_day_group','screen_time_hours_per_day_group',
              'bmi_group','waist_to_hip_ratio_group','systolic_bp_group',
              'diastolic_bp_group','heart_rate_group','cholesterol_total_group',
              'hdl_cholesterol_group','ldl_cholesterol_group','triglycerides_group']


train['diagnosed_diabetes'].sum()/ train['diagnosed_diabetes'].shape[0]*100


classes = np.array([0,1])
weights = compute_class_weight(
    class_weight = 'balanced',
    classes = classes,
    y = y
)

class_weight = {0: weights[0], 1:weights[1]}


class_weight


col_transformer = ColumnTransformer(
    remainder = 'drop',
    transformers = [
    ('num',StandardScaler(), num_columns),
    ('cat',OneHotEncoder(handle_unknown = 'infrequent_if_exist'), cat_columns)
])




def model_diabetes(input_dim):
    model =keras.Sequential([
        layers.Dense(64, activation = 'relu', input_shape = (input_dim)),
        layers.BatchNormalization(),
        layers.Dense(32, activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dense(16, activation = 'relu'),
        layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        loss = 'binary_crossentropy',
        metrics = [keras.metrics.AUC(name='auc')]
    )
    return model
    


import tensorflow as tf

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
else:
    print("GPU is not available. Using CPU.")


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Allow memory growth for GPU usage
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs.")
    except RuntimeError as e:
        print(e)


n_splits = 5

skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)

oof_preds = np.zeros(len(X))
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X,y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx] , y.iloc[val_idx]

    X_train = col_transformer.fit_transform(X_train)
    X_val = col_transformer.transform(X_val)
    model = model_diabetes((X_train.shape[1],))

    es = keras.callbacks.EarlyStopping(
        monitor = 'val_auc',
        mode = 'max',
        patience = 5,
        restore_best_weights = True,
        verbose = 0
    )

    model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs = 100,
        batch_size = 32,
        callbacks = [es],
        class_weight=class_weight,
        verbose = 0
    )

    val_preds = model.predict(X_val).ravel()
    oof_preds[val_idx] = val_preds

    auc = roc_auc_score(y_val, val_preds)

    auc_scores.append(auc)

    print(f"Fold {fold} AUC {auc}")


print(f"Maen AUC: {np.mean(auc_scores)}")


predictions = model.predict(col_transformer.transform(test[[col for col in test.columns if col not in 'id']]))


submission = pd.DataFrame({'id':test['id'],'diagnosed_diabetes': predictions.ravel() })


submission


submission.to_csv('submission.csv', index=False)
print("Submission created")

