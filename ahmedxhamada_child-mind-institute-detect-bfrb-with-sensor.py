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


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Masking, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


np.random.seed(42)
tf.random.set_seed(42)


# Suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
warnings.filterwarnings('ignore', category=UserWarning)



train = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demographics = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_demographics = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")



train.head()


train.describe()



train.info()


train.isnull().sum()


# Pie chart of sequence_type distribution
sample_seq_id = train['sequence_id'].iloc[0]
sample_seq = train[train['sequence_id'] == sample_seq_id]
sequence_type_counts = train.groupby('sequence_id')['sequence_type'].first().value_counts()
plt.figure(figsize=(8, 8))
plt.pie(
    sequence_type_counts,
    labels=sequence_type_counts.index,
    autopct='%1.1f%%',
    colors=['#ff9999', '#66b3ff'],
    startangle=90
)
plt.title('Distribution of Sequence Types (Target vs Non-Target)')
plt.savefig('/kaggle/working/sequence_type_pie.png')
plt.show()
plt.close()


# Heatmap of IMU feature correlations for one sequence
heatmap_seq = train[train['sequence_id'] == sample_seq_id]
imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
imu_data = heatmap_seq[imu_cols].fillna(0)
acc_magnitude = np.sqrt((heatmap_seq[['acc_x', 'acc_y', 'acc_z']] ** 2).sum(axis=1))
velocity = np.cumsum(acc_magnitude) / len(acc_magnitude)
jerk = np.diff(acc_magnitude, prepend=acc_magnitude.iloc[0])
imu_df = pd.DataFrame(imu_data, columns=imu_cols)
imu_df['acc_magnitude'] = acc_magnitude
imu_df['velocity'] = velocity
imu_df['jerk'] = jerk
correlation_matrix = imu_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title(f'Correlation Heatmap of IMU Features for Sequence {sample_seq_id}')



selected_seq_id = train['sequence_id'].unique()[4]  # Get 5th sequence (index 4)
seq_subset = train[train['sequence_id'] == selected_seq_id]
plot_3d = px.scatter_3d(
    seq_subset,
    x='acc_x',
    y='acc_y',
    z='acc_z',
    title=f'3D Acceleration Axes for Sequence {selected_seq_id}',
    color='sequence_type',
    size_max=8,
    width=1000,
    height=800,
    opacity=0.9,
    template='plotly_dark'
)
plot_3d.update_layout(
    font=dict(size=8),
    legend=dict(font=dict(size=16))
)
plot_3d.write_html('/kaggle/working/accel_3d_plot.html')


fig = px.scatter_3d(
    train,
    x="acc_x",
    y="acc_y",
    z="acc_z",
    color="sequence_type",
    size_max=8,
    opacity=0.9,
    title="Linear acceleration along three axes",
    width=1000,
    height=800,
    template="plotly_dark"
)

# TWEAK FONT SIZES
fig.update_layout(
    font_size=8,
    legend_font_size=16
)

fig.show()


# Time-series plot for IMU data
plt.figure(figsize=(10, 6))
plt.plot(sample_seq['sequence_counter'], sample_seq['acc_x'], label='acc_x')
plt.plot(sample_seq['sequence_counter'], sample_seq['acc_y'], label='acc_y')
plt.plot(sample_seq['sequence_counter'], sample_seq['acc_z'], label='acc_z')

plt.title(f'IMU Acceleration for Sequence {sample_seq_id}')
plt.xlabel('Sequence Counter')
plt.ylabel('Acceleration (m/s²)')
plt.legend()

# Display the plot in the notebook
plt.show()

# Save the plot
plt.savefig('/kaggle/working/imu_plot.png')
plt.close()


# Prepare labels
binary_le = LabelEncoder()
binary_labels = binary_le.fit_transform(train['sequence_type'])
gesture_map = {
    'Above ear - Pull hair': 0,
    'Forehead - Pull hairline': 1,
    'Forehead - Scratch': 2,
    'Eyebrow - Pull hair': 3,
    'Eyelash - Pull hair': 4,
    'Neck - Pinch skin': 5,
    'Neck - Scratch': 6,
    'Cheek - Pinch skin': 7
}
multiclass_labels = train['gesture'].apply(lambda x: gesture_map.get(x, 8)).values  


# Preprocess demographics
scaler = StandardScaler()
demo_cols = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
train_demo_data = scaler.fit_transform(train_demographics[demo_cols])
train_demo_cat = train_demographics[['adult_child', 'sex', 'handedness']].values
train_demo_processed = np.concatenate([train_demo_data, train_demo_cat], axis=1)
test_demo_data = scaler.transform(test_demographics[demo_cols])
test_demo_cat = test_demographics[['adult_child', 'sex', 'handedness']].values
test_demo_processed = np.concatenate([test_demo_data, test_demo_cat], axis=1)


# Preprocess training sequences
train_sequences = []
train_demo_final = []
train_binary_final = []
train_multiclass_final = []

for seq_id in train['sequence_id'].unique():
    seq_data = train[train['sequence_id'] == seq_id]
    is_imu_only = 'thm_1' not in seq_data.columns or seq_data['thm_1'].isna().all()
    
    # IMU features
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    imu_data = seq_data[imu_cols].fillna(0).values
    acc_magnitude = np.sqrt((seq_data[['acc_x', 'acc_y', 'acc_z']] ** 2).sum(axis=1))
    velocity = np.cumsum(acc_magnitude) / len(acc_magnitude)
    jerk = np.diff(acc_magnitude, prepend=acc_magnitude.iloc[0])  # Fix: Use .iloc[0]
    imu_data = np.column_stack([imu_data, acc_magnitude, velocity, jerk])
    
    # Thermopile and ToF
    if is_imu_only:
        thermo_data = np.zeros((len(seq_data), 10))  # 5 raw + 5 differences
        tof_data = np.zeros((len(seq_data), 15))  # 5 mean + 5 max + 5 gradient
    else:
        thermo_cols = [f'thm_{i}' for i in range(1, 6)]
        thermo_data = seq_data[thermo_cols].fillna(0).values
        thermo_diff = np.diff(thermo_data, axis=0, prepend=thermo_data[0:1])
        thermo_data = np.concatenate([thermo_data, thermo_diff], axis=1)
        
        tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
        tof_data = seq_data[tof_cols].replace(-1, 0).fillna(0).values
        tof_reshaped = tof_data.reshape(-1, 5, 8, 8)
        tof_mean = tof_reshaped.mean(axis=(2, 3))
        tof_max = tof_reshaped.max(axis=(2, 3))
        tof_gradient = np.abs(tof_reshaped[:, :, :-1, :-1] - tof_reshaped[:, :, 1:, 1:]).mean(axis=(2, 3))
        tof_data = np.concatenate([tof_mean, tof_max, tof_gradient], axis=1)
    
    sensor_data = np.concatenate([imu_data, thermo_data, tof_data], axis=1)
    train_sequences.append(sensor_data)
    
    subject = seq_data['subject'].iloc[0]
    demo_idx = train_demographics[train_demographics['subject'] == subject].index[0]
    train_demo_final.append(train_demo_processed[demo_idx])
    
    binary_label = binary_le.transform([seq_data['sequence_type'].iloc[0]])[0]
    gesture = seq_data['gesture'].iloc[0]
    multiclass_label = gesture_map.get(gesture, 8)
    train_binary_final.append(binary_label)
    train_multiclass_final.append(multiclass_label)


# Pad training sequences
max_len = max(len(seq) for seq in train_sequences)
train_sequences_padded = np.array([np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant') for seq in train_sequences])
train_demo_final = np.array(train_demo_final)
train_binary_final = np.array(train_binary_final)
train_multiclass_final = np.array(train_multiclass_final)


# Preprocess test sequences
test_sequences = []
test_demo_final = []


for seq_id in test['sequence_id'].unique():
    seq_data = test[test['sequence_id'] == seq_id]
    is_imu_only = 'thm_1' not in seq_data.columns or seq_data['thm_1'].isna().all()
    
    # IMU features
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    imu_data = seq_data[imu_cols].fillna(0).values
    acc_magnitude = np.sqrt((seq_data[['acc_x', 'acc_y', 'acc_z']] ** 2).sum(axis=1))
    velocity = np.cumsum(acc_magnitude) / len(acc_magnitude)
    jerk = np.diff(acc_magnitude, prepend=acc_magnitude.iloc[0])  # Fix: Use .iloc[0]
    imu_data = np.column_stack([imu_data, acc_magnitude, velocity, jerk])
    
    # Thermopile and ToF
    if is_imu_only:
        thermo_data = np.zeros((len(seq_data), 10))
        tof_data = np.zeros((len(seq_data), 15))
    else:
        thermo_cols = [f'thm_{i}' for i in range(1, 6)]
        thermo_data = seq_data[thermo_cols].fillna(0).values
        thermo_diff = np.diff(thermo_data, axis=0, prepend=thermo_data[0:1])
        thermo_data = np.concatenate([thermo_data, thermo_diff], axis=1)
        
        tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
        tof_data = seq_data[tof_cols].replace(-1, 0).fillna(0).values
        tof_reshaped = tof_data.reshape(-1, 5, 8, 8)
        tof_mean = tof_reshaped.mean(axis=(2, 3))
        tof_max = tof_reshaped.max(axis=(2, 3))
        tof_gradient = np.abs(tof_reshaped[:, :, :-1, :-1] - tof_reshaped[:, :, 1:, 1:]).mean(axis=(2, 3))
        tof_data = np.concatenate([tof_mean, tof_max, tof_gradient], axis=1)
    
    sensor_data = np.concatenate([imu_data, thermo_data, tof_data], axis=1)
    test_sequences.append(sensor_data)
    
    subject = seq_data['subject'].iloc[0]
    demo_idx = test_demographics[test_demographics['subject'] == subject].index[0]
    test_demo_final.append(test_demo_processed[demo_idx])


# Pad test sequences
test_sequences_padded = np.array([np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant') for seq in test_sequences])
test_demo_final = np.array(test_demo_final)


sensor_input = Input(shape=(train_sequences_padded.shape[1], train_sequences_padded.shape[2]), name='sensor_input')
masked = Masking(mask_value=0.0)(sensor_input)
lstm1 = LSTM(128, return_sequences=True)(masked)
lstm2 = LSTM(64)(lstm1)
dropout1 = Dropout(0.3)(lstm2)
demo_input = Input(shape=(train_demo_final.shape[1],), name='demo_input')
combined = Concatenate()([dropout1, demo_input])
dense = Dense(64, activation='relu')(combined)
dropout2 = Dropout(0.3)(dense)
binary_output = Dense(1, activation='sigmoid', name='binary_output')(dropout2)
multiclass_output = Dense(9, activation='softmax', name='multiclass_output')(dropout2)
model = Model(inputs=[sensor_input, demo_input], outputs=[binary_output, multiclass_output])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'binary_output': 'binary_crossentropy', 'multiclass_output': 'sparse_categorical_crossentropy'},
    loss_weights={'binary_output': 0.5, 'multiclass_output': 0.5},
    metrics={'binary_output': 'accuracy', 'multiclass_output': 'accuracy'}
)


# Train model
model.fit(
    [train_sequences_padded, train_demo_final],
    [train_binary_final, train_multiclass_final],
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)


def generate_submission():
    predictions = []
    inverse_gesture_map = {v: k for k, v in gesture_map.items()}
    inverse_gesture_map[8] = 'non_target'

    for i, seq_id in enumerate(test['sequence_id'].unique()):
        seq_data = test_sequences_padded[i:i+1]
        demo_data = test_demo_final[i:i+1]
        binary_pred, multiclass_pred = model.predict([seq_data, demo_data], verbose=0)
        gesture_idx = np.argmax(multiclass_pred, axis=1)[0]
        gesture = inverse_gesture_map[gesture_idx]
        predictions.append({'sequence_id': seq_id, 'gesture': gesture})

    submission = pd.DataFrame(predictions)
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)



import flask
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

app = flask.Flask(__name__)

model = model  
scaler = scaler  
binary_le = binary_le  
gesture_map = gesture_map 

@app.route('/health', methods=['GET'])
def health_check():
    """Required health check endpoint (Kaggle tests this)."""
    return 'OK', 200

@app.route('/infer', methods=['POST'])
def infer():
    """Main inference endpoint (Kaggle sends test data here)."""
    data = flask.request.json
    
    test_df = pd.DataFrame(data)
    test_demographics_df = pd.DataFrame(data["demographics"])  
    
    test_sequences = []
    test_demo_final = []
    
    for seq_id in test_df['sequence_id'].unique():
        seq_data = test_df[test_df['sequence_id'] == seq_id]
        is_imu_only = 'thm_1' not in seq_data.columns or seq_data['thm_1'].isna().all()
        
        imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
        imu_data = seq_data[imu_cols].fillna(0).values
        acc_magnitude = np.sqrt((seq_data[['acc_x', 'acc_y', 'acc_z']] ** 2).sum(axis=1))
        velocity = np.cumsum(acc_magnitude) / len(acc_magnitude)
        jerk = np.diff(acc_magnitude, prepend=acc_magnitude.iloc[0])
        imu_data = np.column_stack([imu_data, acc_magnitude, velocity, jerk])
        
        if is_imu_only:
            thermo_data = np.zeros((len(seq_data), 10))
            tof_data = np.zeros((len(seq_data), 15))
        else:
            thermo_cols = [f'thm_{i}' for i in range(1, 6)]
            thermo_data = seq_data[thermo_cols].fillna(0).values
            thermo_diff = np.diff(thermo_data, axis=0, prepend=thermo_data[0:1])
            thermo_data = np.concatenate([thermo_data, thermo_diff], axis=1)
            
            tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
            tof_data = seq_data[tof_cols].replace(-1, 0).fillna(0).values
            tof_reshaped = tof_data.reshape(-1, 5, 8, 8)
            tof_mean = tof_reshaped.mean(axis=(2, 3))
            tof_max = tof_reshaped.max(axis=(2, 3))
            tof_gradient = np.abs(tof_reshaped[:, :, :-1, :-1] - tof_reshaped[:, :, 1:, 1:]).mean(axis=(2, 3))
            tof_data = np.concatenate([tof_mean, tof_max, tof_gradient], axis=1)
        
        sensor_data = np.concatenate([imu_data, thermo_data, tof_data], axis=1)
        test_sequences.append(sensor_data)
        
        subject = seq_data['subject'].iloc[0]
        demo_idx = test_demographics_df[test_demographics_df['subject'] == subject].index[0]
        test_demo_final.append(test_demo_processed[demo_idx])  # Adjust if needed
    
    test_sequences_padded = np.array([np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant') for seq in test_sequences])
    test_demo_final = np.array(test_demo_final)
    
    binary_pred, multiclass_pred = model.predict([test_sequences_padded, test_demo_final], verbose=0)
    
    inverse_gesture_map = {v: k for k, v in gesture_map.items()}
    inverse_gesture_map[8] = 'non_target'
    
    results = []
    for i, seq_id in enumerate(test_df['sequence_id'].unique()):
        gesture_idx = np.argmax(multiclass_pred[i])
        gesture = inverse_gesture_map[gesture_idx]
        results.append({"sequence_id": seq_id, "gesture": gesture})
    
    return flask.jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  

