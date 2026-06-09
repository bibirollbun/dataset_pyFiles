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


# import numpy as np
# import pandas as pd
# import joblib
# from tensorflow.keras.models import load_model

# # ===============================
# # 2️⃣ Load Models + Scalers
# # ===============================
# # --- MULTI ---
# model_multi = load_model('/kaggle/input/main-data/multi_model.h5')
# scaler_multi = joblib.load('/kaggle/input/main-data/multi_scaler.pkl')

# # --- BINARY ---
# clf_binary = joblib.load('/kaggle/input/main-data/binary_model.pkl')
# scaler_binary = joblib.load('/kaggle/input/main-data/binary_scaler.pkl')

# print("✅ Models & scalers loaded!")


# test = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
# test_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
# test_df = pd.merge(test, test_demo, on="subject", how="inner")


# acc_cols = [col for col in test_df.columns if col.startswith("acc_")]
# rot_cols = [col for col in test_df.columns if col.startswith("rot_")]
# tuf_cols = [col for col in test_df.columns if col.startswith("tof_")]
# thm_cols = [col for col in test_df.columns if col.startswith("thm_")]
# demo_cols = [
#     'adult_child', 'age', 'sex', 'handedness',
#     'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'
# ]

# label_map = {
#     0: 'Above ear - pull hair',
#     1: 'Cheek - pinch skin',
#     2: 'Eyebrow - pull hair',
#     3: 'Eyelash - pull hair',
#     4: 'Forehead - pull hairline',
#     5: 'Forehead - scratch',
#     6: 'Neck - pinch skin',
#     7: 'Neck - scratch'
# }


# # ===============================
# # 5️⃣ Precompute row means
# # ===============================
# test_df['tuf_mean_row'] = test_df[tuf_cols].replace(-1, np.nan).mean(axis=1)
# test_df['thm_mean_row'] = test_df[thm_cols].mean(axis=1)

# input_features = acc_cols + rot_cols + demo_cols + ['tuf_mean_row', 'thm_mean_row']
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # If you need to know max_len: you MUST store it during training!
# max_len = 693  # <- example, use YOUR actual max_len!

# multi_sequences = []
# multi_ids = []

# for seq_id, group in test_df.groupby("sequence_id"):
#     group = group.sort_values("sequence_counter")
#     data = group[input_features].values
#     multi_sequences.append(data)
#     multi_ids.append(seq_id)

# X_test_padded = pad_sequences(multi_sequences, maxlen=max_len, padding='post', dtype='float32')
# X_test_scaled_multi = np.array([scaler_multi.transform(seq) for seq in X_test_padded])
# X_test_scaled_multi = X_test_scaled_multi[..., np.newaxis]

# # ===============================
# # 7️⃣ Predict MULTI
# # ===============================
# y_pred_probs = model_multi.predict(X_test_scaled_multi)
# y_pred_multi = np.argmax(y_pred_probs, axis=1)

# gesture_pred_multi = [label_map[idx] for idx in y_pred_multi]



# test_df_2 = pd.merge(test,test_demo, on="subject", how="inner")
# FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']
# DEMO_COLUMNS = demo_cols

# # Aggregate per sequence
# rows = []
# for seq_id, group in test_df_2.groupby('sequence_id'):
#     feature_means = group[FEATURE_COLUMNS].mean().to_dict()
#     demo_values = group[DEMO_COLUMNS].iloc[0].to_dict()
#     combined_row = {**feature_means, **demo_values, 'sequence_id': seq_id}
#     rows.append(combined_row)

# test_seq_df = pd.DataFrame(rows)
# X_test_bi = test_seq_df[FEATURE_COLUMNS + DEMO_COLUMNS]
# X_test_scaled_bi = scaler_binary.transform(X_test_bi)

# # Predict
# gesture_pred_bi = clf_binary.predict(X_test_scaled_bi)


# gesture_pred_bi_label = ['Target' if p == 1 else 'Non-Target' for p in gesture_pred_bi]


# # Make DataFrame with BOTH predictions
# final_df = pd.DataFrame({
#     'sequence_id': multi_ids,
#     'gesture_predicted': gesture_pred_multi,     # multi-class gesture names
#     'gesture_type_predicted': gesture_pred_bi_label  # Target or Non-Target
# })

# # If binary = Non-Target → override gesture_predicted to "Non-Target"
# final_df.loc[final_df['gesture_type_predicted'] == 'Non-Target', 'gesture_predicted'] = 'Non-Target'



# final_df[['sequence_id', 'gesture_predicted']].to_csv('submission.csv', index=False)
# print("✅ Submission file saved: submission.csv")


# final_df


# import os

# print("Does submission.csv exist?", os.path.exists("submission.csv"))
# print("Current working dir:", os.getcwd())
# print("Files here:", os.listdir("."))



import os
import pandas as pd
import polars as pl
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pickle  # or joblib for scikit-learn model

from sklearn.preprocessing import StandardScaler

import kaggle_evaluation.cmi_inference_server

# 1️⃣ Load both models & scalers once
model_multi = load_model('/kaggle/input/main-data/multi_model.h5')
scaler_multi = joblib.load('/kaggle/input/main-data/multi_scaler.pkl')

import joblib
clf_binary = joblib.load('/kaggle/input/main-data/binary_model.pkl')
scaler_binary = joblib.load('/kaggle/input/main-data/binary_scaler.pkl')

# 2️⃣ Label mapping for multi-class
label_map = {
    0: "Above ear - pull hair",
    1: "Cheek - pinch skin",
    2: "Eyebrow - pull hair",
    3: "Eyelash - pull hair",
    4: "Forehead - pull hairline",
    5: "Forehead - scratch",
    6: "Neck - pinch skin",
    7: "Neck - scratch"
}



# 4️⃣ Your inference function
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # Convert to pandas for numpy ops
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()
    # 3️⃣ Columns
    acc_cols = [col for col in sequence.columns if col.startswith("acc_")]
    rot_cols = [col for col in sequence.columns if col.startswith("rot_")]
    tuf_cols = [col for col in sequence.columns if col.startswith("tof_")]
    thm_cols = [col for col in sequence.columns if col.startswith("thm_")]
    demo_cols = ['adult_child', 'age', 'sex', 'handedness',
                 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']

    FEATURE_COLUMNS = acc_cols + rot_cols

    # --- Binary prediction ---
    # Get means
    feature_means = seq_df[FEATURE_COLUMNS].mean()
    demo_values = demo_df.iloc[0][demo_cols]

    X_bi = pd.concat([feature_means, demo_values]).values.reshape(1, -1)
    X_bi_scaled = scaler_binary.transform(X_bi)
    pred_bi = clf_binary.predict(X_bi_scaled)[0]  # 0 or 1

    if pred_bi == 0:
        return "Non-Target"

    # --- Multi gesture prediction ---
    # For multi: sequence-level IMU + demo + mean(TUF) + mean(THM)

    seq_df['tuf_mean_row'] = seq_df[tuf_cols].replace(-1, np.nan).mean(axis=1)
    seq_df['thm_mean_row'] = seq_df[thm_cols].mean(axis=1)

    input_features = acc_cols + rot_cols + demo_cols + ['tuf_mean_row', 'thm_mean_row']

    group = seq_df.copy()
    group[demo_cols] = demo_df.iloc[0][demo_cols].values

    data = group[input_features].values

    # Pad to max_len you used in training
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    max_len = 693  # or your actual max_len from training
    padded = pad_sequences([data], maxlen=max_len, padding='post', dtype='float32')[0]

    scaled = scaler_multi.transform(padded)
    scaled = np.expand_dims(scaled, axis=0)  # batch
    scaled = scaled[..., np.newaxis]  # add channel

    y_probs = model_multi.predict(scaled)
    y_class = np.argmax(y_probs, axis=1)[0]

    gesture_label = label_map[y_class]
    return gesture_label

# 5️⃣ Run server
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



# predictions = []
# for seq_id, group in test_df.groupby('sequence_id'):
#     pred = predict(pl.DataFrame(group), pl.DataFrame(test_demo[test_demo['subject'] == group['subject'].iloc[0]]))
#     predictions.append((seq_id, pred))

# pred_df_tr = pd.DataFrame(predictions, columns=['sequence_id', 'gesture_predicted'])
# # pred_df_tr.to_csv('submission.csv', index=False)
# # print("Saved local test predictions as submission.csv")



# pred_df_tr




