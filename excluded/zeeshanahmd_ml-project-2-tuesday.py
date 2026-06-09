# ðŸ“¦ 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GRU, Bidirectional, GlobalMaxPooling1D, GaussianNoise, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

# ðŸ“‚ 2. Load Data (Replace with correct Kaggle paths)
train_meta = pd.read_csv("/kaggle/input/PLAsTiCC-2018/training_set_metadata.csv")
train_lc = pd.read_csv("/kaggle/input/PLAsTiCC-2018/training_set.csv")
# /kaggle/input/PLAsTiCC-2018

# âœ¨ Add the rest of the code from previous cells (Preprocessing + Models + Training + Confusion Matrix)
# Due to message limits, let me know if you'd like the remaining 150+ lines again â€” I'll send them in chunks.


import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

# 3DSubM Preprocessing
def preprocess_3dsubm(lc_df, max_seq_len=162):
    lc_df = lc_df.copy()
    lc_df['flux'] += np.random.normal(0, lc_df['flux_err'] * 2 / 3)
    flux_max, flux_min = lc_df['flux'].max(), lc_df['flux'].min()
    flux_scaler = 1 / np.log2(flux_max - flux_min + 1)
    lc_df['flux'] *= flux_scaler
    lc_df['mjd'] = lc_df.groupby('object_id')['mjd'].transform(lambda x: x - x.min())

    grouped_sequences = {}
    for obj_id, group in lc_df.groupby('object_id'):
        sequence, night_buffer, current_night = [], {}, -10
        for _, row in group.sort_values('mjd').iterrows():
            if row['mjd'] - current_night > 0.33:
                if night_buffer:
                    sequence.append([night_buffer.get(pb, np.nan) for pb in range(6)])
                night_buffer = {}
                current_night = row['mjd']
            night_buffer[row['passband']] = row['flux']
        if night_buffer:
            sequence.append([night_buffer.get(pb, np.nan) for pb in range(6)])

        df_seq = pd.DataFrame(sequence).interpolate(axis=1).fillna(0)
        seq_array = df_seq.values.tolist()
        if len(seq_array) < max_seq_len:
            seq_array += [[0]*6] * (max_seq_len - len(seq_array))
        grouped_sequences[obj_id] = np.array(seq_array[:max_seq_len])

    return grouped_sequences


# 2DSubM Preprocessing
def extract_aggregated_features(lc_df):
    features = []
    for obj_id, group in lc_df.groupby("object_id"):
        obj_feats = {"object_id": obj_id}
        for pb in range(6):
            pb_data = group[group["passband"] == pb]
            obj_feats[f"flux_mean_pb{pb}"] = pb_data["flux"].mean()
            obj_feats[f"flux_std_pb{pb}"] = pb_data["flux"].std()
            obj_feats[f"flux_max_pb{pb}"] = pb_data["flux"].max()
            obj_feats[f"flux_min_pb{pb}"] = pb_data["flux"].min()
        features.append(obj_feats)
    return pd.DataFrame(features)


def preprocess_2dsubm(lc_df, meta_df):
    agg_feats = extract_aggregated_features(lc_df)
    combined = pd.merge(agg_feats, meta_df, on="object_id")
    drop_cols = ['object_id', 'ra', 'decl', 'gal_l', 'gal_b']
    combined = combined.drop(columns=[col for col in drop_cols if col in combined.columns], errors='ignore')
    combined = combined.fillna(0)
    pt = PowerTransformer()
    combined_scaled = pd.DataFrame(pt.fit_transform(combined), columns=combined.columns)
    return combined_scaled


# ðŸ“¦ Import model layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GRU, Bidirectional, GlobalMaxPooling1D, GaussianNoise, Concatenate
from tensorflow.keras.optimizers import Adam

# ðŸ§  3DSubM: GRU-based model
def build_3dsubm_model(input_shape=(162, 6), num_classes=15):
    inp = Input(shape=input_shape)
    x = GaussianNoise(0.5)(inp)
    x = Bidirectional(GRU(256, return_sequences=True, activation='tanh'))(x)
    x = Dropout(0.1)(x)
    x = Bidirectional(GRU(64, return_sequences=True, activation='tanh'))(x)
    x = Dropout(0.1)(x)
    x = Bidirectional(GRU(32, return_sequences=True, activation='tanh'))(x)
    x = Dropout(0.1)(x)
    x = Bidirectional(GRU(32, return_sequences=True, activation='tanh'))(x)
    x = Dropout(0.1)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='tanh')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='tanh')(x)
    x = Dropout(0.1)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ðŸ§  2DSubM: Dense model
def build_2dsubm_model(input_shape, num_classes=15):
    inp = Input(shape=input_shape)
    x = Dense(512, activation='tanh')(inp)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='tanh')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='tanh')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='tanh')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ðŸ”€ Stacking Ensemble
def build_stacked_ensemble(three_d_model, two_d_model, num_classes=15):
    for layer in three_d_model.layers:
        layer.trainable = False
    for layer in two_d_model.layers:
        layer.trainable = False

    input_3d = Input(shape=(162, 6))
    input_2d = Input(shape=(two_d_model.input_shape[1],))
    output_3d = three_d_model(input_3d)
    output_2d = two_d_model(input_2d)
    x = Concatenate()([output_3d, output_2d])
    x = Dense(10, activation='relu')(x)
    final_output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[input_3d, input_2d], outputs=final_output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ðŸ”„ Preprocess input data
seqs = preprocess_3dsubm(train_lc)
X_seq = np.stack([seqs[oid] for oid in train_meta["object_id"] if oid in seqs])

X_dense_df = preprocess_2dsubm(train_lc, train_meta)
X_dense_df = X_dense_df.loc[train_meta["object_id"].isin(seqs)]
X_dense = X_dense_df.values

# ðŸŽ¯ Encode labels dynamically
from tensorflow.keras.utils import to_categorical

label_map = {val: idx for idx, val in enumerate(sorted(train_meta['target'].unique()))}
train_meta["target_id"] = train_meta["target"].map(label_map)
y = to_categorical(train_meta.loc[train_meta["object_id"].isin(seqs), "target_id"])

# Automatically match number of classes
num_classes = y.shape[1]

# âœ‚ 85:15 Train-validation split
from sklearn.model_selection import train_test_split

X_seq_train, X_seq_val, X_dense_train, X_dense_val, y_train, y_val = train_test_split(
    X_seq, X_dense, y, test_size=0.15, random_state=42
)

# ðŸ§  Train base models with correct output size
model_3d = build_3dsubm_model(num_classes=num_classes)
model_2d = build_2dsubm_model(input_shape=X_dense.shape[1:], num_classes=num_classes)

model_3d.fit(X_seq_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)
model_2d.fit(X_dense_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)

# ðŸ”€ Train stacking ensemble
ensemble = build_stacked_ensemble(model_3d, model_2d, num_classes=num_classes)
ensemble.fit([X_seq_train, X_dense_train], y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)

# ðŸ“Š Evaluate on validation set
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = ensemble.predict([X_seq_val, X_dense_val])
y_true = np.argmax(y_val, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_class)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix on Validation Set")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Detailed report
print(classification_report(y_true, y_pred_class))

