import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import os
import random
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers, optimizers, metrics, callbacks
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.initializers import HeNormal, Zeros
from tensorflow.keras import optimizers, losses, metrics, callbacks


targets = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
fmri = pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")

quants = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
quants.set_index("participant_id", inplace=True)



cat = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
cat.set_index("participant_id", inplace=True)

cat.Barratt_Barratt_P1_Occ = cat.Barratt_Barratt_P1_Occ//5
cat.Barratt_Barratt_P2_Occ = cat.Barratt_Barratt_P2_Occ//5
cat.Barratt_Barratt_P1_Edu = cat.Barratt_Barratt_P1_Edu//3
cat.Barratt_Barratt_P2_Edu = cat.Barratt_Barratt_P2_Edu//3

fmri.set_index("participant_id", inplace=True)
targets.set_index("participant_id", inplace=True)



train = fmri.join(quants).join(cat).join(targets)
train = train.drop(columns = ["MRI_Track_Scan_Location","Basic_Demos_Enroll_Year","Basic_Demos_Study_Site"])


fmriTest = pd.read_csv("/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
quantTest = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
fmriTest.set_index("participant_id", inplace=True)
quantTest.set_index("participant_id", inplace=True)



catTest = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
catTest.set_index("participant_id", inplace=True)

catTest.Barratt_Barratt_P1_Edu = catTest.Barratt_Barratt_P1_Edu//3
catTest.Barratt_Barratt_P2_Edu = catTest.Barratt_Barratt_P2_Edu//3
catTest.Barratt_Barratt_P1_Occ = catTest.Barratt_Barratt_P1_Occ//5
catTest.Barratt_Barratt_P2_Occ = catTest.Barratt_Barratt_P2_Occ//5



test = fmriTest.join(quantTest).join(catTest)
test = test.drop(columns = ["MRI_Track_Scan_Location","Basic_Demos_Enroll_Year","Basic_Demos_Study_Site"])


# 1) Load atlas mapping
file_id = "1EP7KzgEhsiqOg7bvfDYSYOg1CZA7pVnt"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

atlas = pd.read_csv(url)   # your CSV file
# build ROI→network dict; note ROI Label runs 1–200
roi2net = atlas.set_index("ROI Label")["Network Name"].to_dict()

upper,lower = np.triu_indices(200,k=1)

mat = np.zeros((200,200))
mat[upper,lower] = fmri.iloc[0].values

# 2) Create list of network labels in ROI order (0‑indexed for numpy)
labels = [roi2net[i].strip() for i in range(1, 201)]
networks = sorted(set(labels))

# 3) Precompute indices for each network
net_idx = {net: [k for k,lab in enumerate(labels) if lab==net]
           for net in networks}

col_names = []
for i,net_i in enumerate(networks,1):
    for net_j in networks[i:]:
        col_names.append(f"{net_i}_{net_j}")


network_conn_df = pd.DataFrame(index=fmri.index, columns=col_names, dtype=float)

u,l = np.triu_indices(17,k=1)

for index in fmri.index:
  net_fc = pd.DataFrame(index=networks, columns=networks, dtype=float)
  mat = np.zeros((200,200))
  mat[upper,lower] = fmri.loc[index].values
  mat = mat + mat.T
  np.fill_diagonal(mat,1)
  for net_i in networks:
      for net_j in networks:
          idx_i = net_idx[net_i]
          idx_j = net_idx[net_j]
          # extract the submatrix of all ROI‑ROI edges between these networks
          submat = mat[np.ix_(idx_i, idx_j)]
          # average them (for within‑network, this includes the diagonal block)
          net_fc.loc[net_i, net_j] = submat.mean()
          network_conn_df.loc[index] = net_fc.to_numpy()[u,l]



network_conn_df_test = pd.DataFrame(index=test.index, columns=col_names, dtype=float)

for index in test.index:
  net_fc = pd.DataFrame(index=networks, columns=networks, dtype=float)
  mat = np.zeros((200,200))
  mat[upper,lower] = test.loc[index][fmri.columns].values
  mat = mat + mat.T
  np.fill_diagonal(mat,1)
  for net_i in networks:
      for net_j in networks:
          idx_i = net_idx[net_i]
          idx_j = net_idx[net_j]
          # extract the submatrix of all ROI‑ROI edges between these networks
          submat = mat[np.ix_(idx_i, idx_j)]
          # average them (for within‑network, this includes the diagonal block)
          net_fc.loc[net_i, net_j] = submat.mean()
          network_conn_df_test.loc[index] = net_fc.to_numpy()[u,l]



X = train.drop(columns=["Sex_F","ADHD_Outcome"])
X = X.join(network_conn_df)
y = train.Sex_F

non_fmri = list(set(X.columns) - set(fmri.columns) - set(col_names))
imputer = KNNImputer(n_neighbors=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train[non_fmri] = imputer.fit_transform(X_train[non_fmri]).round().astype(int)
X_test[non_fmri] = imputer.transform(X_test[non_fmri]).round().astype(int)

test_net = test.join(network_conn_df_test)
test_net[non_fmri] = imputer.transform(test_net[non_fmri]).round().astype(int)


final_train = pd.concat([X_train,X_test],axis=0)
final_train_labels = pd.concat([y_train,y_test],axis=0)


import tensorflow as tf
print("Built with CUDA:", tf.test.is_built_with_cuda())            # True = GPU-enabled build
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))


SEED = 46
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enforce determinism (TF-2.12+ covers both CPU & GPU deterministic kernels)
tf.config.experimental.enable_op_determinism()


class DistillationModel(tf.keras.Model):
    def __init__(self, input_dim):
        super().__init__()
        self.dense1 = layers.Dense(
            512, activation='relu',
            kernel_initializer=initializers.HeNormal(seed=SEED),
            bias_initializer=initializers.Zeros(),
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
        )
        self.drop1 = layers.Dropout(0.3, seed=SEED)

        self.dense2 = layers.Dense(
            256, activation='relu',
            kernel_initializer=initializers.HeNormal(seed=SEED+1),
            bias_initializer=initializers.Zeros(),
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
        )
        self.drop2 = layers.Dropout(0.2, seed=SEED+1)

        self.dense3 = layers.Dense(
            128, activation='relu',
            kernel_initializer=initializers.HeNormal(seed=SEED+2),
            bias_initializer=initializers.Zeros(),
        )
        self.dense4 = layers.Dense(
            64, activation='relu',
            kernel_initializer=initializers.HeNormal(seed=SEED+3),
            bias_initializer=initializers.Zeros(),
        )
        self.logit = layers.Dense(
            1, activation=None,
            kernel_initializer=initializers.HeNormal(seed=SEED+4),
            bias_initializer=initializers.Zeros(),
        )

        # calibrated weights
        self.alpha = self.add_weight(
            name="alpha", shape=(), initializer="ones", trainable=True
        )
        self.beta = self.add_weight(
            name="beta", shape=(), initializer="zeros", trainable=True
        )

    def call(self, x, training=False):
        x = self.dense1(x); x = self.drop1(x, training=training)
        x = self.dense2(x); x = self.drop2(x, training=training)
        x = self.dense3(x); x = self.dense4(x)

        logits = self.logit(x)
        calibrated = logits * self.alpha + self.beta
        return tf.sigmoid(calibrated)




def focal_plus_entropy(y_true, y_pred):
    focal = tf.keras.losses.BinaryFocalCrossentropy()(y_true, y_pred)
    eps = 1e-7
    ent = -(y_pred * tf.math.log(y_pred + eps) +
            (1 - y_pred) * tf.math.log(1 - y_pred + eps))
    entropy_penalty = tf.reduce_mean(ent)
    return focal - 1e-3 * entropy_penalty




model = DistillationModel(input_dim=final_train.shape[1])
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss=focal_plus_entropy,
    metrics=[
        metrics.BinaryAccuracy(name='accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc'),
    ]
)

history = model.fit(
    final_train, final_train_labels,
    validation_data=(final_train, final_train_labels),
    epochs=11,
    batch_size=32,
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
    ],
    verbose=1,
)


# --- inspect the new distribution & threshold sweep ---


y_prob = model.predict(final_train).flatten()

import matplotlib.pyplot as plt
plt.hist(y_prob, bins=30)
plt.title("Smoothed Probability Distribution")
plt.show()

from sklearn.metrics import accuracy_score
for thr in [0.05, 0.1, 0.14, 0.2, 0.5, 0.9]:
    print(f"Thr={thr:.2f}", "Acc=", accuracy_score(final_train_labels, (y_prob>thr).astype(int)))

from sklearn.metrics import accuracy_score
for thr in np.linspace(0.01, 0.99, 99):
    acc = accuracy_score(final_train_labels, (y_prob>thr).astype(int))
    if acc > 0.975:
      print(f"Thr={thr:.2f}", "Acc=", acc)

# --- inspect the new distribution & threshold sweep ---


from sklearn.metrics import accuracy_score
for thr in np.linspace(0.01, 0.99, 99):
    acc = accuracy_score(final_train_labels, (y_prob>thr).astype(int))
    if acc > 0.95:
      print(f"Thr={thr:.2f}", "Acc=", acc)


y_soft = model.predict(final_train, batch_size=32).flatten()

y_s = pd.Series(y_soft, index= final_train.index)

y_train_soft = y_s.loc[X_train.index]
y_test_soft = y_s.loc[X_test.index]


student = Sequential([
    Input(shape=(X_train.shape[1],), name="input"),

    Dense(
      512, activation='relu',
      kernel_regularizer=l1_l2(1e-5,1e-4),
      kernel_initializer=HeNormal(seed=SEED+1),
      bias_initializer=Zeros(),
      name="dense1"
    ),
    Dropout(0.3, seed=SEED+1, name="drop1"),

    Dense(
      256, activation='relu',
      kernel_regularizer=l1_l2(1e-5,1e-4),
      kernel_initializer=HeNormal(seed=SEED+2),
      bias_initializer=Zeros(),
      name="dense2"
    ),
    Dropout(0.2, seed=SEED+2, name="drop2"),

    Dense(
      128, activation='relu',
      kernel_initializer=HeNormal(seed=SEED+3),
      bias_initializer=Zeros(),
      name="dense3"
    ),

    Dense(
      64, activation='relu',
      kernel_initializer=HeNormal(seed=SEED+1),
      bias_initializer=Zeros(),
      name="dense4"),

    Dense(
      1, activation='sigmoid',
      kernel_initializer=HeNormal(seed=SEED+2),
      bias_initializer=Zeros(),
      name="output"
    ),
])

# ─── 4) COMPILE ─────────────────────────────────────────────────────────────────
student.compile(
    optimizer=optimizers.Adam(1e-4),
    loss=losses.MeanSquaredError,
    metrics=[losses.MeanSquaredError(name='mse')]
)



history = student.fit(
    X_train, y_train_soft,
    validation_data=(X_test, y_test_soft),
    epochs=100,
    callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)



import keras
finalStudent = keras.models.load_model("/kaggle/input/s-neural-network/keras/default/1/BEST_S_ANN.keras")


y_prob = finalStudent.predict(test_net.astype(np.float32)).flatten()


sexPreds = (y_prob > 0.1942105).astype(int)
print(pd.Series(sexPreds).value_counts())


import pandas as pd
import numpy as np

targets = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
fmri = pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
quants = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
cat = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
fmriTest = pd.read_csv("/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
quantTest = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
catTest = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize class balance
sns.countplot(data=train, x="ADHD_Outcome")
plt.title("ADHD Outcome Class Balance (Train Set)")
plt.xlabel("ADHD Diagnosis (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


import math

quant_cols = quants.columns
n = len(quant_cols)

n_cols = 4
n_rows = math.ceil(n / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for ax, col in zip(axes, quant_cols):
    try:
        sns.histplot(train[col], bins=20, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    except:
        continue

# turn off any unused subplots
for ax in axes[n:]:
    try:
        ax.axis("off")
    except:
        continue

try:
    plt.tight_layout()
    plt.show()
except:
    pass


# Visualize the correlation between selected features and ADHD outcome
corr_features = ["SDQ_SDQ_Emotional_Problems", "SDQ_SDQ_Hyperactivity", "SDQ_SDQ_Externalizing", "SDQ_SDQ_Internalizing", "ADHD_Outcome"]
sns.heatmap(train[corr_features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Behavioral Scores and ADHD Outcome")
plt.show()


quants.set_index("participant_id", inplace=True)
cat.set_index("participant_id", inplace=True)
fmri.set_index("participant_id", inplace=True)
targets.set_index("participant_id", inplace=True)

cat.Barratt_Barratt_P1_Occ //= 5
cat.Barratt_Barratt_P2_Occ //= 5
cat.Barratt_Barratt_P1_Edu //= 3
cat.Barratt_Barratt_P2_Edu //= 3

train = fmri.join(quants).join(cat).join(targets)
train.drop(columns=["MRI_Track_Scan_Location", "Basic_Demos_Enroll_Year", "Basic_Demos_Study_Site"], inplace=True)

fmriTest.set_index("participant_id", inplace=True)
quantTest.set_index("participant_id", inplace=True)
catTest.set_index("participant_id", inplace=True)

catTest.Barratt_Barratt_P1_Edu //= 3
catTest.Barratt_Barratt_P2_Edu //= 3
catTest.Barratt_Barratt_P1_Occ //= 5
catTest.Barratt_Barratt_P2_Occ //= 5

test = fmriTest.join(quantTest).join(catTest)
test.drop(columns=["MRI_Track_Scan_Location", "Basic_Demos_Enroll_Year", "Basic_Demos_Study_Site"], inplace=True)


from sklearn.impute import KNNImputer
from sklearn.decomposition import FastICA, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


fmri_cols = list(fmri.columns)
non_fmri_cols = [col for col in train.columns if col not in fmri_cols]

train_set, test_set = train_test_split(train, test_size=0.2, random_state=42)
train_fmri = train_set[fmri_cols]
test_fmri = test_set[fmri_cols]

n_ica = 20
X_train_centered = train_fmri - train_fmri.mean()
X_test_centered = test_fmri - train_fmri.mean()

pca = PCA(n_components=0.99, random_state=42)
X_train_pca = pca.fit_transform(X_train_centered)
X_test_pca = pca.transform(X_test_centered)

ica = FastICA(n_components=n_ica, random_state=42, tol=1e-1, max_iter=10000)
X_train_ica = ica.fit_transform(X_train_pca)
X_test_ica = ica.transform(X_test_pca)

ica_cols = [f"ICA_{i}" for i in range(n_ica)]
train_ica = pd.DataFrame(X_train_ica, columns=ica_cols, index=train_set.index)
test_ica = pd.DataFrame(X_test_ica, columns=ica_cols, index=test_set.index)


train_final = train_ica.join(train_set[non_fmri_cols])
test_final = test_ica.join(test_set[non_fmri_cols])

non_ica_cols = [col for col in train_final.columns if not col.startswith("ICA_") and col not in ["Sex_F", "ADHD_Outcome"]]
imputer = KNNImputer(n_neighbors=5)
imputer.fit(train_final[non_ica_cols])

def impute_and_round(df):
    df[non_ica_cols] = imputer.transform(df[non_ica_cols])
    df[non_ica_cols] = df[non_ica_cols].round().astype(int)
    return df

train_final = impute_and_round(train_final)
test_final = impute_and_round(test_final)


X_train = train_final.drop(columns=["ADHD_Outcome", "Sex_F"])
y_train = train_final["ADHD_Outcome"]
X_test = test_final.drop(columns=["ADHD_Outcome", "Sex_F"])
y_test = test_final["ADHD_Outcome"]

model = LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy on test set: {acc:.4f}")


test_no_na = test.copy()
test_no_na[imputer.feature_names_in_] = imputer.transform(test[imputer.feature_names_in_])


test_ica = ica.fit_transform(test[fmri.columns])
test_ica = pd.DataFrame(test_ica, columns=ica_cols, index=test_no_na.index)
test_final = test_ica.join(test_no_na.drop(columns = fmri.columns))
test_final


probs = model.predict_proba(test_final)[:,1]


f"""use 0.65 to predict and submit"""


sub = pd.read_excel("/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx", index_col = "participant_id")
sub['ADHD_Outcome'] = (probs > 0.65 ).astype(int)
sub["Sex_F"] = sexPreds


sub["ADHD_Outcome"].value_counts()


sub.to_csv("submission.csv")




