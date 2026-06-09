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


train_sol = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
train_cate = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
train_func_con = pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
train_quan = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")


"-----TRAINING SOLUTION-----\n", train_sol.head(), "-----TRAINING CATEGORICAL METADATA-----\n", train_cate.head(), "-----TRAIN FUNCTIONAL CONNECTOME-----\n", train_func_con.head(), "-----TRAIN QUANTITATIVE METADATA-----\n", train_quan.head()


def nan_summary(df, name):
    print(f"---- {name} NAN SUMMARY ----")
    print((df.isnull().sum() / len(df) * 100).sort_values(ascending=False), '\n')

nan_summary(train_sol, "TRAINING SOLUTION")
nan_summary(train_cate, "TRAINING CATEGORICAL METADATA")
nan_summary(train_func_con, "TRAIN FUNCTIONAL CONNECTOME")
nan_summary(train_quan, "TRAIN QUANTITATIVE METADATA")


"---- TRAIN QUANTITATIVE DESCRIBE ----\n", train_quan.describe().T  # Chá»©a dá»¯ liá»‡u Ä‘á»‹nh lÆ°á»£ng


import seaborn as sns
import matplotlib.pyplot as plt

def plot_categorical_distributions(df, max_unique=20):
    categorical_cols = df.columns

    for col in categorical_cols:
        num_unique = df[col].nunique()
        if num_unique <= max_unique:
            plt.figure(figsize=(10, 4))
            sns.countplot(data=df, x=col, order=df[col].value_counts().index)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"[âš ï¸� Bá»� qua] Cá»™t '{col}' cÃ³ {num_unique} giÃ¡ trá»‹ duy nháº¥t â€” quÃ¡ nhiá»�u Ä‘á»ƒ trá»±c quan rÃµ rÃ ng.")
for col in train_cate.columns:
    num_unique = train_cate[col].nunique()
    if num_unique <= 20:
            print(f"-----Distribution of {col}-----")
            print(train_cate[col].value_counts())
plot_categorical_distributions(train_cate)


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# BÆ°á»›c 1: Láº¥y dÃ²ng connectome Ä‘áº§u tiÃªn (bá»� participant_id)
participant_vec = train_func_con.drop(columns=['participant_id']).iloc[0].values.astype(float)

# BÆ°á»›c 2: Chuyá»ƒn vá»� ma tráº­n Ä‘á»‘i xá»©ng 200x200
def vector_to_symmetric_matrix(vec, N):
    mat = np.zeros((N, N))
    upper_tri_indices = np.triu_indices(N, k=1)
    mat[upper_tri_indices] = vec
    mat += mat.T  # Ä�á»‘i xá»©ng
    return mat

N = 200
conn_matrix = vector_to_symmetric_matrix(participant_vec, N)

# BÆ°á»›c 3: Váº½ heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conn_matrix, cmap='coolwarm', center=0)
plt.title("Functional Connectome Heatmap (1 Participant)")
plt.xlabel("ROI")
plt.ylabel("ROI")
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Váº½ phÃ¢n phá»‘i cho ADHD_Outcome
plt.figure(figsize=(12, 6))

# ADHD_Outcome
plt.subplot(1, 2, 1)
sns.countplot(data=train_sol, x='ADHD_Outcome', palette='Blues')
plt.title('Distribution of ADHD_Outcome')
plt.xlabel('ADHD_Outcome')
plt.ylabel('Count')

# Sex_F
plt.subplot(1, 2, 2)
sns.countplot(data=train_sol, x='Sex_F', palette='pastel')
plt.title('Distribution of Sex_F')
plt.xlabel('Sex_F')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

print("-----ADHD_Outcome label Distribution-----\n", train_sol["ADHD_Outcome"].value_counts())
print("-----Sex_F label Distribution-----\n",train_sol["Sex_F"].value_counts())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.merge(train_sol, train_cate, on="participant_id")
df = pd.merge(train_quan, df, on="participant_id")

df = df.drop(columns=["participant_id"])

correlation = df.corr()
correlation_with_sex_F = correlation['Sex_F'].sort_values(ascending=False)
correlation_with_ADHD_Outcome = correlation['ADHD_Outcome'].sort_values(ascending=False)

# Váº½ biá»ƒu Ä‘á»“
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_with_sex_F.to_frame(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation with sex_F")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_with_ADHD_Outcome.to_frame(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation with ADHD_Outcome")
plt.show()

print("-----Corr vs Sex_F-----\n", correlation_with_sex_F)
print("\n-----Corr vs ADHD_Outcome-----\n", correlation_with_ADHD_Outcome)



from scipy.stats import chi2_contingency
from scipy.stats import chi2_contingency

df_cate = pd.merge(train_cate, train_sol, on="participant_id") 
df_cate = df_cate.drop(columns=["participant_id"])

results_sex = []
results_adhd = []
for col in df_cate.columns:
    if col in ["Sex_F", "ADHD_Outcome"]:
        continue
        
    contingency_table_sex = pd.crosstab(df_cate[col], df_cate['Sex_F'])
    chi2, p, dof, expected = chi2_contingency(contingency_table_sex)
    results_sex.append({'Feature': col, 'Chi2': chi2, 'p-value': p})

    contingency_table_adhd = pd.crosstab(df_cate[col], df_cate['ADHD_Outcome'])
    chi2, p, dof, expected = chi2_contingency(contingency_table_adhd)
    results_adhd.append({'Feature': col, 'Chi2': chi2, 'p-value': p})

# Táº¡o DataFrame káº¿t quáº£
chi2_sex_df = pd.DataFrame(results_sex).sort_values(by='p-value')
chi2_adhd_df = pd.DataFrame(results_adhd).sort_values(by='p-value')

print("-----Chi2 & P-value both Cate Features vs Sex_F----\n")
print(chi2_sex_df)

print("\n-----Chi2 & P-value both Cate Features vs ADHD----\n")
print(chi2_adhd_df)



from scipy.stats import chi2_contingency
from scipy.stats import chi2_contingency

df_cate = pd.merge(train_quan, train_sol, on="participant_id") 
df_cate = df_cate.drop(columns=["participant_id"])

results_sex = []
results_adhd = []
for col in df_cate.columns:
    if col in ["Sex_F", "ADHD_Outcome"]:
        continue
        
    contingency_table_sex = pd.crosstab(df_cate[col], df_cate['Sex_F'])
    chi2, p, dof, expected = chi2_contingency(contingency_table_sex)
    results_sex.append({'Feature': col, 'Chi2': chi2, 'p-value': p})

    contingency_table_adhd = pd.crosstab(df_cate[col], df_cate['ADHD_Outcome'])
    chi2, p, dof, expected = chi2_contingency(contingency_table_adhd)
    results_adhd.append({'Feature': col, 'Chi2': chi2, 'p-value': p})

# Táº¡o DataFrame káº¿t quáº£
chi2_sex_df = pd.DataFrame(results_sex).sort_values(by='p-value')
chi2_adhd_df = pd.DataFrame(results_adhd).sort_values(by='p-value')

print("-----Chi2 & P-value both Quan Features vs Sex_F----\n")
print(chi2_sex_df)

print("\n-----Chi2 & P-value both Quan Features vs ADHD----\n")
print(chi2_adhd_df)


train_func = pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
train_sol = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")

df = pd.merge(train_func, train_sol, on="participant_id")
df = df.drop(columns=["participant_id", "Sex_F"])

participant_vec = df.drop(columns=[ 'ADHD_Outcome']).iloc[0].values.astype(float)
label = df["ADHD_Outcome"].iloc[0]  # Láº¥y label ADHD_Outcome cho participant Ä‘áº§u tiÃªn

# BÆ°á»›c 2: Chuyá»ƒn vector thÃ nh ma tráº­n Ä‘á»‘i xá»©ng 200x200
def vector_to_symmetric_matrix(vec, N):
    mat = np.zeros((N, N))
    upper_tri_indices = np.triu_indices(N, k=1)
    mat[upper_tri_indices] = vec  # GÃ¡n giÃ¡ trá»‹ tá»« vector vÃ o pháº§n trÃªn Ä‘Æ°á»�ng chÃ©o
    mat += mat.T  # Ä�á»‘i xá»©ng ma tráº­n
    return mat

N = 200  # KÃ­ch thÆ°á»›c cá»§a ma tráº­n (200x200)
conn_matrix = vector_to_symmetric_matrix(participant_vec, N)

# BÆ°á»›c 3: Chuyá»ƒn Ä‘á»•i thÃ nh dataset cho nhiá»�u participants
# Giáº£ sá»­ báº¡n muá»‘n táº¡o dataset cho táº¥t cáº£ cÃ¡c participants trong df
X = []
y = []

for idx in range(len(df)):
    participant_vec = df.drop(columns=['ADHD_Outcome']).iloc[idx].values.astype(float)
    label = df["ADHD_Outcome"].iloc[idx]
    
    # Chuyá»ƒn Ä‘á»•i vector thÃ nh ma tráº­n Ä‘á»‘i xá»©ng
    conn_matrix = vector_to_symmetric_matrix(participant_vec, N)
    
    # ThÃªm ma tráº­n vÃ  label vÃ o dataset
    X.append(conn_matrix)
    y.append(label)

# Chuyá»ƒn Ä‘á»•i danh sÃ¡ch thÃ nh numpy array
X = np.array(X)
y = np.array(y)

# Kiá»ƒm tra kÃ­ch thÆ°á»›c cá»§a dataset
print("Shape of X:", X.shape)  # (sá»‘ participants, 200, 200)
print("Shape of y:", y.shape)  # (sá»‘ participants, )


import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Kiá»ƒm tra náº¿u cÃ³ GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU Ä‘Æ°á»£c nháº­n diá»‡n vÃ  sáº½ Ä‘Æ°á»£c sá»­ dá»¥ng.")
else:
    print("GPU khÃ´ng Ä‘Æ°á»£c nháº­n diá»‡n, Ä‘ang sá»­ dá»¥ng CPU.")

# Chuáº©n bá»‹ dá»¯ liá»‡u
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Chia dá»¯ liá»‡u thÃ nh train, validation, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)  # 60% train, 40% cÃ²n láº¡i cho validation vÃ  test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 50% cho test vÃ  validation (tÆ°Æ¡ng Ä‘Æ°Æ¡ng 20% má»—i)

class LightCNN(tf.keras.Model):
    def __init__(self):
        super(LightCNN, self).__init__()
        
        # CÃ¡c lá»›p Convolutional vÃ  MaxPooling
        self.features = models.Sequential([
            layers.Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=(200, 200, 1)),
            layers.MaxPooling2D(pool_size=2),        # (8, 100, 100)
            layers.Conv2D(16, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=2),        # (16, 50, 50)
        ])
        
        # Lá»›p Adaptive Average Pooling
        self.gap = layers.GlobalAveragePooling2D()  # Chuyá»ƒn thÃ nh GlobalAveragePooling2D
        
        # Lá»›p fully connected (Dense) cuá»‘i cÃ¹ng chá»‰ cÃ³ 1 Ä‘Æ¡n vá»‹ cho phÃ¢n loáº¡i nhá»‹ phÃ¢n
        self.fc = layers.Dense(1, activation='sigmoid')  # Sá»­ dá»¥ng sigmoid cho phÃ¢n loáº¡i nhá»‹ phÃ¢n
        
    def call(self, inputs):
        x = self.features(inputs)
        x = self.gap(x)  # (B, 16)
        x = self.fc(x)   # (B, 1) - PhÃ¢n loáº¡i nhá»‹ phÃ¢n
        return x

model = LightCNN()

# BiÃªn dá»‹ch mÃ´ hÃ¬nh
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# TÃ³m táº¯t mÃ´ hÃ¬nh
model.summary()

# Thiáº¿t láº­p EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Huáº¥n luyá»‡n mÃ´ hÃ¬nh vá»›i EarlyStopping
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])

# Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh trÃªn táº­p kiá»ƒm tra
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Váº½ Ä‘á»“ thá»‹ máº¥t mÃ¡t vÃ  Ä‘á»™ chÃ­nh xÃ¡c trong quÃ¡ trÃ¬nh huáº¥n luyá»‡n
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


encoder = model.features
gap = layers.GlobalAveragePooling2D()

vecs = gap(encoder(X_test))
vecs.shape


from tensorflow.keras import layers, models

# Ä�á»‹nh nghÄ©a mÃ´ hÃ¬nh Autoencoder
autoencoder = models.Sequential([
    layers.Input(shape=(200, 200, 1)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),  # Ä�áº·c trÆ°ng trÃ­ch xuáº¥t á»Ÿ Ä‘Ã¢y
    layers.Dense(32, activation='relu'),
    layers.Dense(200 * 200, activation='sigmoid'),  # TÃ¡i táº¡o láº¡i áº£nh
    layers.Reshape((200, 200, 1))
])

# BiÃªn dá»‹ch vÃ  huáº¥n luyá»‡n autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32)

# TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng tá»« encoder
encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.layers[8].output)
encoded_features = encoder.predict(X_train)
print(encoded_features)


def vector_to_symmetric_matrix(vec, N):
    mat = np.zeros((N, N))
    upper_tri_indices = np.triu_indices(N, k=1)
    mat[upper_tri_indices] = vec  # GÃ¡n giÃ¡ trá»‹ tá»« vector vÃ o pháº§n trÃªn Ä‘Æ°á»�ng chÃ©o
    mat += mat.T  # Ä�á»‘i xá»©ng ma tráº­n
    return mat

def get_features(p_id, func_df, features_df, encoder):
    # Láº¥y vector tá»« func_df
    func_vec = func_df[func_df["participant_id"] == p_id].drop(columns=["participant_id"]).iloc[0].values.astype(float)
    # Chuyá»ƒn vector thÃ nh ma tráº­n Ä‘á»‘i xá»©ng
    func_matrix = vector_to_symmetric_matrix(func_vec, 200)
    # MÃ£ hÃ³a ma tráº­n thÃ nh vector
    emd = encoder(func_matrix)

    # Láº¥y vector tá»« features_df
    features_vec = features_df[features_df["participant_id"] == p_id].drop(columns=["participant_id"]).iloc[0].values.astype(float)
    
    # Ná»‘i 2 vector láº¡i vá»›i nhau
    new_vec = np.concatenate((emd, features_vec))
    
    return new_vec




