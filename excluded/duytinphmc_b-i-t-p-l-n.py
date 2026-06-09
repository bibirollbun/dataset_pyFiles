import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency

# Thiáº¿t láº­p style cho biá»ƒu Ä‘á»“
sns.set(font_scale=1.2)

# 1. Load dá»¯ liá»‡u
train_df = pd.read_csv('/kaggle/input/mushroom-classification-btl/train.csv')
test_df = pd.read_csv('/kaggle/input/mushroom-classification-btl/test.csv')

# 2. Kiá»ƒm tra cáº¥u trÃºc dá»¯ liá»‡u
print("=== Cáº¥u trÃºc Train Dataset ===")
print(f"Sá»‘ máº«u: {train_df.shape[0]}")
print(f"Sá»‘ Ä‘áº·c trÆ°ng (bao gá»“m target vÃ  id): {train_df.shape[1]}")
print("\nKiá»ƒu dá»¯ liá»‡u cÃ¡c cá»™t:")
print(train_df.dtypes)
print("\nThÃ´ng tin tá»•ng quan:")
print(train_df.info())

print("\n=== Cáº¥u trÃºc Test Dataset ===")
print(f"Sá»‘ máº«u: {test_df.shape[0]}")
print(f"Sá»‘ Ä‘áº·c trÆ°ng (bao gá»“m target vÃ  id): {test_df.shape[1]}")
print("\nKiá»ƒu dá»¯ liá»‡u cÃ¡c cá»™t:")
print(test_df.dtypes)

# 3. PhÃ¢n tÃ­ch biáº¿n má»¥c tiÃªu (class)
print("\n=== PhÃ¢n phá»‘i biáº¿n má»¥c tiÃªu (class) trong Train ===")
print(train_df['class'].value_counts(normalize=True))
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=train_df)
plt.title('PhÃ¢n phá»‘i Class (Poisonous vs Edible) - Train')
plt.xlabel('Class (e: Edible, p: Poisonous)')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.show()

# 4. PhÃ¢n tÃ­ch Ä‘áº·c trÆ°ng
# ChÃº thÃ­ch: KhÃ´ng cÃ³ Ä‘áº·c trÆ°ng sá»‘ (numerical), chá»‰ cÃ³ categorical
print("\n=== ChÃº thÃ­ch ===")
print("Dataset khÃ´ng cÃ³ Ä‘áº·c trÆ°ng sá»‘, do Ä‘Ã³ khÃ´ng thá»±c hiá»‡n thá»‘ng kÃª mÃ´ táº£, histogram hoáº·c boxplot.")

# PhÃ¢n tÃ­ch Ä‘áº·c trÆ°ng phÃ¢n loáº¡i
categorical_cols = train_df.drop(columns=['class', 'id']).columns
n_cols = 3  # Sá»‘ cá»™t trong lÆ°á»›i subplot
n_rows = int(np.ceil(len(categorical_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, hue='class', data=train_df, ax=axes[i])
    axes[i].set_title(f'PhÃ¢n phá»‘i {col} theo Class')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].legend(title='Class', loc='upper right')
    axes[i].tick_params(axis='x', rotation=45)

# XÃ³a cÃ¡c subplot trá»‘ng
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('categorical_features_distribution.png')
plt.show()

# 5. PhÃ¢n tÃ­ch dá»¯ liá»‡u thiáº¿u (missing)
print("\n=== PhÃ¢n tÃ­ch dá»¯ liá»‡u thiáº¿u (Train) ===")
missing_train = train_df.replace('?', np.nan).isnull().sum()
print(missing_train[missing_train > 0])

print("\n=== PhÃ¢n tÃ­ch dá»¯ liá»‡u thiáº¿u (Test) ===")
missing_test = test_df.replace('?', np.nan).isnull().sum()
print(missing_test[missing_test > 0])

# PhÃ¢n tÃ­ch missing theo class
print("\n=== PhÃ¢n tÃ­ch Missing theo Class (Train) ===")
missing_by_class = train_df[train_df['stalk-root'] == '?'].groupby('class').size()
print(missing_by_class)
plt.figure(figsize=(8, 6))
sns.countplot(x='stalk-root', hue='class', data=train_df[train_df['stalk-root'] == '?'])
plt.title('PhÃ¢n phá»‘i Missing Values (stalk-root) theo Class')
plt.xlabel('stalk-root (Missing = ?)')
plt.ylabel('Count')
plt.savefig('missing_stalk_root_by_class.png')
plt.show()

# 6. PhÃ¢n tÃ­ch má»‘i quan há»‡: Chi-square test thay cho ma tráº­n tÆ°Æ¡ng quan
print("\n=== PhÃ¢n tÃ­ch má»‘i quan há»‡ (Chi-square test) ===")
chi2_results = {}
for col in categorical_cols:
    contingency_table = pd.crosstab(train_df[col], train_df['class'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    chi2_results[col] = p

# Chuyá»ƒn káº¿t quáº£ chi-square thÃ nh DataFrame
chi2_df = pd.DataFrame.from_dict(chi2_results, orient='index', columns=['p-value'])
chi2_df = chi2_df.sort_values(by='p-value')

# In káº¿t quáº£
print("\nP-values tá»« Chi-square test (tháº¥p hÆ¡n = Ä‘áº·c trÆ°ng quan trá»�ng hÆ¡n):")
print(chi2_df)

# Váº½ heatmap cho p-values
plt.figure(figsize=(10, 8))
sns.heatmap(chi2_df, annot=True, cmap='coolwarm', fmt='.2e')
plt.title('P-values cá»§a Chi-square Test giá»¯a Ä�áº·c trÆ°ng vÃ  Class')
plt.savefig('chi2_pvalues_heatmap.png')
plt.show()

# 7. PhÃ¢n tÃ­ch giÃ¡ trá»‹ unique cá»§a cÃ¡c Ä‘áº·c trÆ°ng
print("\n=== Sá»‘ giÃ¡ trá»‹ unique cá»§a tá»«ng Ä‘áº·c trÆ°ng (Train) ===")
unique_counts = train_df.drop(columns=['id']).nunique()
print(unique_counts)


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.stats import chi2_contingency

# Kiá»ƒm tra cá»™t trong test_df
print("Cá»™t trong test_df:", list(test_df.columns))

# 1. Xá»­ lÃ½ missing values
train_df['stalk-root'] = train_df['stalk-root'].replace('?', train_df['stalk-root'].mode()[0])
test_df['stalk-root'] = test_df['stalk-root'].replace('?', test_df['stalk-root'].mode()[0])

# 2. Loáº¡i bá»� cá»™t khÃ´ng cáº§n thiáº¿t
columns_to_drop = ['veil-type', 'veil-color', 'id']
columns_to_drop = [col for col in columns_to_drop if col in train_df.columns]
train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns])

# 3. TÃ¡ch features vÃ  target
X_train = train_df.drop('class', axis=1)
y_train = train_df['class']
X_test = test_df  # test_df khÃ´ng cÃ³ cá»™t 'class'

# ChÃº thÃ­ch: KhÃ´ng cÃ³ Ä‘áº·c trÆ°ng sá»‘ nÃªn bá»� qua chuáº©n hÃ³a/scale
print("ChÃº thÃ­ch: Dataset khÃ´ng cÃ³ Ä‘áº·c trÆ°ng sá»‘, bá»� qua bÆ°á»›c chuáº©n hÃ³a/scale (StandardScaler, MinMaxScaler).")

# 4. Feature selection dá»±a trÃªn chi-square test
chi2_results = {}
for col in X_train.columns:
    contingency_table = pd.crosstab(X_train[col], y_train)
    chi2, p, _, _ = chi2_contingency(contingency_table)
    chi2_results[col] = p

# Chá»�n Ä‘áº·c trÆ°ng cÃ³ p-value < 0.05 (quan trá»�ng)
important_features = [col for col, p in chi2_results.items() if p < 0.05]
print("Ä�áº·c trÆ°ng quan trá»�ng (p-value < 0.05):", important_features)

# Lá»�c X_train vÃ  X_test chá»‰ giá»¯ Ä‘áº·c trÆ°ng quan trá»�ng
X_train_onehot = X_train[important_features]
X_test_onehot = X_test[important_features]

# 5. MÃ£ hÃ³a categorical: One-Hot Encoding
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_onehot_encoded = ohe.fit_transform(X_train_onehot)
X_test_onehot_encoded = ohe.transform(X_test_onehot)

feature_names_onehot = ohe.get_feature_names_out(important_features)
X_train_onehot_encoded = pd.DataFrame(X_train_onehot_encoded, columns=feature_names_onehot)
X_test_onehot_encoded = pd.DataFrame(X_test_onehot_encoded, columns=feature_names_onehot)

# 6. Feature engineering cho One-Hot
# Táº¡o Ä‘áº·c trÆ°ng: odor + spore-print-color
X_train_onehot_encoded['odor_spore_interaction'] = X_train['odor'] + '_' + X_train['spore-print-color']
X_test_onehot_encoded['odor_spore_interaction'] = X_test['odor'] + '_' + X_test['spore-print-color']
# Táº¡o Ä‘áº·c trÆ°ng: gill-color + spore-print-color
X_train_onehot_encoded['gill_spore_interaction'] = X_train['gill-color'] + '_' + X_train['spore-print-color']
X_test_onehot_encoded['gill_spore_interaction'] = X_test['gill-color'] + '_' + X_test['spore-print-color']
# Nhá»‹ phÃ¢n hÃ³a bruises
X_train_onehot_encoded['bruises_binary'] = X_train['bruises'].map({'f': 0, 't': 1})
X_test_onehot_encoded['bruises_binary'] = X_test['bruises'].map({'f': 0, 't': 1})

# MÃ£ hÃ³a cÃ¡c Ä‘áº·c trÆ°ng má»›i
ohe_interaction = OneHotEncoder(sparse=False, handle_unknown='ignore')
interaction_train = ohe_interaction.fit_transform(X_train_onehot_encoded[['odor_spore_interaction', 'gill_spore_interaction']])
interaction_test = ohe_interaction.transform(X_test_onehot_encoded[['odor_spore_interaction', 'gill_spore_interaction']])

interaction_feature_names = ohe_interaction.get_feature_names_out(['odor_spore_interaction', 'gill_spore_interaction'])
X_train_onehot_encoded = pd.concat([X_train_onehot_encoded.drop(['odor_spore_interaction', 'gill_spore_interaction'], axis=1),
                                    pd.DataFrame(interaction_train, columns=interaction_feature_names)], axis=1)
X_test_onehot_encoded = pd.concat([X_test_onehot_encoded.drop(['odor_spore_interaction', 'gill_spore_interaction'], axis=1),
                                   pd.DataFrame(interaction_test, columns=interaction_feature_names)], axis=1)

# 7. MÃ£ hÃ³a categorical: Label Encoding
X_train_label = X_train.copy()
X_test_label = X_test.copy()
label_encoders = {}
for col in X_train_label.columns:
    le = LabelEncoder()
    X_train_label[col] = le.fit_transform(X_train_label[col])
    X_test_label[col] = le.transform(X_test_label[col])
    label_encoders[col] = le

# Feature engineering cho Label Encoding
X_train_label['odor_spore_interaction'] = X_train['odor'] + '_' + X_train['spore-print-color']
X_test_label['odor_spore_interaction'] = X_test['odor'] + '_' + X_test['spore-print-color']
X_train_label['gill_spore_interaction'] = X_train['gill-color'] + '_' + X_train['spore-print-color']
X_test_label['gill_spore_interaction'] = X_test['gill-color'] + '_' + X_test['spore-print-color']
X_train_label['bruises_binary'] = X_train['bruises'].map({'f': 0, 't': 1})
X_test_label['bruises_binary'] = X_test['bruises'].map({'f': 0, 't': 1})

# MÃ£ hÃ³a cÃ¡c Ä‘áº·c trÆ°ng má»›i
le_interaction = LabelEncoder()
X_train_label['odor_spore_interaction'] = le_interaction.fit_transform(X_train_label['odor_spore_interaction'])
X_test_label['odor_spore_interaction'] = le_interaction.transform(X_test_label['odor_spore_interaction'])
le_gill_spore = LabelEncoder()
X_train_label['gill_spore_interaction'] = le_gill_spore.fit_transform(X_train_label['gill_spore_interaction'])
X_test_label['gill_spore_interaction'] = le_gill_spore.transform(X_test_label['gill_spore_interaction'])

# MÃ£ hÃ³a target
y_train = y_train.map({'e': 0, 'p': 1})

# 8. LÆ°u dá»¯ liá»‡u
X_train_onehot_encoded.to_csv('X_train_onehot.csv', index=False)
X_test_onehot_encoded.to_csv('X_test_onehot.csv', index=False)
X_train_label.to_csv('X_train_label.csv', index=False)
X_test_label.to_csv('X_test_label.csv', index=False)
y_train.to_csv('y_train.csv', index=False)

print("=== Káº¿t quáº£ Luá»“ng A ===")
print(f"X_train_onehot shape: {X_train_onehot_encoded.shape}")
print(f"X_test_onehot shape: {X_test_onehot_encoded.shape}")
print(f"X_train_label shape: {X_train_label.shape}")
print(f"X_test_label shape: {X_test_label.shape}")
print(f"One-Hot Feature names: {list(X_train_onehot_encoded.columns)}")
print("Dá»¯ liá»‡u Ä‘Ã£ Ä‘Æ°á»£c lÆ°u vÃ o: X_train_onehot.csv, X_test_onehot.csv, X_train_label.csv, X_test_label.csv, y_train.csv")


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout  # ThÃªm Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Kiá»ƒm tra cá»™t trong test_df
print("Cá»™t trong test_df:", list(test_df.columns))

# 1. Xá»­ lÃ½ missing values
train_df['stalk-root'] = train_df['stalk-root'].replace('?', train_df['stalk-root'].mode()[0])
test_df['stalk-root'] = test_df['stalk-root'].replace('?', test_df['stalk-root'].mode()[0])

# 2. Loáº¡i bá»� cá»™t khÃ´ng cáº§n thiáº¿t
columns_to_drop = ['veil-type', 'veil-color', 'id']
columns_to_drop = [col for col in columns_to_drop if col in train_df.columns]
train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns])

# 3. TÃ¡ch features vÃ  target
X_train = train_df.drop('class', axis=1)
y_train = train_df['class']
X_test = test_df  # test_df khÃ´ng cÃ³ cá»™t 'class'

# 4. MÃ£ hÃ³a categorical features báº±ng One-Hot Encoding
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_encoded = ohe.fit_transform(X_train)
X_test_encoded = ohe.transform(X_test)

# Láº¥y tÃªn cá»™t
feature_names = ohe.get_feature_names_out(X_train.columns)
X_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names)

# MÃ£ hÃ³a target
y_train = y_train.map({'e': 0, 'p': 1})

# 5. XÃ¢y dá»±ng kiáº¿n trÃºc autoencoder
input_dim = X_train_encoded.shape[1]
encoding_dim = 32  # CÃ³ thá»ƒ tÄƒng lÃªn 64 náº¿u muá»‘n giá»¯ thÃªm thÃ´ng tin

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
# encoded = Dropout(0.2)(encoded)  # Uncomment náº¿u muá»‘n giáº£m overfitting
encoded = Dense(encoding_dim, activation='relu')(encoded)  # Bottleneck
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder_model = Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # CÃ³ thá»ƒ giáº£m xuá»‘ng 0.0001 náº¿u loss dao Ä‘á»™ng

# 6. ThÃªm EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # CÃ³ thá»ƒ tÄƒng patience=15 náº¿u cáº§n

# 7. Huáº¥n luyá»‡n autoencoder
history = autoencoder.fit(X_train_encoded, X_train_encoded,
                         epochs=100,  # CÃ³ thá»ƒ tÄƒng lÃªn 150 náº¿u cáº§n thÃªm thá»�i gian há»�c
                         batch_size=32,  # CÃ³ thá»ƒ tÄƒng lÃªn 64 náº¿u cáº§n á»•n Ä‘á»‹nh gradient
                         validation_data=(X_test_encoded, X_test_encoded),
                         callbacks=[early_stopping],
                         verbose=1)

# 8. TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng tá»« bottleneck
X_train_autoencoder = encoder_model.predict(X_train_encoded)
X_test_autoencoder = encoder_model.predict(X_test_encoded)

# Chuyá»ƒn thÃ nh DataFrame
X_train_autoencoder = pd.DataFrame(X_train_autoencoder, columns=[f'feature_{i}' for i in range(encoding_dim)])
X_test_autoencoder = pd.DataFrame(X_test_autoencoder, columns=[f'feature_{i}' for i in range(encoding_dim)])

# 9. LÆ°u dá»¯ liá»‡u
X_train_autoencoder.to_csv('X_train_autoencoder.csv', index=False)
X_test_autoencoder.to_csv('X_test_autoencoder.csv', index=False)
y_train.to_csv('y_train_autoencoder.csv', index=False)

# 10. Váº½ loss curve
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('autoencoder_loss_curve.png')
plt.show()

print("=== Káº¿t quáº£ Luá»“ng B ===")
print(f"X_train_autoencoder shape: {X_train_autoencoder.shape}")
print(f"X_test_autoencoder shape: {X_test_autoencoder.shape}")
print("Dá»¯ liá»‡u Ä‘Ã£ Ä‘Æ°á»£c lÆ°u vÃ o: X_train_autoencoder.csv, X_test_autoencoder.csv, y_train_autoencoder.csv")


# ===================== 3A: Luá»“ng A â€“ One-Hot & Label (giá»¯ nguyÃªn metrics + CM, sá»­a submission) =====================
import os
import json
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== I/O helpers =====================
def smart_read_csv(candidates):
    """Ä�á»�c CSV tá»« danh sÃ¡ch Ä‘Æ°á»�ng dáº«n á»©ng viÃªn (tráº£ vá»� df Ä‘áº§u tiÃªn Ä‘á»�c Ä‘Æ°á»£c)."""
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

def get_test_ids():
    test_df = smart_read_csv([
        "/kaggle/working/test.csv",
        "./test.csv",
        "/mnt/data/test.csv",
        "/kaggle/input/mushroom-classification-btl/test.csv",
    ])
    if test_df is not None:
        if 'id' in test_df.columns:
            return test_df['id']               # giá»¯ nguyÃªn chá»¯ thÆ°á»�ng
        if 'Id' in test_df.columns:
            return test_df['Id'].rename('id')  # Ã©p vá»� chá»¯ thÆ°á»�ng
    return None

# ===================== Chuáº©n hoÃ¡ nhÃ£n cho pháº§n Ä‘Ã¡nh giÃ¡ =====================
def _to_binary(y):
    """Chuáº©n hoÃ¡ nhÃ£n vá»� 0/1. Há»— trá»£ y lÃ  0/1 hoáº·c 'e'/'p'."""
    s = pd.Series(y)
    uniq = set(s.unique())
    if uniq <= {0, 1}:
        return s.astype(int)
    mapping = {'e': 0, 'p': 1, 'E': 0, 'P': 1}
    out = s.map(mapping)
    if out.isna().any():
        raise ValueError(f"NhÃ£n khÃ´ng thuá»™c {{0,1,'e','p'}}: {sorted(list(uniq))[:5]}")
    return out.astype(int)

def _pos_index_for_proba(model):
    """Láº¥y index cá»§a lá»›p dÆ°Æ¡ng trong predict_proba theo model.classes_."""
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 'p' in classes:  # Æ°u tiÃªn lá»›p 'p' náº¿u há»�c trá»±c tiáº¿p e/p
            return classes.index('p')
        if 1 in classes:    # náº¿u há»�c nhá»‹ phÃ¢n 0/1 thÃ¬ láº¥y lá»›p 1
            return classes.index(1)
        try:
            return classes.index(max(classes))
        except Exception:
            return -1
    return 1

# ===================== Load features/labels =====================
X_train_onehot = pd.read_csv('/kaggle/working/X_train_onehot.csv')
X_test_onehot  = pd.read_csv('/kaggle/working/X_test_onehot.csv')
y_train        = pd.read_csv('/kaggle/working/y_train.csv')['class']  # cÃ³ thá»ƒ lÃ  'e'/'p' hoáº·c 0/1

X_train_label  = pd.read_csv('/kaggle/working/X_train_label.csv')
X_test_label   = pd.read_csv('/kaggle/working/X_test_label.csv')

# Stratified split (dÃ¹ng nhÃ£n gá»‘c, khÃ´ng Ã©p sá»›m Ä‘á»ƒ trÃ¡nh lá»‡ch mapping)
Xtr_oh, Xval_oh, ytr_oh, yval_oh = train_test_split(
    X_train_onehot, y_train, test_size=0.2, random_state=42, stratify=y_train
)
Xtr_lb, Xval_lb, ytr_lb, yval_lb = train_test_split(
    X_train_label,  y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ===================== Models =====================
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
}

# ===================== Train & Evaluate =====================
def evaluate_models(X_train, X_val, y_train, y_val, models, prefix):
    results = {}

    # chuáº©n hoÃ¡ nhÃ£n y_val vá»� 0/1 Ä‘á»ƒ tÃ­nh metric nháº¥t quÃ¡n (0=e, 1=p)
    y_true_bin = _to_binary(y_val)

    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())

        # Dá»± Ä‘oÃ¡n nhÃ£n vÃ  xÃ¡c suáº¥t (náº¿u cÃ³)
        y_pred = model.predict(X_val)
        y_pred_bin = _to_binary(y_pred)

        if hasattr(model, "predict_proba"):
            pos_idx = _pos_index_for_proba(model)
            y_proba = model.predict_proba(X_val)[:, pos_idx]
        else:
            y_proba = None

        # metrics trÃªn nhá»‹ phÃ¢n (pos = 1 = 'p')
        res = {
            'Accuracy':  accuracy_score(y_true_bin, y_pred_bin),
            'Precision': precision_score(y_true_bin, y_pred_bin, zero_division=0),
            'Recall':    recall_score(y_true_bin, y_pred_bin, zero_division=0),
            'F1':        f1_score(y_true_bin, y_pred_bin, zero_division=0),
            'ROC-AUC':   roc_auc_score(y_true_bin, y_proba) if y_proba is not None else 0.0
        }
        results[name] = res

        # Log
        print(f"\n[{prefix}] {name}")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")

        # Confusion matrix hiá»ƒn thá»‹ e/p
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['e', 'p'],
                    yticklabels=['e', 'p'])
        plt.title(f'Confusion Matrix - {prefix} - {name}')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        fn = f'confusion_matrix_{prefix.lower().replace(" ", "_")}_{name.replace(" ", "_")}.png'
        plt.savefig(fn, dpi=150)
        plt.show()

    # LÆ°u JSON káº¿t quáº£ (Ä‘Ãºng pattern BÆ°á»›c 4 Ä‘ang Ä‘á»�c)
    out_json = f'stream_a_{prefix.lower().replace(" ", "_")}_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f)
    print(f"\nÄ�Ã£ lÆ°u káº¿t quáº£: {out_json}")
    return results

print("=== Huáº¥n luyá»‡n & Ä�Ã¡nh giÃ¡: Luá»“ng A - One-Hot Encoding ===")
res_oh = evaluate_models(Xtr_oh, Xval_oh, ytr_oh, yval_oh, models, 'One-Hot Encoding')

print("\n=== Huáº¥n luyá»‡n & Ä�Ã¡nh giÃ¡: Luá»“ng A - Label Encoding ===")
res_lb = evaluate_models(Xtr_lb, Xval_lb, ytr_lb, yval_lb, models, 'Label Encoding')

# ===================== So sÃ¡nh nhanh vá»›i Random Forest =====================
comparison_df = pd.DataFrame({
    'Luá»“ng':    ['One-Hot Encoding', 'Label Encoding'],
    'Accuracy': [res_oh['Random Forest']['Accuracy'], res_lb['Random Forest']['Accuracy']],
    'F1':       [res_oh['Random Forest']['F1'],       res_lb['Random Forest']['F1']]
}).round(4)
print("\n=== Báº£ng So sÃ¡nh Hiá»‡u suáº¥t (Random Forest) cho Luá»“ng A ===")
print(comparison_df)
comparison_df.to_csv('model_comparison_stream_a.csv', index=False)

# ===================== Submission (NHÃƒN e/p theo Ä‘á»� má»›i) =====================
print("\n=== Táº¡o file submission cho Kaggle (id, class = 'e'/'p') ===")

# ğŸ”’ LUÃ”N Ä‘á»�c test.csv Ä‘á»ƒ láº¥y Ä‘Ãºng bá»™ vÃ  thá»© tá»± id (trÃ¡nh lá»‡ch sá»‘ dÃ²ng)
test_df = smart_read_csv([
    "/kaggle/input/mushroom-classification-btl/test.csv",  # competition input
    "/kaggle/working/test.csv",
    "./test.csv",
    "/mnt/data/test.csv",
])
if test_df is None:
    raise FileNotFoundError("â�Œ KhÃ´ng tÃ¬m tháº¥y/Ä‘á»�c Ä‘Æ°á»£c test.csv gá»‘c.")

if 'id' in test_df.columns:
    test_ids = test_df['id']
elif 'Id' in test_df.columns:
    test_ids = test_df['Id'].rename('id')
else:
    raise ValueError("â�Œ test.csv khÃ´ng cÃ³ cá»™t id/Id.")

print(f"âœ… Láº¥y {len(test_ids)} hÃ ng id tá»« test.csv")

# Train mÃ´ hÃ¬nh cuá»‘i Ä‘á»ƒ ná»™p: Random Forest trÃªn One-Hot
rf_onehot = RandomForestClassifier(random_state=42)
rf_onehot.fit(X_train_onehot, y_train.values.ravel())

# Dá»± Ä‘oÃ¡n NHÃƒN cho test (cÃ³ thá»ƒ lÃ  0/1 hoáº·c 'e'/'p')
pred_any = rf_onehot.predict(X_test_onehot)

# Chuáº©n hoÃ¡ nhÃ£n ná»™p vá»� 'e'/'p'
if set(pd.Series(pred_any).unique()) <= {0, 1}:
    pred_cls = np.where(pred_any == 1, 'p', 'e')
else:
    pred_cls = pd.Series(pred_any).astype(str).str.lower()
    ok = pred_cls.isin(['e','p']).all()
    if not ok:
        raise ValueError("â�Œ Dá»± Ä‘oÃ¡n khÃ´ng thuá»™c {'e','p'} hoáº·c {0,1}. Kiá»ƒm tra láº¡i pipeline/labels.")

# Táº¡o submission.csv Ä‘Ãºng format (sá»‘ dÃ²ng = sá»‘ dÃ²ng test)
submission = pd.DataFrame({'id': test_ids.values, 'class': pred_cls})
assert len(submission) == len(test_df), f"Sá»‘ dÃ²ng lá»‡ch: submission={len(submission)} vs test={len(test_df)}"

# Ghi file (khÃ´ng cÃ³ dáº¥u nhÃ¡y kÃ©p)
submission.to_csv('submission.csv', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
print("âœ… Ä�Ãƒ Táº O FILE Ná»˜P: submission.csv  (cá»™t id, class = 'e'/'p')")
print(submission.head())



# ====== 3B: Luá»“ng B â€“ Autoencoder (Classification, output e/p) ======
import os, json, csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Helpers ----------
def smart_read_csv(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

def _to_binary(y):
    """Chuáº©n hoÃ¡ nhÃ£n vá»� 0/1. Há»— trá»£ y lÃ  0/1 hoáº·c 'e'/'p'."""
    s = pd.Series(y)
    uniq = set(s.unique())
    if uniq <= {0, 1}:
        return s.astype(int)
    mapping = {'e': 0, 'p': 1, 'E': 0, 'P': 1}
    out = s.map(mapping)
    if out.isna().any():
        raise ValueError(f"NhÃ£n khÃ´ng thuá»™c {{0,1,'e','p'}}: {sorted(list(uniq))[:5]}")
    return out.astype(int)

def _pos_index_for_proba(model):
    """Láº¥y index cá»§a lá»›p dÆ°Æ¡ng trong predict_proba theo model.classes_."""
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 'p' in classes:
            return classes.index('p')
        if 1 in classes:
            return classes.index(1)
        try:
            return classes.index(max(classes))
        except Exception:
            return -1
    return 1

# ---------- Load ----------
X_train_auto = pd.read_csv("/kaggle/working/X_train_autoencoder.csv")
X_test_auto  = pd.read_csv("/kaggle/working/X_test_autoencoder.csv")
y_train_auto = pd.read_csv("/kaggle/working/y_train_autoencoder.csv")["class"]  # 'e'/'p' hoáº·c 0/1

Xtr_a, Xval_a, ytr_a, yval_a = train_test_split(
    X_train_auto, y_train_auto, test_size=0.2, random_state=42, stratify=y_train_auto
)

# ---------- Models ----------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVM": SVC(random_state=42, probability=True),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss", use_label_encoder=False),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
}

# ---------- Train & Evaluate ----------
def evaluate_models(X_train, X_val, y_train, y_val, models, prefix):
    results = {}
    y_true_bin = _to_binary(y_val)  # 0=e, 1=p

    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())

        y_pred = model.predict(X_val)
        y_pred_bin = _to_binary(y_pred)

        if hasattr(model, "predict_proba"):
            pos_idx = _pos_index_for_proba(model)
            y_proba = model.predict_proba(X_val)[:, pos_idx]
        else:
            y_proba = None

        res = {
            "Accuracy":  accuracy_score(y_true_bin, y_pred_bin),
            "Precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
            "Recall":    recall_score(y_true_bin, y_pred_bin, zero_division=0),
            "F1":        f1_score(y_true_bin, y_pred_bin, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_true_bin, y_proba) if y_proba is not None else 0.0,
        }
        results[name] = res

        print(f"\n[{prefix}] {name}")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")

        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["e","p"], yticklabels=["e","p"])
        plt.title(f"Confusion Matrix - {prefix} - {name}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        fn = f'confusion_matrix_{prefix.lower().replace(" ", "_")}_{name.replace(" ", "_")}.png'
        plt.savefig(fn, dpi=150)
        plt.show()

    out_json = f'stream_b_{prefix.lower().replace(" ", "_")}_results.json'
    with open(out_json, "w") as f:
        json.dump(results, f)
    print(f"\nÄ�Ã£ lÆ°u káº¿t quáº£: {out_json}")
    return results

print("=== Huáº¥n luyá»‡n & Ä�Ã¡nh giÃ¡ cho Luá»“ng B - Autoencoder ===")
results_auto = evaluate_models(Xtr_a, Xval_a, ytr_a, yval_a, models, "Autoencoder")

# ---------- Submission (id,class = e/p) ----------
print("\n=== Dá»± Ä‘oÃ¡n Test Set (Random Forest) & Táº¡o submission cho Kaggle ===")

# Ä�á»�c test.csv gá»‘c Ä‘á»ƒ láº¥y id + sá»‘ dÃ²ng chuáº©n
test_df = smart_read_csv([
    "/kaggle/input/mushroom-classification-btl/test.csv",
    "/kaggle/working/test.csv",
    "./test.csv",
    "/mnt/data/test.csv",
])
if test_df is None:
    raise FileNotFoundError("â�Œ KhÃ´ng tÃ¬m tháº¥y file test.csv gá»‘c.")

if "id" in test_df.columns:
    test_ids = test_df["id"]
elif "Id" in test_df.columns:
    test_ids = test_df["Id"].rename("id")
else:
    raise ValueError("â�Œ test.csv khÃ´ng cÃ³ cá»™t id/Id.")

print(f"âœ… Láº¥y {len(test_ids)} hÃ ng id tá»« test.csv")

# Train láº¡i Random Forest trÃªn toÃ n bá»™ dá»¯ liá»‡u Autoencoder
best_model_auto = RandomForestClassifier(random_state=42)
best_model_auto.fit(X_train_auto, y_train_auto.values.ravel())
pred_any = best_model_auto.predict(X_test_auto)

# Chuáº©n hoÃ¡ nhÃ£n -> e/p
if set(pd.Series(pred_any).unique()) <= {0,1}:
    pred_cls = np.where(pred_any==1, "p", "e")
else:
    pred_cls = pd.Series(pred_any).astype(str).str.lower()
    if not pred_cls.isin(["e","p"]).all():
        raise ValueError("â�Œ Dá»± Ä‘oÃ¡n khÃ´ng há»£p lá»‡, pháº£i thuá»™c {'e','p'} hoáº·c {0,1}.")

# Táº¡o submission.csv Ä‘Ãºng format
submission = pd.DataFrame({"id": test_ids.values, "class": pred_cls})
assert len(submission) == len(test_df), f"Sá»‘ dÃ²ng lá»‡ch: submission={len(submission)} vs test={len(test_df)}"

submission.to_csv("submission_auto.csv", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
print("âœ… Ä�Ãƒ Táº O FILE Ná»˜P: submission_auto.csv (cá»™t id,class = e/p)")
print(submission.head())



import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- Cáº¥u hÃ¬nh tÃ¬m file ---
SEARCH_DIRS = ['.', '/kaggle/working', '/mnt/data']

# Map prefix há»£p lá»‡ -> cÃ¡c tÃªn file cÃ³ thá»ƒ (Ä‘áº£m báº£o tÆ°Æ¡ng thÃ­ch vá»›i BÆ°á»›c 3)
CANDIDATE_JSON_NAMES = {
    'stream_a_one-hot_encoding': ['stream_a_one-hot_encoding_results.json'],
    'stream_a_label_encoding':   ['stream_a_label_encoding_results.json'],
    'stream_b_autoencoder':      ['stream_b_autoencoder_results.json'],
}

def _find_existing_file(candidates):
    """
    TÃ¬m file trong SEARCH_DIRS theo danh sÃ¡ch tÃªn 'candidates'.
    Tráº£ vá»� path náº¿u tÃ¬m tháº¥y, ngÆ°á»£c láº¡i None.
    """
    for d in SEARCH_DIRS:
        for name in candidates:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path
    return None

def load_results(file_or_prefix):
    """
    - Náº¿u truyá»�n Ä‘Æ°á»�ng dáº«n .json: dÃ¹ng trá»±c tiáº¿p.
    - Náº¿u truyá»�n prefix (vÃ­ dá»¥: 'stream_a_one-hot_encoding'):
        + Thá»­ cÃ¡c tÃªn file á»©ng vá»›i prefix trong CANDIDATE_JSON_NAMES
        + TÃ¬m trong SEARCH_DIRS
    Tráº£ vá»� dict {model: {Accuracy, Precision, Recall, F1, ROC-AUC}} hoáº·c None.
    """
    # TH1: truyá»�n trá»±c tiáº¿p Ä‘Æ°á»�ng dáº«n .json
    if isinstance(file_or_prefix, str) and file_or_prefix.endswith('.json'):
        file_path = file_or_prefix if os.path.exists(file_or_prefix) else None
        if file_path is None:
            print(f"File {file_or_prefix} khÃ´ng tá»“n táº¡i.")
            return None
    else:
        # TH2: truyá»�n prefix
        prefix = file_or_prefix
        # Náº¿u khÃ´ng cÃ³ trong map, váº«n thá»­ ghÃ©p _results.json
        candidates = CANDIDATE_JSON_NAMES.get(prefix, [f"{prefix}_results.json"])
        file_path = _find_existing_file(candidates)
        if file_path is None:
            tried = [os.path.join(d, n) for d in SEARCH_DIRS for n in candidates]
            print("KhÃ´ng tÃ¬m tháº¥y file káº¿t quáº£ cho prefix:", prefix)
            print("Ä�Ã£ thá»­ cÃ¡c Ä‘Æ°á»�ng dáº«n sau:")
            for t in tried:
                print("  -", t)
            return None

    # Ä�á»�c JSON
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Lá»—i giáº£i mÃ£ JSON trong file {file_path}. HÃ£y kiá»ƒm tra hoáº·c cháº¡y láº¡i huáº¥n luyá»‡n.")
        return None
    except Exception as e:
        print(f"Lá»—i khi má»Ÿ {file_path}: {e}")
        return None

    # Chuáº©n hÃ³a cáº¥u trÃºc
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        return data
    elif isinstance(data, dict):
        results = {}
        for model, metrics in data.items():
            if isinstance(metrics, dict):
                results[model] = metrics
            else:
                results[model] = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0, 'ROC-AUC': 0}
        return results
    else:
        print(f"Cáº¥u trÃºc file {file_path} khÃ´ng há»£p lá»‡.")
        return None


# ====== Táº¢I Káº¾T QUáº¢ (dÃ¹ng Ä‘Ãºng prefix) ======
results_stream_a_onehot = load_results('stream_a_one-hot_encoding')
results_stream_a_label  = load_results('stream_a_label_encoding')
results_stream_b_auto   = load_results('stream_b_autoencoder')

# Kiá»ƒm tra thiáº¿u
missing = []
if results_stream_a_onehot is None:
    missing.append('stream_a_one-hot_encoding_results.json')
if results_stream_a_label is None:
    missing.append('stream_a_label_encoding_results.json')
if results_stream_b_auto is None:
    missing.append('stream_b_autoencoder_results.json')

if missing:
    print("\nLá»—i: Thiáº¿u file JSON káº¿t quáº£ sau:")
    for m in missing:
        print("  -", m)
    print("\nGá»£i Ã½ kháº¯c phá»¥c:")
    print("- Ä�áº£m báº£o Ä‘Ã£ CHáº Y xong BÆ°á»›c 3 vÃ  cÃ¡c file trÃªn Ä‘Ã£ Ä‘Æ°á»£c táº¡o.")
    print("- Náº¿u báº¡n khÃ´ng cháº¡y trÃªn Kaggle, hÃ£y kiá»ƒm tra thÆ° má»¥c hiá»‡n táº¡i hoáº·c /mnt/data.")
    sys.exit(1)

# ====== Táº O Báº¢NG SO SÃ�NH ======
def create_comparison_df(results, prefix):
    rows = []
    for model, metrics in results.items():
        rows.append({
            'Luá»“ng': prefix,
            'MÃ´ hÃ¬nh': model,
            'Accuracy': metrics.get('Accuracy', 0),
            'Precision': metrics.get('Precision', 0),
            'Recall': metrics.get('Recall', 0),
            'F1': metrics.get('F1', 0),
            'ROC-AUC': metrics.get('ROC-AUC', 0)
        })
    return pd.DataFrame(rows)

df_stream_a_onehot = create_comparison_df(results_stream_a_onehot, 'Luá»“ng A - One-Hot Encoding')
df_stream_a_label  = create_comparison_df(results_stream_a_label, 'Luá»“ng A - Label Encoding')
df_stream_b_auto   = create_comparison_df(results_stream_b_auto,  'Luá»“ng B - Autoencoder')

combined_df = pd.concat([df_stream_a_onehot, df_stream_a_label, df_stream_b_auto], ignore_index=True)

print("\n=== Báº£ng So sÃ¡nh Hiá»‡u suáº¥t Chi tiáº¿t giá»¯a Luá»“ng A vÃ  Luá»“ng B ===")
pd.set_option('display.max_columns', None)
print(combined_df.round(4))
pd.reset_option('display.max_columns')

# LÆ°u
out_csv = 'combined_model_comparison_detailed.csv'
combined_df.to_csv(out_csv, index=False)
print(f"\nÄ�Ã£ lÆ°u báº£ng tá»•ng há»£p: {out_csv}")

# ====== Váº¼ BIá»‚U Ä�á»’ F1 ======
plt.figure(figsize=(12, 6))
sns.barplot(x='MÃ´ hÃ¬nh', y='F1', hue='Luá»“ng', data=combined_df, palette='viridis')
plt.title('So sÃ¡nh F1 Score giá»¯a Luá»“ng A vÃ  Luá»“ng B theo MÃ´ hÃ¬nh')
plt.xlabel('MÃ´ hÃ¬nh')
plt.ylabel('F1 Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Luá»“ng')
plt.tight_layout()
plt.savefig('f1_comparison_plot.png', dpi=150)
plt.show()

# ====== PHÃ‚N TÃ�CH ======
print("\n=== PhÃ¢n tÃ­ch Káº¿t quáº£ ===")
print("1) Náº¿u cÃ¡c chá»‰ sá»‘ Ä‘á»�u ~1.0, nhiá»�u kháº£ nÄƒng dá»¯ liá»‡u xÃ¡c thá»±c quÃ¡ dá»… hoáº·c cÃ³ rÃ² rá»‰ dá»¯ liá»‡u (data leakage).")
print("2) Kiá»ƒm tra cÃ¡c áº£nh confusion_matrix_*.png Ä‘á»ƒ xÃ¡c thá»±c phÃ¢n phá»‘i dá»± Ä‘oÃ¡n.")
print("3) Náº¿u nghi ngá»� overfitting hoáº·c leakage, thá»­ Ä‘á»•i random_state/cÃ¡ch chia data hoáº·c dÃ¹ng cross-validation.")
print("4) Chá»�n mÃ´ hÃ¬nh/luá»“ng cÃ³ F1 cao nháº¥t cho suy diá»…n test (test_predictions_*.csv).")


