!pip install -qU oikan


!pip freeze | grep oikan


import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Set seed ---
np.random.seed(42)


# --- Load Data ---
df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)


# --- Constants ---
TARGET = 'Fertilizer Name'
ID_COL = 'id'
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


import matplotlib.pyplot as plt
%matplotlib inline

# Visualize Class Distribution
target_counts = df_train[TARGET].value_counts(normalize=True)
plt.figure(figsize=(5, 5))
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Target Class Distribution")
plt.axis('equal')
plt.show()


# --- Split Features and Target ---
X_train = df_train.drop([ID_COL, TARGET], axis=1)
y_train = df_train[TARGET]
X_test = df_test.drop(ID_COL, axis=1)
test_ids = df_test[ID_COL]


# --- Feature Engineering Function ---
def apply_feature_engineering(df):
    df = df.copy()
    df['Temp_Humidity_Interaction'] = df['Temparature'] * df['Humidity']
    df['N_P_Ratio'] = df['Nitrogen'] / df['Phosphorous'].replace(0, 1e-6)
    df['K_P_Ratio'] = df['Potassium'] / df['Phosphorous'].replace(0, 1e-6)
    df['Soil_Crop_Combination'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)
    
    for col in numerical_cols:
        df[f'{col}_Binned'] = df[col].astype(str)
    
    return df

# --- Apply Feature Engineering ---
X_train_fe = apply_feature_engineering(X_train)
X_test_fe = apply_feature_engineering(X_test)


# --- Feature Lists ---
numerical_features = numerical_cols + ['Temp_Humidity_Interaction', 'N_P_Ratio', 'K_P_Ratio']
categorical_features = ['Soil Type', 'Crop Type', 'Soil_Crop_Combination'] + [f'{col}_Binned' for col in numerical_cols]
all_features = numerical_features + categorical_features


# --- Reorder Columns ---
X_train_fe = X_train_fe[all_features]
X_test_fe = X_test_fe[all_features]


# --- Label Encode Categorical Features ---
for col in categorical_features:
    le = LabelEncoder()
    all_vals = pd.concat([X_train_fe[col], X_test_fe[col]]).astype(str)
    le.fit(all_vals)
    X_train_fe[col] = le.transform(X_train_fe[col].astype(str))
    X_test_fe[col] = le.transform(X_test_fe[col].astype(str))

print("Label encoding complete.")


# --- Standard Scale Numerical Features ---
scaler = StandardScaler()
X_train_fe[numerical_features] = scaler.fit_transform(X_train_fe[numerical_features])
X_test_fe[numerical_features] = scaler.transform(X_test_fe[numerical_features])

print("Standard scaling complete.")
print("Final train shape:", X_train_fe.shape)
print("Final test shape:", X_test_fe.shape)


# --- Encode Target ---
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
print("Encoded target classes:", {idx: cls for idx, cls in enumerate(label_encoder.classes_)})


# https://github.com/silvermete0r/oikan

from oikan import OIKANClassifier

oikan_model = OIKANClassifier(
    augmentation_factor=1,
    alpha=0.3,
    top_k=10,  
    verbose=True,
    random_state=42
)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_fe, y_train_encoded, test_size=0.2, stratify=y_train_encoded, random_state=42)


%%time

oikan_model.fit(X_train, y_train)


formulas = oikan_model.get_formula()
for formula in formulas:
    print(formula)


print("\n=== OIKAN Feature Importances ===")
feature_names = X_train_fe.columns.tolist()
importances = oikan_model.feature_importances()

top_n = 20
indices = np.argsort(importances)[::-1][:top_n]
top_features = [feature_names[i] for i in indices]
top_importances = importances[indices]

plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importances[::-1], color='skyblue')
plt.xlabel("Importance")
plt.title(f"Top {top_n} OIKAN Feature Importances")
plt.tight_layout()
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.show()


from sklearn.metrics import accuracy_score, classification_report

y_pred = oikan_model.predict(X_test)

print('Accuracy Score:', accuracy_score(y_pred, y_test))
print('Classification Report:\n', classification_report(y_pred, y_test))


%%time

oikan_model.fit(X_train_fe, y_train_encoded)


y_pred = oikan_model.predict(X_test_fe)
y_pred_labels = label_encoder.inverse_transform(y_pred)


submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': y_pred_labels
})

submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

