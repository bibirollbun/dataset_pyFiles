import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss


# 1. Ma'lumotlarni yuklash
train_df = pd.read_csv("/kaggle/input/multiclassificationtask/train.csv")
test_df = pd.read_csv("/kaggle/input/multiclassificationtask/test.csv")


# 2. Kategorik va sonli ustunlarni ajratish
categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
numerical_cols = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']

target_col = 'Status'
status_mapping = {'C': 0, 'CL': 1, 'D': 2}
train_df[target_col] = train_df[target_col].map(status_mapping)

# NaN qiymatlarni olib tashlash
train_df = train_df.dropna(subset=[target_col])


# 3. Ma’lumotlarni tozalash va yetishmayotgan qiymatlarni to‘ldirish
imputer_cat = SimpleImputer(strategy='most_frequent')
imputer_num = SimpleImputer(strategy='median')
train_df[categorical_cols] = imputer_cat.fit_transform(train_df[categorical_cols])
test_df[categorical_cols] = imputer_cat.transform(test_df[categorical_cols])
train_df[numerical_cols] = imputer_num.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = imputer_num.transform(test_df[numerical_cols])


# 4. Kategorik ustunlarni kodlash
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_encoded = pd.DataFrame(encoder.fit_transform(train_df[categorical_cols]))
test_encoded = pd.DataFrame(encoder.transform(test_df[categorical_cols]))
train_encoded.columns = encoder.get_feature_names_out()
test_encoded.columns = encoder.get_feature_names_out()


# 5. Yangi xususiyatlar yaratish (Feature Engineering)
train_df['Bilirubin_Albumin_Ratio'] = train_df['Bilirubin'] / train_df['Albumin']
test_df['Bilirubin_Albumin_Ratio'] = test_df['Bilirubin'] / test_df['Albumin']
numerical_cols.append('Bilirubin_Albumin_Ratio')


# 6. Normalizatsiya qilish
scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_df[numerical_cols]), columns=numerical_cols)
test_scaled = pd.DataFrame(scaler.transform(test_df[numerical_cols]), columns=numerical_cols)


# 7. Ma'lumotlarni birlashtirish
X_train = pd.concat([train_scaled, train_encoded], axis=1)
X_test = pd.concat([test_scaled, test_encoded], axis=1)
y_train = train_df[target_col]


# 8. Modelni tayyorlash
X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)


print(f"NaN in y_train: {y_train.isna().sum()}")
print(f"NaN in y_valid: {y_valid.isna().sum()}")


# 9. Modelni baholash
y_pred_proba = model.predict_proba(X_valid)
logloss = log_loss(y_valid, y_pred_proba)
print(f'Validation Log Loss: {logloss}')


# 10. Test ma'lumotlari bo'yicha bashorat qilish
# Test ma'lumotlari bo'yicha bashorat qilish
test_predictions = model.predict_proba(X_test)

# ID ustunini qo'shish
if 'id' in test_df.columns:
    submission = pd.DataFrame({'id': test_df['id']})
else:
    submission = pd.DataFrame({'id': range(15000, 15000 + len(test_df))})

# Bashorat natijalarini qo'shish
submission[['Status_C', 'Status_CL', 'Status_D']] = test_predictions

# Faylni saqlash
submission.to_csv('serrozsubmission.csv', index=False)
print(submission)
print("Natijalar saqlandi: serrozsubmission.csv")

