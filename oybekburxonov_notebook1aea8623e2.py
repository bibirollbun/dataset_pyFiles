import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np


print("\n[2] Ma'lumotlarni yuklamoqdamiz...")
train_df = pd.read_csv("../input/binaryclassificationwithabankchurndataset/train.csv")
test_df = pd.read_csv("../input/binaryclassificationwithabankchurndataset/test.csv")
print(f" Train fayli: {train_df.shape[0]} ta qatordan iborat.")
print(f" Test fayli: {test_df.shape[0]} ta qatordan iborat.")


df = train_df.copy()
print("\n[3] Keraksiz ustunlarni olib tashlayapmiz...")
df.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)
print(" Ustunlar olib tashlandi: 'id', 'CustomerId', 'Surname'")


print("\n[4] Kategorik ustunlarni raqamlashtirmoqdamiz (Label Encoding)...")
label_encoders = {}
categorical_cols = ['Geography', 'Gender']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f" {col} kodlandi: {le.classes_}")
print(" Barcha kategorik ustunlar kodlandi.")


print("\n[5] Maqsadli ustun (Exited)ni ajratmoqdamiz...")
X = df.drop('Exited', axis=1)
y = df['Exited']
print(f"X shakli: {X.shape}, y shakli: {y.shape}")




print("\n[6] Ma'lumotlarni trening va validatsiyaga bo'lyapmiz...")
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" Train: {X_train.shape[0]} ta, Validatsiya: {X_valid.shape[0]} ta")



print("\n[7] Xususiyatlarni miqyoslamoqdamiz (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
print(" Miqyoslash bajarildi.")



print("\n[8] Random Forest modelini qurmoqdamiz...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print(" Model o'qitildi.")


print("\n[9] Modelni ROC AUC bilan baholayapmiz...")
valid_preds = model.predict_proba(X_valid_scaled)[:, 1]
roc_score = roc_auc_score(y_valid, valid_preds)
print(f" ROC AUC Score (Validatsiya): {roc_score:.4f}")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


valid_classes = model.predict(X_valid_scaled)


accuracy = accuracy_score(y_valid, valid_classes)
print(f" Model aniqligi (Accuracy): {accuracy * 100:.2f}%")


print("\n Classification Report:")
print(classification_report(y_valid, valid_classes))

print("\n Confusion Matrix:")
print(confusion_matrix(y_valid, valid_classes))


print("\n[10] Test ma'lumotlariga ishlov berilmoqda...")
test_ids = test_df['id']
test_df = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)
for col in ['Geography', 'Gender']:
    test_df[col] = label_encoders[col].transform(test_df[col])

print(" Kategorik ustunlar testda ham kodlandi.")


test_scaled = scaler.transform(test_df)
print(" Test ma'lumotlari miqyoslandi.")


print("\n[11] Test uchun ehtimollarni bashorat qilmoqdamiz...")
test_preds = model.predict_proba(test_scaled)[:, 1]

print("\n[12] Submission faylini tayyorlamoqdamiz...")
submission = pd.DataFrame({
    'id': test_ids,
    'Exited': test_preds
})
submission.to_csv("submission.csv", index=False)
print(" submission.csv fayli saqlandi!")



print("\n Jarayon to'liq yakunlandi!")
print("Topshirish fayli quyidagi ko'rinishda:")
print(submission.head())

