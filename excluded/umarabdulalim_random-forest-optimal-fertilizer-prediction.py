import pandas as pd


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df.head()


df.info()


print("Soil Type:")
print(df['Soil Type'].unique())
print("\nCrop Type:")
print(df['Crop Type'].unique())
print("\nFertilizer Name:")
print(df['Fertilizer Name'].unique())


df_encoded = pd.get_dummies(df, columns=['Soil Type', 'Crop Type'], drop_first=True)
df_encoded.head()


df_encoded.columns


X = df_encoded.drop(['id','Fertilizer Name'], axis=1).astype(int)
y = df['Fertilizer Name']
print("\nShape data fitur (X):", X.shape)
print("Shape data target (y):", y.shape)
X.head()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size (feature): {X_train.shape}")
print(f"Testing data size (feature): {X_test.shape}")
print(f"Training data size (target): {y_train.shape}")
print(f"Testing data size (target): {y_test.shape}")


model = RandomForestClassifier(
    n_estimators=200,          # Jumlah pohon
    max_features='sqrt',       # Pilihan umum untuk klasifikasi
    max_depth=None,            # Biarkan pohon tumbuh penuh
    min_samples_split=5,       # Sedikit lebih ketat dari default
    min_samples_leaf=3,        # Sedikit lebih ketat dari default
    random_state=42,           # Untuk hasil yang reproducible
    n_jobs=-1                  # Menggunakan semua core CPU yang tersedia
) 
model.fit(X_train, y_train)

print("\nRandom Forest Model successfully trained!")


y_pred = model.predict(X_test) # Make predictions on testing data

print("\n--- Model Evaluation Results ---")

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}") # Tampilkan 4 angka di belakang koma

# Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


prediction_probabilities = model.predict_proba(X_test)
class_labels = model.classes_
print("Urutan kelas dalam probabilitas prediksi:", class_labels)


import numpy as np
from sklearn.preprocessing import LabelBinarizer

def mean_average_precision_at_k(y_true, y_pred_proba, k=3):

    if y_true.ndim == 1:
        # Ubah y_true ke format one-hot encoding agar sesuai dengan y_pred_proba
        lb = LabelBinarizer()
        y_true_binarized = lb.fit_transform(y_true)
    else:
        y_true_binarized = y_true # Jika y_true sudah one-hot

    n_samples, n_classes = y_pred_proba.shape
    ap_scores = []

    for i in range(n_samples):
        # Dapatkan indeks kelas sebenarnya (index dari 1 di one-hot)
        true_class_idx = np.where(y_true_binarized[i] == 1)[0][0]

        # Dapatkan indeks kelas yang diprediksi berdasarkan probabilitas, diurutkan dari tertinggi ke terendah
        # np.argsort mengembalikan indeks yang akan mengurutkan array
        # [::-1] membalik urutan (dari terbesar ke terkecil)
        predicted_class_indices = np.argsort(y_pred_proba[i])[::-1]

        # Pertimbangkan hanya top-k prediksi
        top_k_predictions = predicted_class_indices[:k]

        # Hitung presisi pada top-k
        relevant_count = 0
        precision_at_j = []
        for j, pred_idx in enumerate(top_k_predictions):
            if pred_idx == true_class_idx:
                relevant_count += 1
            # Presisi pada titik j: (jumlah relevan sampai j) / (jumlah total item sampai j+1)
            precision_at_j.append(relevant_count / (j + 1))

        # Average Precision (AP) untuk sampel ini
        # Hanya menjumlahkan presisi pada titik-titik di mana ada item relevan
        ap = 0.0
        relevant_indices_in_top_k = [j for j, pred_idx in enumerate(top_k_predictions) if pred_idx == true_class_idx]
        if relevant_indices_in_top_k:
            for rel_idx in relevant_indices_in_top_k:
                ap += precision_at_j[rel_idx]
            ap /= relevant_count # Bagi dengan jumlah item relevan yang ditemukan

        ap_scores.append(ap)

    # Mean Average Precision (mAP) adalah rata-rata dari semua AP sampel
    return np.mean(ap_scores)

y_pred_proba = model.predict_proba(X_test)

map_at_3_score = mean_average_precision_at_k(y_test, y_pred_proba, k=3)
print(f"mAP@3: {map_at_3_score:.4f}")



df2 = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df2.head()


df2.info()


print("Soil Type:")
print(df2['Soil Type'].unique())
print("\nCrop Type:")
print(df2['Crop Type'].unique())


df2_encoded = pd.get_dummies(df2, columns=['Soil Type', 'Crop Type'], drop_first=True)
df2_encoded.head()


idData = df2['id']
A = df2_encoded.drop('id', axis=1).astype(int)
A.head()


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
class_names = model.classes_
prediction_probabilities = model.predict_proba(A)
top3_fertilizers_list = []

for probs in prediction_probabilities:
    top_k_indices = np.argsort(probs)[::-1][:3]
    top_k_fertilizers_names = [class_names[idx] for idx in top_k_indices]
    combined_fertilizers_string = " ".join(top_k_fertilizers_names)
    top3_fertilizers_list.append(combined_fertilizers_string)

results = pd.DataFrame({
    'id': idData,
    'Fertilizer Name': top3_fertilizers_list # Gunakan list string top-3
})
results.head()


output_filename = '/kaggle/working/optimal_fertilizer_random_forest.csv'
results.to_csv(output_filename, index=False)

