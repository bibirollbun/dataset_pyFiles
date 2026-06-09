import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from scipy.stats import uniform, randint


# --- Path ke data di Kaggle Notebook ---
data_path = '/kaggle/input/airbnb-recruiting-new-user-bookings/'

# --- Fungsi untuk Mengekstrak dan Memuat CSV dari ZIP ---
def load_and_unzip_csv(zip_file_name, extract_to_path='./'):
    zip_file_full_path = os.path.join(data_path, zip_file_name)
    csv_file_name_in_zip = zip_file_name.replace('.zip', '') 
    csv_file_extracted_path = os.path.join(extract_to_path, csv_file_name_in_zip)

    print(f"Mengekstrak '{zip_file_full_path}'...")
    try:
        with zipfile.ZipFile(zip_file_full_path, 'r') as zip_ref:
            if csv_file_name_in_zip in zip_ref.namelist():
                zip_ref.extract(csv_file_name_in_zip, extract_to_path)
            else:
                found_csv = False
                for name_in_zip in zip_ref.namelist():
                    if name_in_zip.endswith('.csv') and csv_file_name_in_zip.split('.')[0] in name_in_zip:
                        zip_ref.extract(name_in_zip, extract_to_path)
                        csv_file_extracted_path = os.path.join(extract_to_path, name_in_zip)
                        found_csv = True
                        break
                if not found_csv:
                    raise FileNotFoundError(f"CSV file corresponding to '{zip_file_name}' not found inside the zip.")

        print(f"'{csv_file_extracted_path}' berhasil diekstrak dan siap dimuat.")
        df = pd.read_csv(csv_file_extracted_path)
        print(f"Dataset '{os.path.basename(csv_file_extracted_path)}' berhasil dimuat.")
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}. Pastikan path dan nama file sudah benar.")
        return None
    except Exception as e:
        print(f"Terjadi kesalahan saat mengekstrak atau memuat file '{zip_file_name}': {e}")
        return None


# --- Memuat Semua Dataset yang Disediakan ---
zip_files_to_load = [
    'train_users_2.csv.zip',
    'test_users.csv.zip',
    'sessions.csv.zip',
    'age_gender_bkts.csv.zip',
    'countries.csv.zip',
    # 'sample_submission_NDF.csv.zip' # Ini adalah template submission, tidak perlu dimuat untuk EDA
]

# Dictionary untuk menyimpan semua DataFrame yang dimuat
loaded_dataframes = {}

for zip_file in zip_files_to_load:
    df = load_and_unzip_csv(zip_file)
    if df is not None:
        # Nama kunci di dictionary berdasarkan nama file CSV (tanpa .csv.zip)
        df_name = zip_file.replace('.csv.zip', '')
        loaded_dataframes[df_name] = df

df_train = loaded_dataframes.get('train_users_2')
df_test = loaded_dataframes.get('test_users')
df_sessions = loaded_dataframes.get('sessions')
df_age_gender = loaded_dataframes.get('age_gender_bkts')
df_countries = loaded_dataframes.get('countries')


from IPython.display import display

# Iterasi melalui semua DataFrame yang berhasil dimuat
for name, df in loaded_dataframes.items():
    if df is not None:
        print(f"\n{'='*60}")
        print(f" INFORMASI DATASET: {name.upper()}")
        print(f"{'='*60}")

        print(f"Jumlah baris: {df.shape[0]}")
        print(f"Jumlah fitur: {df.shape[1]}")
        
        print("\n Tipe Data dan Non-Null Counts:")
        print("-" * 50)
        df.info()
        
        print("\n 5 Baris Pertama:")
        print("-" * 50)
        display(df.head())

        print("\n Jumlah Nilai yang Hilang (Missing Values):")
        print("-" * 50)
        missing = df.isnull().sum()
        display(missing[missing > 0].to_frame(name='Jumlah Missing'))

        print("\n Statistik Deskriptif (Numerik):")
        print("-" * 50)
        display(df.describe(include='number'))

        print("\n Jumlah Baris Duplikat:")
        print("-" * 50)
        print(df.duplicated().sum())

        print("\n\n")


# Fungsi untuk membersihkan dan merekayasa fitur tanggal & kategori pada DataFrame user
def clean_and_engineer_user_data(df, df_name="Unknown"):
    if df is None:
        print(f"Tidak dapat membersihkan data karena {df_name} adalah None.")
        return None

    print(f"\n--- Membersihkan dan Merekayasa Fitur untuk Dataset: {df_name} ---")

    # date_first_booking: Mengubah ke datetime dan menangani missing
    # Missing values kemungkinan berarti pengguna belum melakukan booking pertama
    df['date_first_booking'] = pd.to_datetime(df['date_first_booking'])

    # age: Capping outliers (usia di bawah 18 dianggap 18, di atas 100 dianggap 100)
    df['age'] = df['age'].apply(lambda x: 18 if x < 18 else (100 if x > 100 else x))
    # Mengisi NaN dengan median usia dari data yang sudah di-capping
    median_age = df['age'].median()
    df['age'].fillna(median_age, inplace=True)

    # gender: Mengisi NaN dengan 'unknown'
    df['gender'].fillna('unknown', inplace=True)

    # first_affiliate_tracked: Mengisi NaN dengan mode (nilai yang paling sering muncul)
    if 'first_affiliate_tracked' in df.columns: # Pastikan kolom ada
        mode_first_affiliate_tracked = df['first_affiliate_tracked'].mode()[0]
        df['first_affiliate_tracked'].fillna(mode_first_affiliate_tracked, inplace=True)

    # Ekstraksi fitur dari kolom tanggal
    df['date_account_created'] = pd.to_datetime(df['date_account_created'])
    # Mengonversi 'timestamp_first_active' yang formatnya YYYYMMDDHHMMSS ke datetime
    df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'], format='%Y%m%d%H%M%S')

    df['year_created'] = df['date_account_created'].dt.year
    df['month_created'] = df['date_account_created'].dt.month
    df['day_created'] = df['date_account_created'].dt.day
    df['weekday_created'] = df['date_account_created'].dt.weekday

    df['year_active'] = df['timestamp_first_active'].dt.year
    df['month_active'] = df['timestamp_first_active'].dt.month
    df['day_active'] = df['timestamp_first_active'].dt.day
    df['weekday_active'] = df['timestamp_first_active'].dt.weekday

    # Untuk 'first_browser', grup kategori langka menjadi 'Other'
    if 'first_browser' in df.columns:
        if 'train_users_2' in loaded_dataframes and loaded_dataframes['train_users_2'] is not None:
            # Hitung frekuensi browser dari data training untuk menentukan "rare"
            browser_counts = loaded_dataframes['train_users_2']['first_browser'].value_counts()
            # Menentukan ambang batas untuk kategori langka 
            threshold = 0.01 * len(loaded_dataframes['train_users_2'])
            rare_browsers = browser_counts[browser_counts < threshold].index
            df['first_browser_grouped'] = df['first_browser'].replace(rare_browsers, 'Other')
        else:
            print("Peringatan: df_train tidak tersedia untuk pengelompokan browser yang konsisten. Melakukan pengelompokan berdasarkan df saat ini.")
            browser_counts = df['first_browser'].value_counts()
            threshold = 0.01 * len(df)
            rare_browsers = browser_counts[browser_counts < threshold].index
            df['first_browser_grouped'] = df['first_browser'].replace(rare_browsers, 'Other')
    
    print("Missing values setelah penanganan:")
    missing_after = df.isnull().sum()[df.isnull().sum() > 0]
    if missing_after.empty:
        print("Tidak ada missing values tersisa.")
    else:
        print(missing_after)

    return df

# Terapkan fungsi pembersihan ke dataset pengguna
if df_train is not None:
    df_train = clean_and_engineer_user_data(df_train, "train_users_2")
if df_test is not None:
    df_test = clean_and_engineer_user_data(df_test, "test_users")


if df_sessions is not None and df_train is not None and df_test is not None:
    print("\n--- Melakukan Feature Engineering dari sessions.csv (Disesuaikan untuk 'secs_elapsed') ---")

    df_sessions_copy = df_sessions.copy()
    df_sessions_copy.rename(columns={'user_id': 'id'}, inplace=True)
    # Mengisi missing values 'secs_elapsed'
    df_sessions_copy['secs_elapsed'].fillna(0, inplace=True) 

    # Fitur agregasi dasar
    sessions_features = df_sessions_copy.groupby('id').agg(
        total_sessions=('id', 'count'),
        total_session_seconds=('secs_elapsed', 'sum'), 
        avg_session_seconds=('secs_elapsed', 'mean'),  
        min_session_seconds=('secs_elapsed', 'min'),
        max_session_seconds=('secs_elapsed', 'max'),
        std_session_seconds=('secs_elapsed', 'std'),
        
        num_unique_actions=('action', 'nunique'),
        num_unique_action_types=('action_type', 'nunique'),
        num_unique_action_details=('action_detail', 'nunique'),
        num_unique_devices=('device_type', 'nunique'),

        most_frequent_action=('action', lambda x: x.mode()[0] if not x.mode().empty else 'unknown_action'),
        most_frequent_action_type=('action_type', lambda x: x.mode()[0] if not x.mode().empty else 'unknown_type'),
        most_frequent_device=('device_type', lambda x: x.mode()[0] if not x.mode().empty else 'unknown_device'),
    )

    # Fitur frekuensi aksi spesifik
    # Cek unik nilai 'action' di df_sessions_copy
    unique_actions = df_sessions_copy['action'].unique()
    action_counts = df_sessions_copy.groupby(['id', 'action']).size().unstack(fill_value=0)
    action_counts.columns = [f'action_{col}_count' for col in action_counts.columns]
    sessions_features = sessions_features.merge(action_counts, on='id', how='left')

    unique_action_types = df_sessions_copy['action_type'].unique()
    action_type_counts = df_sessions_copy.groupby(['id', 'action_type']).size().unstack(fill_value=0)
    action_type_counts.columns = [f'action_type_{col}_count' for col in action_type_counts.columns]
    sessions_features = sessions_features.merge(action_type_counts, on='id', how='left')

    unique_action_details = df_sessions_copy['action_detail'].unique()
    action_detail_counts = df_sessions_copy.groupby(['id', 'action_detail']).size().unstack(fill_value=0)
    action_detail_counts.columns = [f'action_detail_{col}_count' for col in action_detail_counts.columns]
    sessions_features = sessions_features.merge(action_detail_counts, on='id', how='left')

    unique_device_types = df_sessions_copy['device_type'].unique()
    device_type_counts = df_sessions_copy.groupby(['id', 'device_type']).size().unstack(fill_value=0)
    device_type_counts.columns = [f'device_type_{col}_count' for col in device_type_counts.columns]
    sessions_features = sessions_features.merge(device_type_counts, on='id', how='left')

    # Fitur Rasio Aksi
    # Ambil daftar kolom action_counts yang tersedia
    available_action_counts = [col for col in action_counts.columns if col.startswith('action_')]

    if 'action_booking_request_count' in available_action_counts:
        sessions_features['ratio_booking_req_total_sessions'] = sessions_features['action_booking_request_count'] / sessions_features['total_sessions']
        sessions_features['ratio_booking_req_total_sessions'].fillna(0, inplace=True) # handle division by zero
    else:
        sessions_features['ratio_booking_req_total_sessions'] = 0 # Tambahkan kolom ini jika tidak ada aksi

    if 'action_search_results_count' in available_action_counts:
        sessions_features['ratio_search_results_total_sessions'] = sessions_features['action_search_results_count'] / sessions_features['total_sessions']
        sessions_features['ratio_search_results_total_sessions'].fillna(0, inplace=True)
    else:
        sessions_features['ratio_search_results_total_sessions'] = 0

    # Fitur Waktu Aksi Spesifik (rata-rata secs_elapsed per action_type)
    action_type_time_avg = df_sessions_copy.groupby(['id', 'action_type'])['secs_elapsed'].mean().unstack(fill_value=0)
    action_type_time_avg.columns = [f'avg_sec_per_action_type_{col}' for col in action_type_time_avg.columns]
    sessions_features = sessions_features.merge(action_type_time_avg, on='id', how='left')

    sessions_features.reset_index(inplace=True)

    # Isi missing values yang mungkin muncul setelah merge atau unstack
    numerical_session_cols = sessions_features.select_dtypes(include=np.number).columns.drop('id', errors='ignore')
    for col in numerical_session_cols:
        sessions_features[col].fillna(0, inplace=True) # Gunakan 0 untuk count/time yang tidak ada

    categorical_session_cols = sessions_features.select_dtypes(include='object').columns.drop('id', errors='ignore')
    for col in categorical_session_cols:
        sessions_features[col].fillna('no_session_data', inplace=True)

    df_train = pd.merge(df_train, sessions_features, on='id', how='left')
    df_test = pd.merge(df_test, sessions_features, on='id', how='left')

    for col in sessions_features.columns:
        if col != 'id':
            if sessions_features[col].dtype == 'object':
                df_train[col].fillna('no_session_data', inplace=True)
                df_test[col].fillna('no_session_data', inplace=True)
            else:
                df_train[col].fillna(0, inplace=True)
                df_test[col].fillna(0, inplace=True)

    print("Fitur sesi berhasil digabungkan dan missing values ditangani.")
    print(f"df_train shape setelah merge sesi: {df_train.shape}")
    print(f"df_test shape setelah merge sesi: {df_test.shape}")
else:
    print("df_sessions tidak tersedia atau df_train/df_test belum dimuat, melewati rekayasa fitur sesi.")


if df_train is not None:
    df_train['booking_status'] = df_train['country_destination'].apply(lambda x: 'booked' if x != 'NDF' else 'NDF')
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_train, x='booking_status', order=df_train['booking_status'].value_counts().index, palette='coolwarm')
    plt.title('Distribusi Status Booking Pengguna (Data Latih)')
    plt.xlabel('Status Booking')
    plt.ylabel('Jumlah Pengguna')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.violinplot(data=df_train, x='booking_status', y='age', palette='viridis')
    plt.title('Distribusi Usia Berdasarkan Status Booking (Data Latih)')
    plt.xlabel('Status Booking')
    plt.ylabel('Usia')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_train, x='signup_app', hue='booking_status', palette='muted')
    plt.title('Status Booking Berdasarkan Aplikasi Pendaftaran (Data Latih)')
    plt.xlabel('Aplikasi Pendaftaran')
    plt.ylabel('Jumlah Pengguna')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Status Booking')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

else:
    print("Dataset 'train_users_2' tidak tersedia untuk melakukan EDA visualisasi.")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from scipy.stats import uniform, randint

if df_train is not None and df_test is not None:
    base_features = [
        'gender', 'age', 'signup_method', 'signup_flow', 'language',
        'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
        'signup_app', 'first_device_type', 'first_browser_grouped',
        'year_created', 'month_created', 'day_created', 'weekday_created',
        'year_active', 'month_active', 'day_active', 'weekday_active',
        'days_diff_created_active'
    ]

    session_added_features_candidates = [col for col in sessions_features.columns if col != 'id']
    
    session_added_features = []
    for col in session_added_features_candidates:
        if col in df_train.columns and col in df_test.columns:
            session_added_features.append(col)

    features = base_features + session_added_features

    actual_features_train = [f for f in features if f in df_train.columns]
    actual_features_test = [f for f in features if f in df_test.columns]
    
    features = list(set(actual_features_train) & set(actual_features_test))

    missing_train = [f for f in features if f not in df_train.columns]
    missing_test = [f for f in features if f not in df_test.columns]
    if missing_train:
        print(f"Peringatan: Beberapa fitur yang diharapkan tidak ada di df_train: {missing_train}")
    if missing_test:
        print(f"Peringatan: Beberapa fitur yang diharapkan tidak ada di df_test: {missing_test}")

    X = df_train[features].copy()
    y = df_train['country_destination'].apply(lambda x: 'booked' if x != 'NDF' else 'NDF')
    X_test_final = df_test[features].copy()

    categorical_features_names = []
    for col in X.columns:
        if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]): 
            categorical_features_names.append(col)
            X[col] = X[col].astype('category')
            X_test_final[col] = X_test_final[col].astype('category')
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    target_names_binary = le.inverse_transform([0, 1])

    class_counts = pd.Series(y_encoded).value_counts()
    booked_label_encoded = le.transform(['booked'])[0]
    ndf_label_encoded = le.transform(['NDF'])[0]
    
    class_weights_dict = {}
    class_weights_dict[ndf_label_encoded] = 1 
    class_weights_dict[booked_label_encoded] = class_counts[ndf_label_encoded] / class_counts[booked_label_encoded]
    
    print(f"Bobot Kelas untuk CatBoost: {class_weights_dict}")

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print(f"\nDimensi data setelah identifikasi fitur dan sebelum CatBoost:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test_final shape: {X_test_final.shape}")
    print(f"Fitur Kategorikal yang diidentifikasi: {categorical_features_names}")

    # Inisialisasi CatBoostClassifier  
    base_model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
        cat_features=categorical_features_names,
        class_weights=class_weights_dict,
        task_type="GPU",
        gpu_ram_part=0.75, 
        # devices='0' # Jika ingin memaksa hanya menggunakan satu GPU 
    )

    # --- Hyperparameter Tuning dengan RandomizedSearchCV ---
    print("\nMemulai Hyperparameter Tuning dengan RandomizedSearchCV...")
    
    param_distributions = {
        
        'iterations': randint(300, 1200), 
        'learning_rate': uniform(loc=0.01, scale=0.08), 
        'depth': randint(4, 8), 
        'l2_leaf_reg': uniform(loc=1, scale=4), 
        'bootstrap_type': ['Bernoulli'], 
        'subsample': uniform(loc=0.6, scale=0.3), 
    }

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=10, 
        cv=2, 
        scoring='accuracy',
        random_state=42,
        n_jobs=1, 
        verbose=10
    )

    random_search.fit(X_train, y_train)

    print("\nHyperparameter Tuning selesai.")
    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation accuracy: {:.4f}".format(random_search.best_score_))

    catboost_model = random_search.best_estimator_
    
    # Simpan model ke file
    catboost_model.save_model("catboost_model.cbm")
    print("\nModel Catboost berhasil disimpan ke file: best_catboost_model.cbm")

    y_pred = catboost_model.predict(X_val)
    
    print("\n--- Evaluasi Model CatBoost Terbaik (Binary Classification) ---")
    print(f"Akurasi: {accuracy_score(y_val, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=target_names_binary))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_binary, yticklabels=target_names_binary)
    plt.title('Confusion Matrix (Binary Classification - CatBoost)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    test_predictions_encoded = catboost_model.predict(X_test_final)
    test_predictions_status = le.inverse_transform(test_predictions_encoded)

    # Buat DataFrame hasil prediksi
    sample_submission = pd.DataFrame({'id': df_test['id'], 'booking_status': test_predictions_status})

    # Simpan hasil prediksi ke file CSV
    sample_submission.to_csv("catboost_test_predictions.csv", index=False)
    print("\n Hasil prediksi test set berhasil disimpan ke file: catboost_test_predictions.csv")

else:
    print("Tidak dapat melakukan modeling karena dataset train_users_2 atau test_users tidak tersedia.")



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  

if df_train is not None and df_test is not None:
    base_features = [
        'gender', 'age', 'signup_method', 'signup_flow', 'language',
        'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
        'signup_app', 'first_device_type', 'first_browser_grouped',
        'year_created', 'month_created', 'day_created', 'weekday_created',
        'year_active', 'month_active', 'day_active', 'weekday_active',
        'days_diff_created_active'
    ]

    session_added_features_candidates = [col for col in sessions_features.columns if col != 'id']
    session_added_features = [col for col in session_added_features_candidates if col in df_train.columns and col in df_test.columns]
    features = list(set(base_features + session_added_features) & set(df_train.columns) & set(df_test.columns))

    X = df_train[features].copy()
    y = df_train['country_destination'].apply(lambda x: 'booked' if x != 'NDF' else 'NDF')
    X_test_final = df_test[features].copy()

    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    print(f"Melakukan one-hot encoding pada kolom: {categorical_cols}")

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_test_final = pd.get_dummies(X_test_final, columns=categorical_cols, drop_first=True)
    X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    target_names_binary = le.inverse_transform([0, 1])

    class_counts = pd.Series(y_encoded).value_counts()
    booked_label_encoded = le.transform(['booked'])[0]
    ndf_label_encoded = le.transform(['NDF'])[0]
    class_weights_dict = {
        ndf_label_encoded: 1.0,
        booked_label_encoded: class_counts[ndf_label_encoded] / class_counts[booked_label_encoded]
    }

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    print(f"\nShape data untuk Random Forest:")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test_final.shape}")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight=class_weights_dict,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Simpan model ke file
    joblib.dump(rf_model, "best_random_forest_model.pkl")
    print("\nModel Random Forest berhasil disimpan sebagai best_random_forest_model.pkl")

    y_pred = rf_model.predict(X_val)

    print("\n--- Evaluasi Model Random Forest ---")
    print(f"Akurasi: {accuracy_score(y_val, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=target_names_binary))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=target_names_binary, yticklabels=target_names_binary)
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    test_preds_encoded = rf_model.predict(X_test_final)
    test_preds_label = le.inverse_transform(test_preds_encoded)

    submission_rf = pd.DataFrame({
        'id': df_test['id'],
        'booking_status': test_preds_label
    })

    # Simpan hasil prediksi ke file CSV
    submission_rf.to_csv("random_forest_test_predictions.csv", index=False)
    print("Hasil prediksi test set disimpan ke file: random_forest_test_predictions.csv")
else:
    print("Tidak dapat menjalankan Random Forest karena df_train atau df_test tidak tersedia.")




