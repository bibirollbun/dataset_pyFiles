import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Loading Data
train = pd.read_csv("C:\\Users\\asus\\Downloads\\train.csv")
test = pd.read_csv("C:\\Users\\asus\\Downloads\\test.csv")
submission = pd.read_csv("C:\\Users\\asus\\Downloads\\sample_submission.csv")


# Preprocessing Data
def preprocess(df): 
    df = df.copy()
    
    cols = [
        'name', 'description', 'neighborhood_overview', 'host_url', 'host_name',
        'host_about', 'host_verifications', 'first_review', 'last_review',
        'host_location', 'host_neighbourhood', 'host_response_time', 'host_since',
        'bathrooms_text', 'amenities', 'neighbourhood', 'neighbourhood_cleansed'
    ]
    df = df.drop(columns=[col for col in cols if col in df.columns], errors = 'ignore')
    
    # Konversi boolean ke 1/0 
    torf = [
        'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability'
    ]
    
    for col in torf :
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0})
    
    # Isi NaN menjadi median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
        
    # Konversi persen ke float
    percent = ['host_response_rate', 'host_acceptance_rate']
    for col in percent:
        if col in df.columns:
            df[col] = df[col].str.rstrip('%').astype(float)
    # Membatasi dan Encoding kategorikal
    cat_cols = ['property_type','room_type','city']
    df = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns], drop_first = True)
    
    return df


# Menyiapkan Fitur dan Target
if 'price' in train.columns:
    train['price'] = train['price'].replace(r'[\$,]','', regex = True).astype(float)
    y = train['price']
    x = preprocess(train.drop(columns=['price']))
else:
    raise ValueError("Kolom 'price' tidak ditemukan di train.csv")

# Simpan id test untuk submission
if 'id' in test.columns:
    ID = test['id']
else:
    raise ValueError("Kolom 'id' tidak ditemukan di test.csv")

xTest = preprocess(test)
xTest = xTest.reindex(columns=x.columns, fill_value=0)



# Train-test split buat validasi
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.2, random_state=42)


# Training Model
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(x_train, y_train)



# Evaluasi Model
valuation = model.predict(x_val)
rmse = np.sqrt(mean_squared_error(y_val,valuation))
print(f"Validation RMSE : {rmse:.4f}")


# Prediksi Data Test Dan Buat CSV
y_test = model.predict(xTest)

submission = pd.DataFrame({
    'id': ID,
    'price': y_test
})

submission.to_csv("submission.csv",index=False)
print("File submission.csv telah disimpan")

