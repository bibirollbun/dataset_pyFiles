import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import seaborn as sns


data=pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
data.head()


test=pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")
test.head()


data.describe()


t=data.isnull().sum()
print(t)


def outliers_removal(df,columnName):
    q1=df[columnName].quantile(0.25)
    q3=df[columnName].quantile(0.75)
    IQR=q3-q1
    lower_bound=q1-1.5*IQR
    higher_bound=q3+1.5*IQR
    df_cleaned = df[(df[columnName] >= lower_bound) & (df[columnName] <= higher_bound)]
    return df_cleaned
data=outliers_removal(data,'Previous Claims')
test=outliers_removal(test,'Previous Claims')
data=outliers_removal(data,'Credit Score')
test=outliers_removal(data,'Credit Score')
print(data)


t=data.isnull().sum()
print(t)


print(test.isnull().sum())


categorical_iunique=["Marital Status","Number of Dependents","Occupation","Customer Feedback","Customer Feedback"]
drive_unique=[]
for i in categorical_iunique:
    en=list(data[i].unique())
    drive_unique.append(en)
dataframe_unique=pd.DataFrame(drive_unique)
print(dataframe_unique)
print(data["Number of Dependents"].mode()[0])


def input_preprocessing(data):
    if "Age" in data.columns:
        if not data["Age"].mode().empty:
            data["Age"].fillna(data["Age"].mode()[0], inplace=True)
    if "Occupation" in data.columns:
        data["Occupation"].fillna("eks", inplace=True)
    if "Marital Status" in data.columns:
        data["Marital Status"].fillna("Single", inplace=True)
    if "Number of Dependents" in data.columns:
        if not data["Number of Dependents"].mode().empty:
            data["Number of Dependents"].fillna(data["Number of Dependents"].mode()[0], inplace=True)
    if "Annual Income" in data.columns:
        if not data["Annual Income"].isnull().all():
            data["Annual Income"].fillna(data["Annual Income"].mean(), inplace=True)
    if "Health Score" in data.columns:
        if not data["Health Score"].isnull().all():
            data["Health Score"].fillna(data["Health Score"].mean(), inplace=True)
    if "Customer Feedback" in data.columns:
        if not data["Customer Feedback"].mode().empty:
            data["Customer Feedback"].fillna(data["Customer Feedback"].mode()[0], inplace=True)

# Apply preprocessing to both datasets
input_preprocessing(data)
input_preprocessing(test)

# Check for remaining null values
print(data.isnull().sum())



print(data.isnull().sum())


data=data.dropna()
print(data.isnull().sum())


def date(df):

    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Year'] = df['Policy Start Date'].dt.year
    df['Day'] = df['Policy Start Date'].dt.day
    df['Month'] = df['Policy Start Date'].dt.month

    df['Year_sin'] = np.sin(2 * np.pi * df['Year'])
    df['Year_cos'] = np.cos(2 * np.pi * df['Year'])
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12) 
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)  
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df['Group']=(df['Year']-2020)*48+df['Month']*4+df['Day']//7
    
    df.drop('Policy Start Date', axis=1, inplace=True)

    return df

data=date(data)



#implement pca to reduction and get correlation each dimension
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
def encode_all(data,k):
    for i in k:
        data[i]=encoder.fit_transform(data[i])
categorical_iunique=["Marital Status","Location","Occupation","Smoking Status","Policy Type","Exercise Frequency","Property Type","Customer Feedback","Gender","Education Level"]
encode_all(data,categorical_iunique)
encode_all(test,categorical_iunique)


data.head()


from sklearn.preprocessing import StandardScaler
skalarisasi=StandardScaler()
scaled_data=skalarisasi.fit_transform(data)
# columns_with_premium = data.apply(lambda col: col.astype(str).str.contains('Premium')).any()
# print(columns_with_premium)


pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Hasil PC1 dan PC2
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
print(df_pca)

# Visualisasi PC1 dan PC2
plt.scatter(df_pca['PC1'], df_pca['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Visualisasi PCA')
plt.show()



explained_variance = pca.explained_variance_ratio_
plt.bar([f'PC{i+1}' for i in range(len(explained_variance))], explained_variance)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance per Komponen Utama')
plt.show()


loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=data.columns)

# 4. Menampilkan kontribusi kolom terhadap PC1
print("Kontribusi Kolom terhadap PC1:")
print(loadings['PC1'])

# 5. Menampilkan komponen utama pertama (nilai PC1 untuk setiap observasi)
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
print("\nNilai PC1 untuk setiap observasi:")
print(df_pca['PC1'])


from sklearn.ensemble import RandomForestRegressor
good_pca=["Education Level","Occupation","Policy Type","Previous Claims","Smoking Status","Exercise Frequency","Property Type","Year"]
target="Premium Amount"
X=data[good_pca]
y=data[target]
rfc=RandomForestRegressor()
rfc.fit(X,y)
importances = rfc.feature_importances_
feature_names = good_pca

# Visualisasi Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance dari Random Forest')
plt.show()

