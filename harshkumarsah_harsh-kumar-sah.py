import pandas as pd 
import numpy as np 
import seaborn as sns
import scipy.stats as stat
import pylab 
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df=pd.read_csv('/kaggle/input/k-means-clustering-for-heart-disease-analysis/heart_disease.csv')
df.head()


df.info()


df.isnull


df.shape


df.columns


submission_id=df['id']


df=df.drop(["id"],axis=1)


numerical_features = [feature for feature in df.columns if df[feature].dtypes !='object']
categorical_features= [feature for feature in df.columns if df[feature].dtypes =='object']
discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())<20]
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]


#continuos feature korelasyon
plt.figure(figsize=(10, 5))
sns.heatmap(df[continuous_feature].corr().abs(), cmap='Greens',mask=np.triu(df[continuous_feature].corr()),fmt = '.2%', annot=True)


###HEATMAP ÜZERİNDEN EKSİK VERİLERİ KONTROL EDELİM:
plt.figure(figsize=(18,6))
plt.title('Heatmap of Missing Values')
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


#LİSTE ÜZERİNDEN EKSİK VERİLERİN YÜZDELİK ORANLARINA BAK :
features_with_na=[features for features in df.columns if df[features].isnull().sum()>1]
for feature in features_with_na:
    print(feature, ' % ',np.round(df[feature].isnull().mean(), 4),  'missing values')


#EKSİK VERİLERİN BİRBİRLERİ İLE  KORELASYONLARINI İNCELEYİM(MUTLAK DEĞERİNE GÖRE(abs))
plt.figure(figsize=(12, 6))
plt.title('Correlation of Missing Values')
sns.heatmap(df.isnull().corr().abs(), cmap='rainbow',linewidths = 1,mask=np.triu(df.isnull().corr()),fmt = '.2f', annot=True)


df[['thalch']]=df[['thalch']].fillna(0)
df[['oldpeak']]=df[['oldpeak']].fillna(0)
df[['chol']]=df[['chol']].fillna(0)
df[['trestbps']]=df[['trestbps']].fillna(0)
df[['ca']]=df[['ca']].fillna(0)
df[['exang']]=df[['exang']].fillna('None')
df[['restecg']]=df[['restecg']].fillna('None')
df[['fbs']]=df[['fbs']].fillna('None')
df[['thal']]=df[['thal']].fillna('None')
df[['slope']]=df[['slope']].fillna('None')


df.isna().sum()


df['oldpeak'].unique()


# Örnek Oldpeak sınıflandırması
def classify_oldpeak(oldpeak_value):
    if oldpeak_value < -2.5:
        return 'Şiddetli ST Depresyonu'
    elif -2.5 <= oldpeak_value < -1:
        return 'Orta Derecede ST Depresyonu'
    elif -1 <= oldpeak_value < 0:
        return 'Hafif ST Depresyonu'
    elif 0 <= oldpeak_value < 1:
        return 'Hafif ST Yükselmesi'
    elif 1 <= oldpeak_value < 2:
        return 'Orta Derecede ST Yükselmesi'
    else:
        return 'Şiddetli ST Yükselmesi'

# DataFrame'e uygula
df['Oldpeak_Category'] = df['oldpeak'].apply(classify_oldpeak)


cholesterol_values = np.array([199, 200, 240])
low_limit = 200
high_limit = 240
df['Cholesterol_Group'] = np.where(df['chol'] <= low_limit, 0, np.where(df['chol']< high_limit, 1, 2)).astype(int)


df['Age_Group'] = pd.cut(df['age'], bins=[19, 29, 49, 64, 99], labels = [0, 1, 2, 3]).astype('int16')


#Normal Dinlenme Kan Basıncı:
#Sistolik: 90 - 120 mmHg
#Diyastolik: 60 - 80 mmHg
#Ancak, bu değerler kişisel faktörlere ve sağlık durumuna bağlı olarak değişebilir. Ayrıca, hipertansiyon (yüksek kan basıncı) tanısı için kullanılan sınıflandırmalar genellikle şu şekildedir:

#Normal: Sistolik < 120 mmHg ve Diyastolik < 80 mmHg
#Yüksek Normal: Sistolik 120-129 mmHg ve Diyastolik < 80 mmHg
#Hipertansiyon Evre 1: Sistolik 130-139 mmHg veya Diyastolik 80-89 mmHg
#Hipertansiyon Evre 2: Sistolik ≥ 140 mmHg veya Diyastolik ≥ 90 mmHg
normal_trestbps=(60,120)
df['trestbps_Group'] = np.where(df['trestbps'] <= low_limit, 0, np.where(df['trestbps']< high_limit, 1, 2)).astype(int)



#Normal: 120≤Thalach≤160
normal_thalch=(60,120)
df['thalch_Group'] = np.where(np.logical_and(df['thalch'] >= normal_thalch[0] , df['thalch']<=normal_thalch[1]), 1, 0).astype(int)


df=df.drop(['chol','age','trestbps'],axis=1)


df=pd.get_dummies(df, drop_first=True, dtype=int)


df.head()


numerical_features = [feature for feature in df.columns if df[feature].dtypes !='object']
categorical_features= [feature for feature in df.columns if df[feature].dtypes =='object']
discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())<20]
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]


###HİSTOGRAM ÜZERİNDE SÜREKLİ SAYISAL DEĞİŞKENLERİN ÖN KONTROLLERİNİ YAP VE ÇARPIKLIKLARI VE AYKIRILIKLARI VARSA GİDER:

plt.figure(figsize=(30,120),facecolor='white')
plotnumber=1
for feature in continuous_feature:
                plt.figure(figsize=(23,4))
                plt.subplot(1,3,1)
                sns.histplot(df[feature],color="gray", kde=True)
                plt.subplot(1,3,2)
                sns.boxplot(df[feature],color='red')
                plt.subplot(1,3,3)
                stat.probplot(df[feature],dist='norm',plot=pylab)
plt.show()


for col in continuous_feature:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)
    df[col]= np.where(df[col] > upper, upper, np.where(df[col] < lower, lower, df[col]))


plt.figure(figsize=(30,120),facecolor='white')
plotnumber=1
for feature in continuous_feature:
                plt.figure(figsize=(23,4))
                plt.subplot(1,3,1)
                sns.histplot(df[feature],color="green", kde=True)
                plt.subplot(1,3,2)
                sns.boxplot(df[feature],color='green')
                plt.subplot(1,3,3)
                stat.probplot(df[feature],dist='norm',plot=pylab)
plt.show()


X = df.iloc[:, ].values
X


from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Farklı küme sayıları için Silhouette Score'u hesapla
for n_clusters in range(2, 11):
    kmeans =  KMeans(n_clusters = n_clusters, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"Küme Sayısı = {n_clusters}, Silhouette Score = {silhouette_avg}")


kmeans = KMeans(n_clusters = 2, init = 'k-means++', n_init=10)

y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Heart Disease')
plt.legend()
plt.show()


submission=pd.read_csv("/kaggle/input/k-means-clustering-for-heart-disease-analysis/sample.csv")
submission["cluster"] = y_kmeans[:len(submission)]
submission.to_csv('submission6.csv', index=False)
submission

