#kütüphaneleri çağırma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


#verilerin incelenmesi, veriler eğitim ve test olarak hazırlanmıştır. Verilerin eğitimi için train set kullanılacaktır,
#verilerin içeriği, eksik veri olup olmadığı değerlendirilecektir.

df_train=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

df_train.head(3)


df_test.head(3)


df_train.info()


df_train.isnull().values.any()


sns.histplot(data=df_train,x="day",hue="rainfall",kde=True, bins=30)


max_day = df_train["day"].max()
print(f"Gün sayısının maksimum değeri: {max_day}")


df=df_train.drop(columns=["id"])
df.head()


#Veri görselleştirme ile hangi parametrenin Rainfall üzerindeki etkisi değerlendirilir- Amacımız gün içinde değişen parametrelere göre yağmurun yağıp yağmayacağıdır.
tum_parametreler=df[["pressure","maxtemp","temparature","mintemp","dewpoint","humidity","cloud","sunshine","winddirection","windspeed","rainfall"]]
corr=tum_parametreler.corr()
sns.heatmap(corr,cmap="RdBu", vmin=-1, vmax=1, annot=True, annot_kws={"size": 8},linewidths=0.3)
plt.show()


# yalnızca rainfall için heatmap çizimi
corr=df.corr()[["rainfall"]].sort_values(by="rainfall",ascending=False)
heatmap=sns.heatmap(corr,vmin=-1, vmax=1,annot=True,cmap="BrBG")
heatmap.set_title("Features  Correlating with Rainfall",fontdict={"fontsize":18},pad=12);
plt.show()


#Train setini train ve test olarak böl;
y=df["rainfall"]
x=df.drop("rainfall",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=27,train_size=0.73)


#modeli kur
rf=RandomForestClassifier(n_estimators=470,max_depth=7,random_state=27,criterion="entropy")
model=rf.fit(x_train,y_train)
model.score(x_test,y_test)


# Feature Importance değerlerini al
feature_importances = model.feature_importances_

# Sonuçları görselleştir
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=x.columns, palette="coolwarm")
plt.xlabel("Özellik Önem Skoru")
plt.ylabel("Özellikler")
plt.title("Özelliklerin Rainfall Üzerindeki Etkisi")
plt.show()


#Doğrulama setinde tahmin yap (olasılık olarak)
y_test_pred_proba = rf.predict_proba(x_test)[:, 1]  # Pozitif sınıfın olasılıklarını al

#ROC AUC Skorunu Hesapla
auc_score = roc_auc_score(y_test, y_test_pred_proba)
print(f"Validation ROC AUC Score: {auc_score:.4f}")


fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


#eğitimle oluşturulan modelin test veri  seti ile çalışılarak rainfall değerlerinin tahmin edilmesi
df_test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
df_test.info() 


print(df_test.describe())


median_value = df_test["winddirection"].median()
print(f"Winddirection Median (Medyan): {median_value:.4f}")


#eksik veri olan rüzgaryönü "winddirection" için ortalama değeri ile bu boşluk tamamlanır
df_test["winddirection"].fillna(df_test.winddirection.mean(),inplace=True)
df_test.isnull().sum()


df_test.head()


test_modeli=model.predict_proba(df_test.drop(columns=["id"]))[:,1]
submission=pd.DataFrame({"id": df_test["id"],"rainfall": test_modeli})
print(submission.head())


submission.to_csv("submission.csv", index=False)
print("✅ Submission file 'submission.csv' başarıyla oluşturuldu!")


print(submission.head())  
print(submission.info()) 

