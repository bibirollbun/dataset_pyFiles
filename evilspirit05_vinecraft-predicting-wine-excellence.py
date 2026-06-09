import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats.mstats import winsorize
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

%matplotlib inline


df=pd.read_csv("/kaggle/input/chydv-hackathon-2025/train.csv")


df.head()


df.shape


df.isnull().sum()


df.info()


df.describe()


df.duplicated()


df["quality"].value_counts()


df.drop(columns=["id"],axis=1,inplace=True)


plt.figure(figsize=(10,8))
sns.countplot(x="quality",data=df,palette="gnuplot2")
plt.show()


quality_counts = df['quality'].value_counts()

plt.figure(figsize=(15,10))
plt.pie(quality_counts, labels=quality_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2", 6).as_hex(), wedgeprops={'edgecolor': 'black'})
plt.gca().set_aspect('equal')
plt.title('Wine Quality Distribution')
plt.axis("equal")
plt.show()


final_df=df.copy()


X = df.drop('quality', axis=1)
y = df['quality']

scaler = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform', subsample=None)
X_discretized = scaler.fit_transform(X)

chi2_selector = SelectKBest(chi2, k='all')
X_kbest = chi2_selector.fit_transform(X_discretized, y)

chi2_values = chi2_selector.scores_
p_values = chi2_selector.pvalues_

chi2_df = pd.DataFrame({'feature': X.columns, 'chi2_value': chi2_values, 'p_value': p_values})
chi2_df = chi2_df.sort_values(by='chi2_value', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='chi2_value', y='feature', data=chi2_df, palette="brg")
plt.title('Chi-square Feature Importance')
plt.xlabel("Chi-square Score")
plt.ylabel("Features")
plt.show()


chi2_df.head()


col=["alcohol","total sulfur dioxide","fixed acidity","sulphates","citric acid","free sulfur dioxide","quality"]


final_df=final_df[col]


final_df.head()



plt.figure(figsize=(12, 8))
for i, feature in enumerate(["alcohol", "total sulfur dioxide", "fixed acidity", "free sulfur dioxide"]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(final_df[feature], bins=30, kde=True, color=sns.color_palette("husl")[i])
    plt.title(feature)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.barplot(x="quality", y="alcohol", data=final_df, palette="muted")
plt.title('Mean Alcohol Content by Quality')
plt.show()



plt.figure(figsize=(12, 8))
features = ["alcohol", "total sulfur dioxide", "fixed acidity", "sulphates", "citric acid", "free sulfur dioxide"]
sns.boxplot(data=final_df[features])
plt.xticks(rotation=45)
plt.title("Box Plot of Selected Features")
plt.show()



final_df=final_df.rename({"total sulfur dioxide": "total_sulfur_dioxide"},axis=1)


lower_bound = final_df["total_sulfur_dioxide"].quantile(0.05)
upper_bound = final_df["total_sulfur_dioxide"].quantile(0.95)

final_df["total_sulfur_dioxide"] = np.clip(final_df["total_sulfur_dioxide"], lower_bound, upper_bound)

lower_bound = final_df["fixed acidity"].quantile(0.05)
upper_bound = final_df["fixed acidity"].quantile(0.95)

final_df["fixed acidity"] = np.clip(final_df["fixed acidity"], lower_bound, upper_bound)



plt.figure(figsize=(12, 8))
features = ["alcohol", "total_sulfur_dioxide", "fixed acidity", "sulphates", "citric acid", "free sulfur dioxide"]
sns.boxplot(data=final_df[features])
plt.xticks(rotation=45)
plt.title("Box Plot of Selected Features")
plt.show()



final_df['sulfur_ratio'] = final_df['free sulfur dioxide'] / final_df['total_sulfur_dioxide']


new_col=["alcohol","total_sulfur_dioxide","fixed acidity","sulphates","citric acid","free sulfur dioxide","sulfur_ratio","quality"]


final_df=final_df[new_col]


print(final_df.corr()['quality'].sort_values(ascending=False))



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features_to_scale = ["alcohol", "total_sulfur_dioxide", "fixed acidity", 
                     "sulphates", "citric acid", "free sulfur dioxide", "sulfur_ratio"]

final_df[features_to_scale] = scaler.fit_transform(final_df[features_to_scale])



final_df.head()


X_1 = final_df.drop("quality", axis=1)
y_1 = final_df["quality"].astype(int)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_1, y_1, test_size=0.2, random_state=42, stratify=y)


model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=100)
model.fit(X_train2, y_train2)

y_pred = model.predict(X_test2)


accuracy = accuracy_score(y_test2, y_pred)
print(f"Accuracy: {accuracy:.4f}")


from sklearn.ensemble import GradientBoostingClassifier
model2 = GradientBoostingClassifier()
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)


accuracy = accuracy_score(y_test2, y_pred2)
print("Accuracy: ", accuracy)


df_test=pd.read_csv("/kaggle/input/chydv-hackathon-2025/test.csv")


df_test.head()


Id=df_test.id


col=["alcohol","total sulfur dioxide","fixed acidity","sulphates","citric acid","free sulfur dioxide"]
df_test=df_test[col]


df_test.head()


df_test['sulfur_ratio'] = df_test['free sulfur dioxide'] / df_test['total sulfur dioxide']


df_test=df_test.rename({"total sulfur dioxide": "total_sulfur_dioxide"},axis=1)


df_test.head()


prediction=model.predict(df_test)
prediction=prediction.flatten()
submission=pd.DataFrame({"id":Id,"quality":prediction})
submission.to_csv("submission.csv",index=False)


submission.head()




