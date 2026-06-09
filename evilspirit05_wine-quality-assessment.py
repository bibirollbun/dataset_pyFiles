import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report

%matplotlib inline


df=pd.read_csv("/kaggle/input/chydv-hackathon-2025/train.csv")


df.head()


df.isnull().sum()


df.shape


df.drop(columns=["id"],axis=1,inplace=True)


df.describe()


df.info()


df["quality"].value_counts()


continuous_features = [
    'fixed acidity', 
    'volatile acidity', 
    'citric acid', 
    'residual sugar', 
    'chlorides', 
    'free sulfur dioxide', 
    'total sulfur dioxide', 
    'density', 
    'pH', 
    'sulphates', 
    'alcohol']



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[continuous_features] = scaler.fit_transform(df[continuous_features])


df.head()


from sklearn.utils import resample
majority_class = df[df["quality"] == 5.0]
majority_class_2=df[df["quality"]==6.0]
minority_class_3=df[df["quality"]==3.0]
minority_class_7=df[df["quality"]==7.0]
minority_class_4 = df[df["quality"] == 4.0]
minority_class_8 = df[df["quality"] == 8.0]
class_3_upsample=resample(minority_class_3,replace=True,n_samples=len(majority_class),random_state=42)
class_4_upsample=resample(minority_class_4,replace=True,n_samples=len(majority_class),random_state=42)
class_7_upsample=resample(minority_class_7,replace=True,n_samples=len(majority_class),random_state=42)
class_8_upsample=resample(minority_class_8,replace=True,n_samples=len(majority_class),random_state=42)

df_balanced = pd.concat([majority_class,majority_class_2,class_3_upsample, class_4_upsample, class_7_upsample, class_8_upsample])


df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


df_balanced.head()


df_balanced.shape


df_balanced.isnull().sum()


df_balanced["quality"].value_counts()


X=df_balanced.drop(columns=["quality"],axis=1)
y=df_balanced["quality"]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from catboost import CatBoostClassifier



from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer

# Define the model
model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8,verbose=500, early_stopping_rounds=50)

# Define stratified k-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get cross-validated predictions
y_pred = cross_val_predict(model, X_train, y_train, cv=cv)

# Calculate Quadratic Weighted Kappa score
qwk_score = cohen_kappa_score(y_train, y_pred, weights="quadratic")

print(f'Quadratic Weighted Kappa Score: {qwk_score}')



model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8,verbose=500,early_stopping_rounds=50)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)  

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Cohen's Kappa score
kappa = cohen_kappa_score(y_test, y_pred,weights='quadratic')
print(f"Cohen's Kappa score: {kappa}")


class_label=["Three","Four","Five","Six","Seven","Eight"]

class_report = classification_report(y_test, y_pred,target_names=class_label)
print(f"Classification Report:\n{class_report}")


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="gnuplot2", linewidths=2, linecolor="black", xticklabels=class_label, yticklabels=class_label)

plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.show()


df_test=pd.read_csv("/kaggle/input/chydv-hackathon-2025/test.csv")


df_test.head()


Id=df_test.id


df_test.drop(columns=["id"],axis=1,inplace=True)


pred = model.predict(df_test)
pred = np.array(pred).ravel()

pred = np.round(pred).astype(int)

submission = pd.DataFrame({"id": Id, "quality": pred})
submission.to_csv("submission.csv", index=False)



submission.head()







