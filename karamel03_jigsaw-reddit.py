import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

print("Eksik değer sayısı:\n", train.isnull().sum())


vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train = vectorizer.fit_transform(train["body"].astype(str))
X_test = vectorizer.transform(test["body"].astype(str))

y_train = train["rule_violation"].astype(int)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred_train = model.predict(X_train)


cm = confusion_matrix(y_train, y_pred_train)
disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Eğitim Verisi - Confusion Matrix")
plt.show()


print("\nClassification Report (Eğitim Verisi):\n")
print(classification_report(y_train, y_pred_train))


y_prob_train = model.predict_proba(X_train)[:,1]
sns.histplot(y_prob_train, bins=50)
plt.title("Pozitif Tahmin Olasılıkları (Eğitim Verisi)")
plt.xlabel("Olasılık")
plt.ylabel("Frekans")
plt.show()



test_pred = model.predict(X_test)


submission = pd.DataFrame({
    "id": test.index,  # test["id"] yerine
    "rule_violation": test_pred
})
submission.to_csv("submission.csv", index=False)

