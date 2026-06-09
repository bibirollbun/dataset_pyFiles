# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# تحميل البيانات
train = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')
test = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')

# عرض أول 5 صفوف للتأكد
train.head()


!pip install -q transformers


from transformers import BertTokenizer, AutoModel

model_path = "/kaggle/input/bert-base-uncased-local/bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

print("✅ تم تحميل التوكنايزر والمودل بنجاح!")


import torch
from tqdm import tqdm

# تأكد إن المودل في وضع التقييم (inference)
model.eval()

# لو عندك GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# نجهز الأسئلة
questions = train["Question"].tolist()

# هنا نحفظ الـ embeddings
embeddings = []

# تحويل كل سؤال إلى embedding
with torch.no_grad():
    for question in tqdm(questions):
        # Tokenization
        encoded_input = tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        # مرري السؤال على المودل
        output = model(**encoded_input)

        # نأخذ الـ CLS token (أول token)
        cls_embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings.append(cls_embedding)


import numpy as np

X = np.array(embeddings)
y = train["label"].values

print("✅ شكل X:", X.shape)
print("✅ شكل y:", y.shape)


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))


test_questions = test["Question"].tolist()


test_embeddings = []

model.eval()
with torch.no_grad():
    for question in tqdm(test_questions):
        encoded_input = tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        output = model(**encoded_input)
        cls_embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        test_embeddings.append(cls_embedding)


import numpy as np
X_test = np.array(test_embeddings)


test_preds = clf.predict(X_test)


submission = pd.DataFrame({
    "id": test.index,
    "label": test_preds
})

submission.to_csv("submission.csv", index=False)
print("✅ تم إنشاء ملف submission.csv!")

