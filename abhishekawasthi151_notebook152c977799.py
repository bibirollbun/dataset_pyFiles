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


import os
from pathlib import Path

DATA_DIR = Path("/kaggle/input/messenger-leakage-dataset")

for root, dirs, files in os.walk(DATA_DIR):
    print(root)
    for f in files[:3]:  # show a few sample files per folder
        print("   ", f)



import pandas as pd
from pathlib import Path

# Set the correct dataset path
DATA_DIR = Path("/kaggle/input/messenger-leakage-dataset/messenger_leakage_dataset")

# Load the metadata
meta = pd.read_csv(DATA_DIR / "metadata.csv")

print(meta.head())



meta['method'].unique()


meta['method'] = meta['method'].replace({
    'paraphrased': 'llm_modified',
    'paraphrased_hidden': 'llm_modified_hidden'
})


meta['method'].unique()


texts, labels = [], []

for _, row in meta.iterrows():
    text_path = DATA_DIR / f"data/{row['method']}/{row['filename']}"
    texts.append(open(text_path, encoding='utf-8').read())
    labels.append(row['label'])


from pathlib import Path

texts, labels = [], []

for _, row in meta.iterrows():
    text_path = DATA_DIR / f"data/{row['method']}/{row['filename']}"
    if text_path.exists():
        texts.append(open(text_path, encoding='utf-8').read())
        labels.append(row['label'])
    else:
        print(f"⚠️ Missing file: {text_path}")


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)



from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


import numpy as np

tfidf = model.named_steps['tfidf']
clf = model.named_steps['clf']
feature_names = np.array(tfidf.get_feature_names_out())
coefs = clf.coef_

for i, cls in enumerate(clf.classes_):
    print(f"\nTop words for {cls}:")
    top = np.argsort(coefs[i])[-10:]
    print(feature_names[top])


import pandas as pd

# Assuming y_pred are your model predictions (0 or 1)
submission = pd.DataFrame({
    "id": range(len(y_pred)),
    "hidden_message": y_pred
})

submission.to_csv("submission.csv", index=False)



import pandas as pd

# Create submission DataFrame (you already have y_pred)
submission = pd.DataFrame({
    "id": range(len(y_pred)),
    "hidden_message": y_pred
})

# If you want at least 5 rows
while len(submission) < 5:
    submission = pd.concat([submission, submission], ignore_index=True)

submission = submission.head(5)
submission.to_csv("submission.csv", index=False)
submission.head()



!ls -lh submission.csv



import pandas as pd
pd.read_csv("submission.csv").head()



















