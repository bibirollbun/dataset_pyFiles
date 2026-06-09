
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

train.head()



train.isnull().mean().sort_values(ascending=False)



import pandas as pd

train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

print(train.shape)
print(test.shape)




import pandas as pd

train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

print(train.shape)
print(test.shape)



from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=100)
X = vectorizer.fit_transform(train['SMILES'].fillna('')).toarray()
X_test = vectorizer.transform(test['SMILES'].fillna('')).toarray()

print("Train shape:", X.shape)
print("Test shape:", X_test.shape)



from sklearn.ensemble import RandomForestRegressor
import numpy as np

# These are the target columns
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Create a copy of test dataframe to store predictions
preds = test[['id']].copy()

# For each target, train and predict
for target in targets:
    print(f"Training for: {target}")
    
    # Drop rows where target is NaN
    notnull_idx = train[target].notnull()
    
    # Select features and target
    X_train_target = X[notnull_idx]
    y_train_target = train.loc[notnull_idx, target]
    
    # Define and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_target, y_train_target)
    
    # Predict on test set
    preds[target] = model.predict(X_test)



preds.head()



preds.to_csv("submission.csv", index=False)



import os
print("submission.csv" in os.listdir())


