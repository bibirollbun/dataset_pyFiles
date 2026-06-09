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


import matplotlib as pyplot

df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df.head(5)


df.isnull().sum(axis=0)


df.drop("id",axis=1,inplace=True)
df_test.drop("id",axis=1,inplace=True)


df.head(5)


import matplotlib.pyplot as plt
viz_col=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
num_plots=len(viz_col)
fig,axes=plt.subplots(nrows=1, ncols=num_plots, figsize=(4 * num_plots, 4), sharey=True)
for idx,col_name in enumerate(viz_col):
    ax=axes[idx]
    df[col_name].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
    ax.set_title(col_name)
    ax.set_xlabel("value")
    if idx==0:
        ax.set_ylabel("frequency")

plt.show()
    


df[viz_col]=df[viz_col].fillna(df[viz_col].median())
df_test[viz_col]=df_test[viz_col].fillna(df[viz_col].median())
for col_name in viz_col:
    col_median=df[col_name].median()
    print(col_median)


other_col=['Stage_fear','Drained_after_socializing']
df[other_col]=df[other_col].fillna('Unknown')
df_test[other_col]=df_test[other_col].fillna('Unknown')


dummies=pd.get_dummies(df['Personality'],drop_first=True)
df=pd.concat([df,dummies],axis=1)
df=df.drop('Personality',axis=1)
df.head(5)
df.dtypes






from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
train,valid=train_test_split(df, test_size=0.2, random_state=42)

TARGET_COL='Introvert'
X_train = train.drop(TARGET_COL, axis=1)
y_train = train[TARGET_COL]

X_valid = valid.drop(TARGET_COL, axis=1)
y_valid = valid[TARGET_COL]
categorical_features_indices=X_train.select_dtypes(include=['object', 'category','bool']).columns.tolist()
model = CatBoostClassifier(
    iterations=100,           # Number of trees/rounds
    learning_rate=0.1,        # How fast the model learns
    depth=6,                  # Depth of trees
    loss_function='Logloss',  # Standard loss for binary classification
    eval_metric='Accuracy',   # Metric to watch during training
    random_seed=42,
    verbose=5                 # Set to 100 to see training logs every 100 iterations
)

model.fit(
    X_train, 
    y_train,
    cat_features=categorical_features_indices, 
    eval_set=(X_valid,y_valid),
    plot=True # Set to True to see an interactive training graph in notebooks
)


from sklearn.metrics import accuracy_score, classification_report
pred=model.predict(X_valid)
acc=accuracy_score(y_valid,pred)
print(acc)
print(classification_report(y_valid, pred))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_valid, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


predictions=model.predict(df_test)
print(predictions)

personality_labels = np.where(predictions == False, 'Extrovert', 'Introvert')
num_test_samples = len(df_test)
start_id = 18524
submission_ids = np.arange(start_id, start_id + num_test_samples)


submission_df = pd.DataFrame({
    'id': submission_ids,
    'Personality': personality_labels  
})

# 5. Verify and Export
print("\n--- Submission DataFrame Head ---")
print(submission_df.head())

print("\n--- Submission DataFrame Shape ---")
print(submission_df.shape)

# Export to CSV
submission_df.to_csv('submission.csv', index=False)

