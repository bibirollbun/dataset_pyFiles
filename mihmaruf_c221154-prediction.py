# %% [code] {"execution":{"iopub.status.busy":"2025-08-08T03:16:52.418761Z","iopub.execute_input":"2025-08-08T03:16:52.419092Z","iopub.status.idle":"2025-08-08T03:16:56.529365Z","shell.execute_reply.started":"2025-08-08T03:16:52.419048Z","shell.execute_reply":"2025-08-08T03:16:56.528281Z"},"jupyter":{"outputs_hidden":false}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,r2_score
from xgboost import XGBClassifier
%matplotlib inline
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.preprocessing import LabelEncoder

# %% [code] {"execution":{"iopub.status.busy":"2025-08-08T03:16:56.530406Z","iopub.execute_input":"2025-08-08T03:16:56.531006Z","iopub.status.idle":"2025-08-08T03:16:56.793700Z","shell.execute_reply.started":"2025-08-08T03:16:56.530971Z","shell.execute_reply":"2025-08-08T03:16:56.792827Z"},"jupyter":{"outputs_hidden":false}}
# Load both train and test datasets
train_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")
test_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv")

# %% [code] {"execution":{"iopub.status.busy":"2025-08-08T03:16:56.796101Z","iopub.execute_input":"2025-08-08T03:16:56.796473Z","iopub.status.idle":"2025-08-08T03:16:56.847210Z","shell.execute_reply.started":"2025-08-08T03:16:56.796439Z","shell.execute_reply":"2025-08-08T03:16:56.846023Z"},"jupyter":{"outputs_hidden":false}}
train_df.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-08T03:16:56.848539Z","iopub.execute_input":"2025-08-08T03:16:56.848916Z","iopub.status.idle":"2025-08-08T03:16:56.888557Z","shell.execute_reply.started":"2025-08-08T03:16:56.848878Z","shell.execute_reply":"2025-08-08T03:16:56.887493Z"},"jupyter":{"outputs_hidden":false}}
train_df.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2025-08-08T03:16:56.889681Z","iopub.execute_input":"2025-08-08T03:16:56.890147Z","iopub.status.idle":"2025-08-08T03:16:56.906991Z","shell.execute_reply.started":"2025-08-08T03:16:56.890110Z","shell.execute_reply":"2025-08-08T03:16:56.905918Z"},"jupyter":{"outputs_hidden":false}}
train_df['Arrival Delay in Minutes'] = train_df['Arrival Delay in Minutes'].fillna(train_df['Arrival Delay in Minutes'].median())

# %% [code] {"execution":{"iopub.status.busy":"2025-08-08T03:16:56.908097Z","iopub.execute_input":"2025-08-08T03:16:56.908583Z","iopub.status.idle":"2025-08-08T03:16:56.948074Z","shell.execute_reply.started":"2025-08-08T03:16:56.908498Z","shell.execute_reply":"2025-08-08T03:16:56.946913Z"},"jupyter":{"outputs_hidden":false}}
train_df.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2025-08-08T03:16:56.949157Z","iopub.execute_input":"2025-08-08T03:16:56.949528Z","iopub.status.idle":"2025-08-08T03:16:57.026048Z","shell.execute_reply.started":"2025-08-08T03:16:56.949492Z","shell.execute_reply":"2025-08-08T03:16:57.025078Z"},"jupyter":{"outputs_hidden":false}}
# Label encode categorical columns for both train and test
label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

le = LabelEncoder()
for col in label_cols:
    if col in train_df.columns:
        train_df[col] = le.fit_transform(train_df[col])
    if col in test_df.columns and col != 'satisfaction':  # test doesn't have satisfaction column
        test_df[col] = le.transform(test_df[col])

# %% [code] {"execution":{"iopub.status.busy":"2025-08-08T03:16:57.027014Z","iopub.execute_input":"2025-08-08T03:16:57.027369Z","iopub.status.idle":"2025-08-08T03:16:57.037288Z","shell.execute_reply.started":"2025-08-08T03:16:57.027297Z","shell.execute_reply":"2025-08-08T03:16:57.036294Z"},"jupyter":{"outputs_hidden":false}}
X = train_df.drop(columns=['satisfaction', 'id'], errors='ignore')
y = train_df['satisfaction']

# %% [code] {"execution":{"iopub.status.busy":"2025-08-08T03:16:57.041198Z","iopub.execute_input":"2025-08-08T03:16:57.041660Z","iopub.status.idle":"2025-08-08T03:16:57.071499Z","shell.execute_reply.started":"2025-08-08T03:16:57.041590Z","shell.execute_reply":"2025-08-08T03:16:57.070413Z"},"jupyter":{"outputs_hidden":false}}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-08T03:16:57.072544Z","iopub.execute_input":"2025-08-08T03:16:57.072824Z","iopub.status.idle":"2025-08-08T03:17:08.681936Z","shell.execute_reply.started":"2025-08-08T03:16:57.072803Z","shell.execute_reply":"2025-08-08T03:17:08.681004Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

gboost = GradientBoostingClassifier()
gboost.fit(X_train, y_train)

y_pred = gboost.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='satisfaction', data=train_df)
plt.show()


from sklearn.ensemble import GradientBoostingClassifier
gboost = GradientBoostingClassifier()
gboost.fit(X_train, y_train)



from sklearn.metrics import classification_report, accuracy_score
y_pred = gboost.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import numpy as np

importances = gboost.feature_importances_
indices = np.argsort(importances)
features = X_train.columns

plt.figure(figsize=(10,6))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


import shap

explainer = shap.Explainer(gboost, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)



from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = gboost.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



# %% [code]
# Check for missing values in test data
print("Missing values in test data before processing:")
print(test_df.isnull().sum())

# Fill missing values in test data (using the same approach as training)
test_df['Arrival Delay in Minutes'] = test_df['Arrival Delay in Minutes'].fillna(test_df['Arrival Delay in Minutes'].median())

# Verify no more missing values
#print("\nMissing values in test data after processing:")
#print(test_df.isnull().sum())

# Prepare test data for prediction
X_test_final = test_df.drop(columns=['id'], errors='ignore')

# Make predictions on test data
test_preds = gboost.predict(X_test_final)

# Create submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_df['id'],
    'satisfaction': le.inverse_transform(test_preds)  # Convert back to original labels
})

# Save submission file
submission_df.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")

# Display first few rows
print(submission_df.head())

