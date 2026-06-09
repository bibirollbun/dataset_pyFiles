import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')



train.shape


test.shape


train.info()


train.head()


train.isnull().sum()


import numpy as np

# Find columns with non‑finite numbers
mask_nonfinite = ~np.isfinite(train.select_dtypes(include='number'))
cols_with_nonfinite = mask_nonfinite.any()

print("Columns with NaN/Inf:", cols_with_nonfinite[cols_with_nonfinite].index.tolist())

# Count how many in each column
print(mask_nonfinite.sum())


train.dtypes


train['Personality'].value_counts(normalize=True)


train.hist(figsize=(12, 8))


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(train.corr(numeric_only=True), annot=True, cmap='coolwarm')


from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Features and target
X = train.drop(columns=['id', 'Personality'])
y = train['Personality']

label_encoders = {}

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le  # ✅ Save encoder for later
# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# LightGBM model using scikit-learn API
model = LGBMClassifier(
    objective='multiclass',
    num_class=y.nunique(),     # Number of classes
    n_estimators=100,
    early_stopping_rounds=10,
    verbose=1

)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    
)

# Predict and evaluate
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))



from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))


import lightgbm as lgb
import matplotlib.pyplot as plt

lgb.plot_importance(model, max_num_features=10)
plt.show()


sns.countplot(x='Stage_fear', hue='Personality', data=train)
plt.title("Stage Fear by Personality")
plt.show()


prop_df = (
    train.groupby(['Stage_fear', 'Personality'])
         .size()
         .reset_index(name='count')
)

# Normalize counts within each Stage_fear group
prop_df['proportion'] = prop_df.groupby('Stage_fear')['count'].transform(lambda x: x / x.sum())

sns.barplot(x='Stage_fear', y='proportion', hue='Personality', data=prop_df)
plt.title("Proportion of Personality Types by Stage Fear")
plt.show()


pip install shap


import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

shap.summary_plot(shap_values, X_val)


    sns.boxplot(x='Personality', y='Time_spent_Alone', data=train)


sns.countplot(x='Stage_fear', hue='Personality', data=train)


test.head()


X_test = test.drop(columns=['id'])

for col in X_test.columns:
    if col in label_encoders:
        X_test[col] = label_encoders[col].transform(X_test[col].astype(str))



y_test_preds = model.predict(X_test)


# Fill submission file
submission = sample_submission.copy()
submission['Personality'] = y_test_preds

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("✅ Submission file created: submission.csv")





