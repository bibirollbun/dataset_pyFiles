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

# Load main data
app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
prev_app = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')



# ================================
# Loan Default Risk - Starter Code
# ================================

# ğŸ“¦ Import Libraries
import pandas as pd
import numpy as np

# ğŸ“� Load Datasets
app = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
prev = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')

print("âœ… Data loaded successfully")

# ğŸ‘€ Quick Glimpse
print("Main application data shape:", app.shape)
print("Bureau data shape:", bureau.shape)
print("Previous application data shape:", prev.shape)

# ========================
# ğŸ“Š Basic Feature Engineering
# ========================

# â�• Add debt-to-income ratio
app['DEBT_TO_INCOME'] = app['AMT_CREDIT'] / (app['AMT_INCOME_TOTAL'] + 1)

# â�• Add credit utilization (credit to goods price ratio)
app['CREDIT_GOODS_RATIO'] = app['AMT_CREDIT'] / (app['AMT_GOODS_PRICE'] + 1)

# ğŸ§  Binary: Has a phone and email?
app['HAS_PHONE'] = app['FLAG_MOBIL']
app['HAS_EMAIL'] = app['FLAG_EMAIL']

# ========================
# ğŸ”— Merge with Bureau Data
# ========================

# ğŸ§® Aggregate bureau info per customer
bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'AMT_CREDIT_SUM': ['sum', 'mean'],
    'CREDIT_DAY_OVERDUE': ['max', 'mean'],
    'SK_ID_BUREAU': 'count'
})
bureau_agg.columns = ['BUREAU_CREDIT_SUM_SUM', 'BUREAU_CREDIT_SUM_MEAN', 
                      'BUREAU_OVERDUE_MAX', 'BUREAU_OVERDUE_MEAN', 'NUM_PAST_LOANS']
bureau_agg.reset_index(inplace=True)

# ğŸ§¬ Merge with main app data
app = app.merge(bureau_agg, on='SK_ID_CURR', how='left')

# ========================
# ğŸ”— Merge with Previous Application Data
# ========================

# ğŸ§® Aggregate previous application info
prev_agg = prev.groupby('SK_ID_CURR').agg({
    'AMT_CREDIT': ['mean', 'max'],
    'NAME_CONTRACT_STATUS': lambda x: (x == 'Approved').mean()
})
prev_agg.columns = ['PREV_CREDIT_MEAN', 'PREV_CREDIT_MAX', 'APPROVAL_RATE']
prev_agg.reset_index(inplace=True)

# ğŸ§¬ Merge with app data
app = app.merge(prev_agg, on='SK_ID_CURR', how='left')

print("âœ… Feature merging complete. Final shape:", app.shape)

# ========================
# ğŸ§¼ Handle Missing Values (Optional Start)
# ========================
app.fillna(0, inplace=True)

# ========================
# ğŸ�¯ Ready for Modeling
# ========================
# Separate target
target = app['TARGET']
features = app.drop(columns=['SK_ID_CURR', 'TARGET'])

print("âœ… Data is now ready for modeling!")



# ================================
# ğŸ›  1. Import Libraries
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier, plot_importance

# ================================
# ğŸ“¥ 2. Load Data
# ================================
app = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
print("âœ… Loaded data. Shape:", app.shape)

# ================================
# ğŸ§¹ 3. Preprocess Data
# ================================
# Drop ID and target columns from features
df = app.drop(columns=['SK_ID_CURR', 'TARGET'])

# Convert categorical features (object) to dummy/one-hot encoded columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Convert all features to float32 to save memory
df_encoded = df_encoded.astype('float32')

# Define target
target = app['TARGET']

# ================================
# âœ‚ï¸� 4. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded, target, test_size=0.2, random_state=42
)

print("âœ… Training shape:", X_train.shape)

# ================================
# ğŸš€ 5. Train XGBoost Classifier
# ================================
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("âœ… Model training complete.")

# ================================
# ğŸ“ˆ 6. Evaluate the Model
# ================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("ğŸ”� Accuracy:", accuracy_score(y_test, y_pred))
print("ğŸ�† ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("\nğŸ§¾ Classification Report:\n", classification_report(y_test, y_pred))

# ================================
# ğŸ”� 7. Confusion Matrix
# ================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================
# ğŸŒŸ 8. Feature Importance Plot
# ================================
plt.figure(figsize=(10,6))
plot_importance(model, max_num_features=15, height=0.5)
plt.title('Top 15 Feature Importances')
plt.show()



# ğŸ”„ Create a DataFrame with predictions and probabilities
results = app[['SK_ID_CURR']].copy()
results = results.iloc[y_test.index]  # Align with test set
results['Actual'] = y_test.values
results['Predicted'] = y_pred
results['Probability'] = y_proba

# ğŸ’¾ Export to CSV
results.to_csv("loan_default_predictions.csv", index=False)
print("ğŸ“� Saved predictions to loan_default_predictions.csv")


