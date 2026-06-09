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


# Basic libraries
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')


# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder


# Define the dataset directions
train_file='/kaggle/input/playground-series-s5e7/train.csv'
test_file = '/kaggle/input/playground-series-s5e7/test.csv'

# Read the dataset
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
display(HTML("<span style = 'color: blue; font-weight:bold;'> Dataset reading be completed</span>"))




# Basic understanding of Train dataset
print("-----"*10 + "Overview Train Dataset" + "------"*10)
display(HTML("<span style = 'color: blue; font-weight:bold;'> Train dataset\'s Information</span>"))
# Print Top 5 samples
print(train_data.info())
display(HTML("<span style = 'color: blue; font-weight:bold;'> Train dataset\'s Top 5 rows</span>"))
display(train_data.head())
# Train's description understanding
display(HTML("<span style = 'color: blue; font-weight:bold;'> Train dataset\'s Description</span>"))
display(train_data.describe())
# Train's Null/ NaN checking
display(HTML("<span style = 'color: blue; font-weight:bold;'> Train dataset\'s Null/ NaN checking</span>"))
# Null/ NaN quatity and percentage calculation
null_counts = train_data.isnull().sum()
null_percent = (null_counts / len(train_data)) * 100
# Create Dataframe for display
null_summary = pd.DataFrame({
    'Missing Count': null_counts,
    'Missing %': null_percent.round(2)
})
display(null_summary)
# Duplicate checking
display(HTML("<span style = 'color: blue; font-weight:bold;'> Train dataset\'s duplicated checking</span>"))
print(f"Number of Train's duplicated rows: {train_data.duplicated().sum()}")
# Unique values checking
display(HTML("<span style = 'color: blue; font-weight:bold;'> Train dataset\'s unique values checking</span>"))
print(train_data.nunique())

# Basic understanding of Test dataset
print("-----"*10 + "Overview Test Dataset" + "------"*10)
display(HTML("<span style = 'color: red; font-weight:bold;'> Test dataset\'s Information</span>"))
print(test_data.info())
# Print Top 5 samples
display(HTML("<span style = 'color: red; font-weight:bold;'> Test dataset\'s Top 5 rows</span>"))
display(test_data.head())
# Test's description understanding
display(HTML("<span style = 'color: red; font-weight:bold;'> Test dataset\'s Description</span>"))
display(test_data.describe())
# Test's Null/ NaN checking
# Null/ NaN quatity and percentage calculation
display(HTML("<span style = 'color: red; font-weight:bold;'> Test dataset\'s Null/ NaN checking</span>"))
# Null/ NaN quatity and percentage calculation
null_counts = test_data.isnull().sum()
null_percent = (null_counts / len(test_data)) * 100
# Create Dataframe for display
null_summary = pd.DataFrame({
    'Missing Count': null_counts,
    'Missing %': null_percent.round(2)
})
display(null_summary)
# Duplicate checking
display(HTML("<span style = 'color: red; font-weight:bold;'> Test dataset\'s duplicated checking</span>"))
print(f"Number of Test's duplicated rows: {test_data.duplicated().sum()}")
display(HTML("<span style = 'color: red; font-weight:bold;'> Test dataset\'s unique values checking</span>"))
print(test_data.nunique())


# Define all needed variables
id_col = 'id'
target_col = 'Personality'
all_features = [col for col in train_data.columns.tolist() if col != id_col and col != target_col] 
print(all_features)


# Data preparation
train_df = train_data.copy()
test_ids = test_data[id_col].copy()
test_df = test_data.copy()
# ----------------------------
# 1. Chuáº©n hÃ³a Yes/No => 1/0
# ----------------------------
for col in ['Stage_fear', 'Drained_after_socializing']:
    train_df[col] = train_df[col].map({'Yes': 1, 'No': 0})
    test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})

# ----------------------------
# 2. Gá»™p train + test Ä‘á»ƒ Impute
# ----------------------------
train_features = train_df.drop(columns=['id', 'Personality']).copy()
test_features = test_df.drop(columns=['id']).copy()

combined = pd.concat([train_features, test_features], ignore_index=True)

# ----------------------------
# 2.1. Feature Engineering
# ----------------------------

# Social_activity = tá»•ng cÃ¡c Ä‘áº·c trÆ°ng liÃªn quan xÃ£ há»™i
combined['Social_activity'] = (
    combined['Stage_fear'].fillna(0) + 
    combined['Social_event_attendance'].fillna(0) +
    combined['Going_outside'].fillna(0) +
    combined['Drained_after_socializing'].fillna(0)
)

combined['Social_ratio'] = combined['Drained_after_socializing']/(combined['Going_outside'] + 1)
combined['Isolation_index'] = combined['Time_spent_Alone']/(combined['Social_event_attendance'] + 1)
combined['Fear_vs_outside_ratio'] = combined['Stage_fear']/(combined['Going_outside'] + 1)
combined['Online_social_gap'] = combined['Post_frequency']/(combined['Friends_circle_size'] + 1)
combined['Drained_ratio'] = combined['Drained_after_socializing']/(combined['Social_event_attendance'] + 1)
combined['Fear_ratio'] = combined['Stage_fear']/(combined['Social_activity'] + 1)
combined['Alone_to_social'] = combined['Time_spent_Alone']/(combined['Social_activity'] + 1)
combined['Interaction_terms'] = combined['Stage_fear'] * combined['Going_outside'] 

# Binning Time_spent_Alone 
combined['Time_spent_Alone_bin'] = pd.cut(
    combined['Time_spent_Alone'],
    bins=[-1, 2, 5, 10, np.inf],
    labels=[0, 1, 2, 3]
).astype(float)

# ----------------------------
# 3. Chuáº©n hÃ³a trÆ°á»›c khi Impute
# ----------------------------
scaler = StandardScaler()
combined_scaled = pd.DataFrame(scaler.fit_transform(combined), columns=combined.columns)

# ----------------------------
# 4. Impute missing báº±ng KNN
# ----------------------------
imputer = KNNImputer(n_neighbors=5)
combined_imputed = pd.DataFrame(imputer.fit_transform(combined_scaled), columns=combined.columns)

# ----------------------------
# 5. TÃ¡ch láº¡i train / test
# ----------------------------
X = combined_imputed.iloc[:len(train_df), :].copy()
test_df_cleaned = combined_imputed.iloc[len(train_df):, :].copy()

# ----------------------------
# 6. Encode target
# ----------------------------
le = LabelEncoder()
y = le.fit_transform(train_df['Personality'])
display(HTML("<span style = 'color: blue; font-weight:bold;'> Data preprocessing is completed</span>"))
print("Shape of X:", X.shape)
print("Featues of X:", X.columns.tolist())


corr_train = X.corr()
sns.heatmap(corr_train, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.title("Heatmap - Train Data")
plt.show()


kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

# Táº¡o model XGBClassifier vá»›i cÃ¡c tham sá»‘ tÆ°Æ¡ng Ä‘Æ°Æ¡ng
xgb_clf = XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    max_depth=7,
    learning_rate=0.0001,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.5,
    min_child_weight=10,
    max_bin=256,
    gamma=0,
    grow_policy='lossguide',
    random_state=42,
    eval_metric='merror',
    tree_method='gpu_hist',
    device='cuda',  # 'gpu_hist' náº¿u báº¡n dÃ¹ng GPU hiá»‡u quáº£ hÆ¡n
    use_label_encoder=False
)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nğŸŒ€ Fold {fold}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = xgb_clf.fit(
        X_train, y_train,
        early_stopping_rounds=500,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=200
    )

    # âš ï¸� Dá»± Ä‘oÃ¡n ra xÃ¡c suáº¥t, láº¥y argmax Ä‘á»ƒ chuyá»ƒn thÃ nh nhÃ£n
    preds = np.argmax(model.predict(X_val), axis=1)

    acc = accuracy_score(y_val, preds)
    print(f"âœ… Accuracy (fold {fold}): {acc:.4f}")
    accuracies.append(acc)

print(f"\nğŸ�¯ Mean CV Accuracy: {np.mean(accuracies):.4f}")






# Fit toÃ n bá»™ dá»¯ liá»‡u vÃ o model sau khi KFold xong
xgb_clf.fit(X, y)
# Dá»± Ä‘oÃ¡n xÃ¡c suáº¥t
test_preds_proba = xgb_clf.predict_proba(test_df_cleaned)

# Láº¥y nhÃ£n dá»± Ä‘oÃ¡n
test_preds = np.argmax(test_preds_proba, axis=1)

# Giáº£i mÃ£ nhÃ£n thÃ nh tÃªn ban Ä‘áº§u
final_preds_labels = le.inverse_transform(test_preds)

submission = pd.DataFrame({
    'id': test_ids,
    'Personality': final_preds_labels
})
submission.to_csv('submission.csv', index=False)

display(HTML("<span style = 'color: blue; font-weight:bold;'> Submission file 'submission.csv' was saved</span>"))



fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(xgb_clf,  ax=ax)
ax.set_title('Feature Importance')
plt.tight_layout()
plt.show()

