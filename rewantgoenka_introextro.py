# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Playground Series - S5E7: Predict Introvert or Extrovert

# ğŸ“¦ 1. Imports & Settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

# ğŸ“… 2. Data Loading & Preprocessing
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Encode binary categorical features
for df in [train, test]:
    df['Stage_fear'] = df['Stage_fear'].map({'No': 0, 'Yes': 1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No': 0, 'Yes': 1})

# Encode target
train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# Handle potential NaNs caused by incorrect operations later
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# ğŸ§� 3. Feature Engineering
features = [col for col in train.columns if col not in ['Personality']]
for df in [train, test]:
    df['row_mean'] = df[features].mean(axis=1)
    df['row_std'] = df[features].std(axis=1)
    df['social_score'] = df['Social_event_attendance'] + df['Going_outside'] + df['Post_frequency']
    df['introvert_score'] = df['Time_spent_Alone'] + df['Drained_after_socializing'] + df['Stage_fear']
    df['social_minus_intro'] = df['social_score'] - df['introvert_score']
    df['combined_engagement'] = df['Friends_circle_size'] * df['Post_frequency']

# Fill any new NaNs introduced
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# ğŸŒŸ 4. Feature Selection using Random Forest importance
X = train.drop('Personality', axis=1)
y = train['Personality']
X_test = test.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X_scaled, y)
X_selected = selector.transform(X_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()].tolist()
print("Selected features:", selected_features)

# ğŸ“Š 5. Modeling (RandomForest only with selected features)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_rf = RandomForestClassifier(n_estimators=300, random_state=42)
model_rf.fit(X_selected, y)

# Predict
predictions = model_rf.predict(X_test_selected)

# ğŸ“„ 6. Submission
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': predictions
})
submission['Personality'] = submission['Personality'].map({0: 'Introvert', 1: 'Extrovert'})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as submission.csv")

# ğŸ“Œ 7. Quick EDA Visualizations
train['Personality_Label'] = train['Personality'].map({0: 'Introvert', 1: 'Extrovert'})

# Countplot
sns.countplot(x='Personality_Label', data=train)
plt.title('Class Distribution')
plt.savefig("class_distribution.png")
plt.show()

# Correlation Heatmap
sns.heatmap(train.drop(['Personality', 'Personality_Label'], axis=1).corr(), annot=True, fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig("feature_correlation.png")
plt.show()

# Pairplot
sns.pairplot(train[selected_features + ['Personality_Label']], hue='Personality_Label')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.savefig("pairplot.png")
plt.show()

# Violin Plots for Selected Features
for feature in selected_features:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='Personality_Label', y=feature, data=train)
    plt.title(f'Violin Plot: {feature} by Personality')
    plt.savefig(f"violin_{feature}.png")
    plt.show()


