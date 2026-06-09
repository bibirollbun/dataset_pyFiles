import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train_df.head()


train_df.info()


train_df.describe()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Bin wind direction to compass points
def direction_bin(degrees):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    idx = int(((degrees + 22.5) % 360) // 45)
    return dirs[idx]

train_df['wind_bin'] = train_df['winddirection'].apply(direction_bin)

# Set plotting style
sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Feature Distributions by Rainfall', fontsize=16, y=1.02)

# Plot 1 - Pressure
sns.kdeplot(data=train_df, x='pressure', hue='rainfall', fill=True, ax=axes[0, 0], palette='coolwarm', common_norm=False)
axes[0, 0].set_title('Pressure')

# Plot 2 - Dewpoint
sns.kdeplot(data=train_df, x='dewpoint', hue='rainfall', fill=True, ax=axes[0, 1], palette='coolwarm', common_norm=False)
axes[0, 1].set_title('Dew Point')

# Plot 3 - Humidity
sns.kdeplot(data=train_df, x='humidity', hue='rainfall', fill=True, ax=axes[1, 0], palette='coolwarm', common_norm=False)
axes[1, 0].set_title('Humidity')

# Plot 4 - Cloud
sns.kdeplot(data=train_df, x='cloud', hue='rainfall', fill=True, ax=axes[1, 1], palette='coolwarm', common_norm=False)
axes[1, 1].set_title('Cloud Cover')

# Plot 5 - Sunshine
sns.kdeplot(data=train_df, x='sunshine', hue='rainfall', fill=True, ax=axes[2, 0], palette='coolwarm', common_norm=False)
axes[2, 0].set_title('Sunshine Hours')

# Plot 6 - Wind Direction (binned countplot)
sns.countplot(data=train_df, x='wind_bin', hue='rainfall', ax=axes[2, 1], palette='coolwarm')
axes[2, 1].set_title('Wind Direction (Binned)')

for ax in axes.flat: # remove labels
    ax.set_xlabel('')
    ax.set_ylabel('')

fig.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()


print(train_df.duplicated().sum())
print(train_df[train_df.duplicated()])


num_features = [col for col in train_df.select_dtypes(include=['int64', 'float64']).columns if col not in ['rainfall', 'id', 'day']]

n_cols = 3
n_rows = (len(num_features) + n_cols - 1) // n_cols

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.boxplot(y=train_df[col], ax=axes[i], color='skyblue')
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_xlabel('')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

outliers = {col: detect_outliers_iqr(train_df, col) for col in num_features}

for col, outlier_df in outliers.items():
    print(f"ğŸ›‘ Found {len(outlier_df)} outliers in {col}")



train_df.shape


from scipy.spatial import KDTree

num_features = [col for col in train_df.select_dtypes(include=['int64', 'float64']).columns if col not in ['rainfall', 'id', 'day']]

def replace_outliers_by_nearest_value(df, target_column, feature_columns):
    """Ğ—Ğ°Ğ¼Ñ–Ğ½Ñ�Ñ” Ğ»Ğ¸ÑˆĞµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ½Ñ� Ğ²Ğ¸ĞºĞ¸Ğ´Ñƒ Ñƒ target_column Ğ½Ğ° Ğ½Ğ°Ğ¹Ğ±Ğ»Ğ¸Ğ¶Ñ‡Ğµ ĞºĞ¾Ñ€ĞµĞºÑ‚Ğ½Ğµ"""
    Q1 = df[target_column].quantile(0.25)
    Q3 = df[target_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    valid_data = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]
    outliers_idx = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)].index

    before_replacement = df.loc[outliers_idx, [target_column] + feature_columns].copy()

    tree = KDTree(valid_data[feature_columns])

    for idx in outliers_idx:
        row = df.loc[idx, feature_columns]
        _, nearest_idx = tree.query(row)
        closest_valid_value = valid_data.iloc[nearest_idx][target_column]

        df.at[idx, target_column] = closest_valid_value

    after_replacement = df.loc[outliers_idx, [target_column] + feature_columns].copy()

    comparison = before_replacement.merge(after_replacement, left_index=True, right_index=True, suffixes=('_before', '_after'))

    print(f"ğŸ”„ Ğ—Ğ°Ğ¼Ñ–Ğ½ĞµĞ½Ğ¾ {len(outliers_idx)} Ğ²Ğ¸ĞºĞ¸Ğ´Ñ–Ğ² Ñƒ {target_column} Ğ½Ğ° Ğ½Ğ°Ğ¹Ğ±Ğ»Ğ¸Ğ¶Ñ‡Ñ– ĞºĞ¾Ñ€ĞµĞºÑ‚Ğ½Ñ– Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ½Ñ�")
    return df, comparison


comparison_dfs = []

for col in num_features:
    print(f"â–¶ Ğ�Ğ±Ñ€Ğ¾Ğ±ĞºĞ° {col}")
    train_df, comparison_df = replace_outliers_by_nearest_value(train_df, col, num_features)

    if not comparison_df.empty:
        cols_to_keep = [column for column in comparison_df.columns if col in column]
        comparison_dfs.append(comparison_df[cols_to_keep])



comparison_dfs[0]


train_df.head()


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

outliers = {col: detect_outliers_iqr(train_df, col) for col in num_features}

for col, outlier_df in outliers.items():
    print(f"ğŸ›‘ Found {len(outlier_df)} outliers in {col}")



def classify_wind_origin(degrees):
    if 22.5 <= degrees < 67.5 or 67.5 <= degrees < 112.5 or 202.5 <= degrees < 247.5:
        return 'moist'
    elif 112.5 <= degrees < 202.5:
        return 'neutral'
    else:
        return 'dry'

train_df_classes = train_df.copy()
train_df_classes['wind_class'] = train_df['winddirection'].apply(classify_wind_origin)


mapping = {'dry': 0, 'neutral': 1, 'moist': 2}
train_df_classes['wind_class_encoded'] = train_df_classes['wind_class'].map(mapping)
train_df_classes.head()


X = train_df_classes.drop(columns=['rainfall','id', 'day', 'winddirection', 'wind_bin', 'wind_class'])  # features
y = train_df_classes['rainfall']                 # target


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_proba))



X2 = train_df_classes.drop(columns=['rainfall','id', 'day', 'wind_bin', 'wind_class_encoded', 'wind_class'])  # features
y2 = train_df_classes['rainfall'] # target


X_train2, X_val2, y_train2, y_val2 = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train2, y_train2)

y_pred2 = model2.predict(X_val2)
y_proba2 = model2.predict_proba(X_val2)[:, 1]

print(classification_report(y_val2, y_pred2))
print("ROC-AUC:", roc_auc_score(y_val2, y_proba2))



df_corr = train_df_classes.copy()
df_corr = df_corr.drop(columns=['id', 'wind_bin', 'wind_class'])

corr_matrix = df_corr.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Feature Correlation Map", fontsize=16)
plt.tight_layout()
plt.show()




X4 = train_df_classes.drop(columns=['rainfall','id', 'day', 'maxtemp', 'mintemp', 'wind_bin', 'winddirection', 'wind_class'])  # features
y4 = train_df_classes['rainfall'] # target

X_train4, X_val4, y_train4, y_val4 = train_test_split(X4, y4, test_size=0.2, random_state=42, stratify=y4)

model4 = RandomForestClassifier(n_estimators=100, random_state=42)
model4.fit(X_train4, y_train4)

y_pred4 = model4.predict(X_val4)
y_proba4 = model4.predict_proba(X_val4)[:, 1]

print(classification_report(y_val4, y_pred4))
print("ROC-AUC:", roc_auc_score(y_val4, y_proba4))



from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
min_max_scaler = MinMaxScaler()

num_features = [col for col in train_df_classes.select_dtypes(include=['int64', 'float64']).columns if col not in ['rainfall', 'id', 'day']]

train_df_classes[num_features] = min_max_scaler.fit_transform(train_df_classes[num_features])



train_df_classes["rainfall"].value_counts()


from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler(sampling_strategy=1.0, random_state=42)  # 1.0 means equal balance

X3 = train_df_classes.drop(columns=['rainfall','id', 'day', 'maxtemp', 'mintemp', 'wind_bin', 'wind_class_encoded', 'wind_class'])  # features
y3 = train_df_classes['rainfall'] # target

X_under, y_under = undersampler.fit_resample(X3, y3)
print("After Undersampling:")
print(y_under.value_counts())

X_train3, X_val3, y_train3, y_val3 = train_test_split(X_under, y_under, test_size=0.2, random_state=42, stratify=y_under)

model3 = RandomForestClassifier(n_estimators=100, random_state=42)
model3.fit(X_train3, y_train3)

y_pred3 = model3.predict(X_val3)
y_proba3 = model3.predict_proba(X_val3)[:, 1]

print(classification_report(y_val3, y_pred3))
print("ROC-AUC:", roc_auc_score(y_val3, y_proba3))



model3.feature_importances_


importances = pd.Series(model3.feature_importances_, index=X3.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



from sklearn.model_selection import KFold
import numpy as np

# Set up 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create a list to store scores and splits
split_results = []

for train_index, val_index in kf.split(X3):
    X_train_fold, X_val_fold = X3.iloc[train_index], X3.iloc[val_index]
    y_train_fold, y_val_fold = y3.iloc[train_index], y3.iloc[val_index]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_fold, y_train_fold)

    y_val_proba = model.predict_proba(X_val_fold)[:, 1]
    auc = roc_auc_score(y_val_fold, y_val_proba)

    split_results.append({
        'train_idx': train_index,
        'val_idx': val_index,
        'auc': auc
    })

auc_results = [split_results[idx]['auc'] for idx in range(5)]
best_model_idx = auc_results.index(max(auc_results))
print(best_model_idx)
print("Best fold ROC-AUC:", split_results[best_model_idx]['auc'])

best_model = split_results[best_model_idx]
best_model


from sklearn.model_selection import PredefinedSplit

# Create split mask: -1 = training, 0 = validation
split_index = np.full(len(X3), -1)
split_index[best_model['val_idx']] = 0  # mark validation set

# Create the predefined split
predefined_split = PredefinedSplit(test_fold=split_index)
predefined_split


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=predefined_split,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X3, y3)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best ROC-AUC Score (CV avg):", grid_search.best_score_)



train_df_classes


X_test = train_df_classes.drop(columns=['id', 'day', 'maxtemp', 'mintemp'])
X_test_filled = X_test.fillna(X_test.mean(numeric_only=True))



rainfall_probs = best_model.predict_proba(X_test_filled)[:, 1]


submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': rainfall_probs
})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")




