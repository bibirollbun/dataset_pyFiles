import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_t = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df.head(10)


df_t.head()


df.dtypes


cat_cols = df.select_dtypes(include=['object']).columns
print('Categorical columns:', cat_cols.tolist())

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
print('Numerical columns:', num_cols.tolist())


df.isnull().sum()


for col in df.columns:
    print(f'unique values of column {col}:')
    print(f'unique values number is {len(df[col].unique())}')
    print(df[col].unique())
    print('-------------------------------')


plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols.difference(['id'])):
    plt.subplot(2, (len(num_cols)+1)//2, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()


def detect_outliers_iqr(df, columns):
    outlier_indices = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices[col] = outliers.tolist()
        print(f'{col}: {len(outliers)} outliers')
    return outlier_indices


# List of numerical columns (excluding id column)
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['id'])

print('Outliers in Data')
outliers_before = detect_outliers_iqr(df, num_cols)


for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    elif df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())


df.isnull().sum().sum()


le = LabelEncoder()


for col in cat_cols:
    df[col] = le.fit_transform(df[col])


df.dtypes


# for col in num_cols:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
#     df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# print('\nOutliers after handling:')
# outliers_after = detect_outliers_iqr(df, num_cols)


df.drop(columns=['id'], inplace=True)


corr = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Show correlation with target
corr_target = corr['Personality'].sort_values(ascending=False)
print('Correlation with target (Personality):')
print(corr_target)


X = df.drop(['Personality'], axis=1)
y = df['Personality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestClassifier(random_state=42)


rf.fit(X_train, y_train)


importances = rf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_importances.plot(kind='bar')
plt.title('Random Forest Feature Importances')
plt.show()

print('Top features:')
print(feat_importances)


top_n = 4
top_features = feat_importances.head(top_n).index.tolist()
top_features


# Select top features for training and testing
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Retrain the model
rf_selected = RandomForestClassifier(random_state=42)
rf_selected.fit(X_train_selected, y_train)

# Predict on validation set
rf_pred = rf_selected.predict(X_test_selected)

# Calculate validation accuracy
val_accuracy = accuracy_score(y_test, rf_pred)
print(f"Validation Accuracy using top {top_n} features: {val_accuracy:.4f}")


for col in df_t.columns:
    if df_t[col].dtype == 'object':
        df_t[col] = df_t[col].fillna(df_t[col].mode()[0])
    elif df_t[col].dtype in ['float64', 'int64']:
        df_t[col] = df_t[col].fillna(df_t[col].median())


for col in cat_cols.difference(['Personality']):
    le = LabelEncoder()
    le.fit(df[col])
    df_t[col] = df_t[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)


# Select only the top features for prediction
X_test_final = df_t[top_features]

# Predict using the model trained on top features
test_preds = rf_selected.predict(X_test_final)

# Prepare submission DataFrame
submission = pd.DataFrame({
    'id': df_t['id'],
    'Personality': test_preds
})

le_personality = LabelEncoder()
le_personality.fit(df['Personality'])

# Convert predictions back to original labels
submission['Personality'] = submission['Personality'].map({0: 'Extrovert', 1: 'Introvert'})

# Save to CSV
submission.to_csv('submission.csv', index=False)

