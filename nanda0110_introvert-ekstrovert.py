!pip install scikit-learn==1.5.1


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
import numpy as np

df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_train


df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_test


sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
sample_sub


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


X = df_train.drop(columns=['id', 'Personality'])
y = df_train['Personality']


num_cols = X.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])


import matplotlib.pyplot as plt
import seaborn as sns

missing = X.isna().sum()
print("Missing values:\n", missing)


missing = df_train.isna().sum()
print("Missing Values per Column:")
for col, val in missing.items():
    print(f"- {col}: {val} missing values")


target_counts = df_train['Personality'].value_counts()
print("\nTarget Distribution (Personality):")
for label, count in target_counts.items():
    print(f"- {label}: {count} samples ({count/len(df_train)*100:.2f}%)")


print("\nNumeric Features Summary:")
for col in num_cols:
    mean_val = X[col].mean()
    median_val = X[col].median()
    min_val = X[col].min()
    max_val = X[col].max()
    std_val = X[col].std()
    print(f"- {col}: mean={mean_val:.2f}, median={median_val:.2f}, min={min_val:.2f}, max={max_val:.2f}, std={std_val:.2f}")



print("\nCategorical Features Summary:")
for col in cat_cols:
    counts = X[col].value_counts()
    print(f"\n- {col}:")
    for category, count in counts.items():
        print(f"    {category}: {count} samples ({count/len(X)*100:.2f}%)")


print("\nCorrelation between Numeric Features:")
corr = X[num_cols].corr()
print(corr)



print("\nNumeric Features vs Personality:")
for col in num_cols:
    grouped = df_train.groupby('Personality')[col].describe()
    print(f"\n- {col}:\n{grouped}")



print("\nCategorical Features vs Personality:")
for col in cat_cols:
    cross_tab = pd.crosstab(df_train[col], df_train['Personality'])
    print(f"\n- {col}:\n{cross_tab}")


from sklearn.preprocessing import LabelEncoder

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

target_le = LabelEncoder()
y = target_le.fit_transform(df_train['Personality'])


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


X_train['social_ratio'] = X_train['Social_event_attendance'] / (X_train['Friends_circle_size'] + 1)
X_train['outside_per_friend'] = X_train['Going_outside'] / (X_train['Friends_circle_size'] + 1)
X_train['posts_per_friend'] = X_train['Post_frequency'] / (X_train['Friends_circle_size'] + 1)





for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside']:
    X_train[f'{col}_bin'] = pd.qcut(X_train[col], q=5, labels=False)



X_train['num_mean'] = X_train[num_cols].mean(axis=1)
X_train['num_std'] = X_train[num_cols].std(axis=1)
X_train['num_max'] = X_train[num_cols].max(axis=1)
X_train['num_min'] = X_train[num_cols].min(axis=1)



X_train['alone_times_outside'] = X_train['Time_spent_Alone'] * X_train['Going_outside']
X_train['social_posts_ratio'] = X_train['Social_event_attendance'] * X_train['Post_frequency']
X_train['friends_activity'] = X_train['Friends_circle_size'] * X_train['Going_outside']
X_train['social_per_outside'] = X_train['Social_event_attendance'] / (X_train['Going_outside'] + 1)



from sklearn.preprocessing import PolynomialFeatures

poly_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(X_train[poly_cols])
poly_feature_names = poly.get_feature_names_out(poly_cols)

import pandas as pd
X_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=X_train.index)
X_train = pd.concat([X_train, X_poly], axis=1)



numeric_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
X_train['num_sum'] = X_train[numeric_cols].sum(axis=1)
X_train['num_mean'] = X_train[numeric_cols].mean(axis=1)
X_train['num_std'] = X_train[numeric_cols].std(axis=1)
X_train['num_max'] = X_train[numeric_cols].max(axis=1)
X_train['num_min'] = X_train[numeric_cols].min(axis=1)
X_train['num_range'] = X_train['num_max'] - X_train['num_min']



X_train['log_time_alone'] = np.log1p(X_train['Time_spent_Alone'].iloc[:, 0])
X_train['log_post_freq'] = np.log1p(X_train['Post_frequency'].iloc[:, 0])
X_train['friends_per_post'] = X_train['Friends_circle_size'].iloc[:, 0] / (X_train['Post_frequency'].iloc[:, 0] + 1)



X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_train


X_train = X_train.copy()
X_val = X_val.copy()


X_train['high_alone'] = (X_train['Time_spent_Alone_bin'] > 2).astype(int)
X_train['high_social'] = (X_train['Social_event_attendance_bin'] > 2).astype(int)
X_train['high_outside'] = (X_train['Going_outside_bin'] > 2).astype(int)



X_train


X_train = X_train.copy()
X_val = X_val.copy()
y_train = y_train.copy()
y_val = y_val.copy()


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Original train shape:", X_train.shape, y_train.shape)
print("After SMOTE:", X_train_res.shape, y_train_res.shape)



common_cols = X_train_res.columns.intersection(X_val.columns)
X_val_sync = X_val[common_cols].copy()


missing_cols = X_train_res.columns.difference(X_val.columns)
for col in missing_cols:
    X_val_sync[col] = 0 


from sklearn.cluster import KMeans
cluster_cols = [col for col in X_train_res.columns if X_train_res[col].dtype in [float, int] and 'bin' not in col]

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
X_train_res['cluster_label'] = kmeans.fit_predict(X_train_res[cluster_cols])
X_val_sync['cluster_label'] = kmeans.predict(X_val_sync[cluster_cols])


X_val_sync


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

feature_cols = [col for col in X_train_res.columns if X_train_res[col].dtype in [float, int]]
X_cluster_plot = X_train_res[feature_cols]

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_plot)

import pandas as pd
plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plot_df['cluster_label'] = X_train_res['cluster_label']
plot_df['Personality'] = y_train_res



plt.figure(figsize=(10,6))
sns.scatterplot(
    x='PC1', y='PC2',
    hue='cluster_label',
    style='Personality',
    palette='tab10',
    data=plot_df,
    alpha=0.7
)
plt.title("Clustering Result (PCA 2D) with Personality Labels")
plt.show()



from mpl_toolkits.mplot3d import Axes3D

pca3 = PCA(n_components=3, random_state=42)
X_pca3 = pca3.fit_transform(X_cluster_plot)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2],
                     c=X_train_res['cluster_label'], cmap='tab10', alpha=0.7)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title("3D Clustering Visualization")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()



distances = kmeans.transform(X_train_res[cluster_cols])
X_train_res['min_dist_to_centroid'] = distances.min(axis=1)

threshold = np.percentile(X_train_res['min_dist_to_centroid'], 95)
outliers = X_train_res[X_train_res['min_dist_to_centroid'] > threshold]

print("Number of outliers:", len(outliers))


X_train_clean = X_train_res[X_train_res['min_dist_to_centroid'] <= threshold].copy()


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train_res[cluster_cols])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
X_train_res['cluster_label'] = kmeans.fit_predict(X_scaled)


distances = kmeans.transform(X_train_res[cluster_cols])
X_train_res['min_dist_to_centroid'] = distances.min(axis=1)

threshold = np.percentile(X_train_res['min_dist_to_centroid'], 95)

mask = X_train_res['min_dist_to_centroid'] <= threshold
X_train_clean = X_train_res[mask].copy()
y_train_clean = y_train_res[mask].copy()


X_train_clean = X_train_clean.drop(columns=['min_dist_to_centroid'])


X_val_sync


X_train_clean


y_train_clean


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


num_cols = X_train_clean.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X_train_clean.select_dtypes(include=['object']).columns


smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_clean, y_train_clean)
print("Shape after SMOTE:", X_train_res.shape, y_train_res.shape)


scaler = StandardScaler()
X_train_res[num_cols] = scaler.fit_transform(X_train_res[num_cols])
X_val_sync[num_cols] = scaler.transform(X_val_sync[num_cols])


from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',       
    objective='binary:logistic',
    use_label_encoder=False,
    random_state=42,
    tree_method='hist'
)


X_val_sync = X_val_sync[X_train_res.columns]


xgb_model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_val_sync, y_val)],
    early_stopping_rounds=50,
    verbose=50
)


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

y_pred = xgb_model.predict(X_val_sync)
y_pred_proba = xgb_model.predict_proba(X_val_sync)[:, 1]

acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_val, y_pred))


cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



results = xgb_model.evals_result()
train_logloss = results['validation_0']['logloss']  # XGBoost dengan eval_set pertama adalah validation_0
plt.figure(figsize=(8,5))
plt.plot(train_logloss, label='Validation LogLoss')
plt.xlabel('Iteration')
plt.ylabel('LogLoss')
plt.title('XGBoost Validation LogLoss Over Iterations')
plt.legend()
plt.show()


df_test


sample_sub


import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

X_test = df_test.copy()

X_test['social_ratio'] = X_test['Social_event_attendance'] / (X_test['Friends_circle_size'] + 1)
X_test['outside_per_friend'] = X_test['Going_outside'] / (X_test['Friends_circle_size'] + 1)
X_test['posts_per_friend'] = X_test['Post_frequency'] / (X_test['Friends_circle_size'] + 1)

for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside']:
    # Gunakan cut yang sama dari training bisa pakai pd.qcut dengan q=5
    X_test[f'{col}_bin'] = pd.qcut(X_test[col].rank(method='first'), q=5, labels=False)

num_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
X_test['num_mean'] = X_test[num_cols].mean(axis=1)
X_test['num_std'] = X_test[num_cols].std(axis=1)
X_test['num_max'] = X_test[num_cols].max(axis=1)
X_test['num_min'] = X_test[num_cols].min(axis=1)
X_test['num_range'] = X_test['num_max'] - X_test['num_min']
X_test['num_sum'] = X_test[num_cols].sum(axis=1)


X_test['Time_spent_Alone'] = X_test['Time_spent_Alone'].fillna(X_train_res['Time_spent_Alone'].median())
X_test['Social_event_attendance'] = X_test['Social_event_attendance'].fillna(X_train_res['Social_event_attendance'].median())
X_test['Going_outside'] = X_test['Going_outside'].fillna(X_train_res['Going_outside'].median())
X_test['Friends_circle_size'] = X_test['Friends_circle_size'].fillna(X_train_res['Friends_circle_size'].median())
X_test['Post_frequency'] = X_test['Post_frequency'].fillna(X_train_res['Post_frequency'].median())

for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside']:
    X_test[f'{col}_bin'] = pd.qcut(X_test[col], q=5, labels=False)

X_test['alone_times_outside'] = X_test['Time_spent_Alone'] * X_test['Going_outside']
X_test['social_posts_ratio'] = X_test['Social_event_attendance'] * X_test['Post_frequency']
X_test['friends_activity'] = X_test['Friends_circle_size'] * X_test['Going_outside']
X_test['social_per_outside'] = X_test['Social_event_attendance'] / (X_test['Going_outside'] + 1)

X_test['log_time_alone'] = np.log1p(X_test['Time_spent_Alone'])
X_test['log_post_freq'] = np.log1p(X_test['Post_frequency'])
X_test['friends_per_post'] = X_test['Friends_circle_size'] / (X_test['Post_frequency'] + 1)

poly_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
poly_features = poly.transform(X_test[poly_cols])  # pakai transform, jangan fit_transform
poly_feature_names = poly.get_feature_names_out(poly_cols)
X_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=X_test.index)
X_test = pd.concat([X_test, X_poly], axis=1)

X_test['high_alone'] = (X_test['Time_spent_Alone_bin'] > 2).astype(int)
X_test['high_social'] = (X_test['Social_event_attendance_bin'] > 2).astype(int)
X_test['high_outside'] = (X_test['Going_outside_bin'] > 2).astype(int)

X_test = X_test.loc[:, ~X_test.columns.duplicated()]


for col in X_train_clean.columns:
    if col not in X_test.columns:
        X_test[col] = 0  
        
X_test = X_test[X_train_clean.columns]



for col in ['Stage_fear', 'Drained_after_socializing']:
    X_test[col] = X_test[col].map({'No': 0, 'Yes': 1})
    
model = XGBClassifier(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    tree_method='hist'
)

model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_val_sync, y_val)],
    early_stopping_rounds=50,
    verbose=50
)

X_test = X_test[X_train_clean.columns] 
preds = model.predict(X_test)


submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': ['Extrovert' if p==1 else 'Introvert' for p in preds]
})
print(submission.head(10))


submission.to_csv('/kaggle/working/submission.csv', index=False)

!ls /kaggle/working/


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()
submission.info()

