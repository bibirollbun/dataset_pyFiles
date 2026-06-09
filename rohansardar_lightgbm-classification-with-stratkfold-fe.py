import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


# Displaying the first 5 rows of the dataset
train.head()


# Checking for missing values
print(train.isna().sum())
print()
print(test.isna().sum())


# As per mutual information, duration column has the highest score
for df in [train, test]:
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_squared'] = df['duration'] ** 2
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_sqrt'] = np.sqrt(df['duration'])


# Displaying dataset with new features
train.head()


# Dropping id & target vector from test dataset
X = train.drop(['id', 'y'], axis=1)
y = train['y']

# Dropping id from test dataset
test.drop(['id'], axis=1, inplace=True)


# printing the columns of 'object' datatype
object_cols = X.select_dtypes(include="object").columns.tolist()
print(f"The object columns are: \n{object_cols}")


encoder = LabelEncoder()
for obj in object_cols:
    X[obj] = encoder.fit_transform(X[obj])
    test[obj] = encoder.transform(test[obj])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


target_variance = 0.99
pca = PCA(target_variance)
principalComponents = pca.fit(X_scaled)


print(f"The number of components to achieve {target_variance} variance is \
{principalComponents.n_components_}")


plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title(f'PCA: {principalComponents.n_components_} \
Components to Explain {target_variance:.0%} Variance')
plt.axhline(y=target_variance, color='r', linestyle='--')
plt.axvline(x=principalComponents.n_components_, color='g', linestyle='--',
            label=f'n_components={principalComponents.n_components_}')
plt.grid(True)
plt.legend()
plt.show()


pca2d = PCA(n_components=2)

principalComponents2d = pca2d.fit_transform(X_scaled)

principalDf2d = pd.DataFrame(
    data=principalComponents2d, 
    columns=['principalComponent1', 'principalComponent2']
)

# Concatenating the target vector y with 2 principal component values
finalDf2d = pd.concat([principalDf2d, y], axis=1)


fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1) 
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_title('PCA - 2D Projection', fontsize = 20)

targets = [0, 1]
colors = ['#008080', '#FF6F61']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf2d['y'] == target
    ax.scatter(finalDf2d.loc[indicesToKeep, 'principalComponent1'], 
               finalDf2d.loc[indicesToKeep, 'principalComponent2'], 
               c = color, s = 20, label=f'Class {target}')
ax.legend()
ax.grid()


pca3d = PCA(n_components=3)

principalComponents3d = pca3d.fit_transform(X_scaled)

principalDf3d = pd.DataFrame(
    data=principalComponents3d, 
    columns=['principalComponent1', 'principalComponent2', 'principalComponent3']
)

# Concatenating the target vector y with 3 principal component values
finalDf3d = pd.concat([principalDf3d, y], axis=1)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Principal Component 1', fontsize=10)
ax.set_ylabel('Principal Component 2', fontsize=10)
ax.set_zlabel('Principal Component 3', fontsize=10)
ax.set_title('PCA - 3D Projection', fontsize=20)

targets = [0, 1]
colors = ['#008080', '#FF6F61']

for target, color in zip(targets, colors):
    indicesToKeep = finalDf3d['y'] == target
    ax.scatter(
        finalDf3d.loc[indicesToKeep, 'principalComponent1'],
        finalDf3d.loc[indicesToKeep, 'principalComponent2'],
        finalDf3d.loc[indicesToKeep, 'principalComponent3'],
        c=color,
        s=20,
        label=f'Class {target}'
    )

ax.legend()
plt.show()


mi_scores = mutual_info_classif(X_scaled, y, random_state=42)


mi_series = pd.Series(mi_scores, index=X.columns)
mi_series = mi_series.sort_values(ascending=True)

print(mi_series)


mi_series.plot(kind='barh', figsize=(10, 6), color='#FF6F61')
plt.title('Mutual Information Scores')
plt.ylabel('MI Score')
plt.xlabel('Features')
plt.tight_layout()
plt.show()


X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df.head()


test_scaled_df = pd.DataFrame(test_scaled, columns=test.columns)
test_scaled_df.head()


def train_lightgbm(train, test, target):
    X = train
    y = target
    
    X_test = test.copy()
    
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_probs = np.zeros(len(X_test))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n<== Training fold {fold + 1}/{n_splits} ==>")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=30000,
            class_weights='balanced',
            learning_rate=0.055,
            num_leaves=100,
            max_depth=10,
            min_child_samples=8,
            subsample=0.85,
            colsample_bytree=0.5,
            reg_alpha=0.8,
            reg_lambda=0.3,
            max_bin=4851,
            random_state=2003,
            verbosity=-1,
            boosting_type='gbdt',
            eval_metric='auc',
            metric='auc'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(300),
                lgb.log_evaluation(500)
            ]
        )
        
        models.append(model)
        y_probs += model.predict_proba(X_test)[:, 1] / n_splits
    
    print("\nLightGBM model training complete.")
    return y_probs, models


y_probs, models = train_lightgbm(X_scaled_df, test_scaled_df, y)


sub_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

submission = pd.DataFrame({
    'id': sub_df['id'],
    'target': y_probs 
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

