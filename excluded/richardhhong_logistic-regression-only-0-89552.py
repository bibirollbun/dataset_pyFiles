import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


df_raw = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


df_raw.head()


df_raw.describe()


df_raw.info()


df_clean = df_raw.drop(columns=["id"])


features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']


sns.histplot(df_clean["rainfall"], bins=2)
plt.show()


fig, axes = plt.subplots(5, 2, figsize=(16, 8))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(df_clean[feature], ax=axes[i], bins=min(df_clean[feature].nunique(), 30))
    axes[i].set_title(f"Distribution of {feature}")

plt.tight_layout()
plt.show()


sns.pairplot(df_clean[features])


corr_matrix = df_clean[features + ["rainfall"]].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title(f"Correlation Matrix for Rain")


# viewing relationship between each variable and date
fig, axes = plt.subplots(5, 2, figsize=(16, 12))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.scatterplot(data=df_clean, x="day", y=feature, hue="rainfall", ax=axes[i])
    axes[i].set_title(f"{feature} over Time")

plt.tight_layout()
plt.show()


# distribution of features with respect to rainfall
fig, axes = plt.subplots(5, 2, figsize = (16, 12))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(x = feature, data = df_clean, ax=axes[i], bins=min(df_clean[feature].nunique(), 30), multiple="stack", hue="rainfall", zorder=2)
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].grid(zorder=1)

plt.tight_layout()
plt.show()


# distribution of features with respect to rainfall
fig, axes = plt.subplots(5, 2, figsize = (16, 12))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(x = feature, data = df_clean, ax=axes[i], bins=min(df_clean[feature].nunique(), 30), multiple="fill", stat="probability", hue="rainfall", zorder=2)
    axes[i].set_title(f"Proportion of Rain with respect to {feature}")

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df_clean.columns


features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']


X = df_clean.drop(columns=["rainfall"])
y = df_clean["rainfall"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


X_train.describe()


def make_cool_features(df):
    df_temp = df.copy()
    # scaler = StandardScaler()
    # df_temp[features] = scaler.fit_transform(df_temp[features])

    for feature in features:
        df_temp[f"Log_{feature}"] = np.log(df_temp[feature] + 1)

    scaler = StandardScaler()
    df_temp[df_temp.drop(columns="day").columns] = scaler.fit_transform(df_temp[df_temp.drop(columns="day").columns])

    for i in range(len(features)-1):
        # polynomial terms
        df_temp[f"{features[i]}2"] = df_temp.apply(lambda x: x[features[i]]**2, axis=1)
        df_temp[f"{features[i]}3"] = df_temp.apply(lambda x: x[features[i]]**3, axis=1)
        for j in range(i+1, len(features)-1):
            # interaction term
            df_temp[f"{features[i]}_{features[j]}"] = df_temp[features[i]] * df_temp[features[j]]
    
    return df_temp


X_train_cool = make_cool_features(X_train)
X_val_cool = make_cool_features(X_val)


X_train_cool.head()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


# baseline model
clf0 = LogisticRegression(random_state=42, max_iter=100).fit(X_train, y_train)

# actual model
# using l1 regularization for feature selection as well
clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)

param_grid = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}

grid_search = GridSearchCV(
    clf,
    param_grid,
    cv=10,
    scoring='roc_auc'
)

grid_search.fit(X_train_cool, y_train)

best_model = grid_search.best_estimator_


feature_importance = pd.DataFrame({
    'Feature': X_train_cool.columns,
    'Coefficient': best_model.coef_[0]
})
selected_features = feature_importance[feature_importance['Coefficient'] != 0]['Feature'].tolist()


print(best_model)
print(selected_features)
print(len(selected_features))


y_val_probs = clf0.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_probs)
print(f"ROC AUC Score Baseline: {roc_auc:.4f}")

y_val_probs = best_model.predict_proba(X_val_cool)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_probs)
print(f"ROC AUC Score Good: {roc_auc:.4f}")


X_cool = make_cool_features(X)

best_model.fit(X_cool, y)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_test.info()


df_test.describe()


df_test = df_test.fillna(0)
scaler_test = StandardScaler()
df_test_cool = make_cool_features(df_test)


y_pred = best_model.predict_proba(df_test_cool.drop(columns=["id"]))[:, 1]
df_test["rainfall"] = y_pred
df_test.head() 


submission = df_test[["id", "rainfall"]]
submission.to_csv("submission.csv", index=False)
submission.head()

