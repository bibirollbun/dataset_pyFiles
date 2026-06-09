import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
inference_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train_df.head()


train_df.isnull().values.any()


train_df.duplicated().values.any()


for cols in train_df.columns:
    print(f"{cols}: {train_df[cols].dtype}")


train_df.nunique()


num_col = [col for col in train_df.drop(['id', 'diagnosed_diabetes'], axis=1) if train_df[col].dtype in ['float64', 'int64']]
cat_col = [col for col in train_df.drop(['id', 'diagnosed_diabetes'], axis=1) if train_df[col].dtype == 'object']


train_df.describe()


sns.pairplot(
    data=train_df.sample(1000, random_state=42),
    hue="diagnosed_diabetes",
    vars=[col for col in train_df.drop(['id'], axis=1) if len(train_df[col].unique()) > 100],
)
plt.show()


df_x = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
df_y = pd.DataFrame({'diagnosed_diabetes': train_df.diagnosed_diabetes})

inference_id = inference_df['id']
inference_df.drop(['id'], axis=1, inplace=True)


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories="auto")
enc_ = pd.DataFrame(encoder.fit_transform(df_x[cat_col]), columns=cat_col)
df_x_ = pd.concat([df_x[num_col], enc_], axis=1)


corr = df_x_.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5, vmax=.3)
plt.show()


upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column]>0.8) or any(upper[column]<0.0002)]


df_x.drop(to_drop, axis=1, inplace=True)
inference_df.drop(to_drop, axis=1, inplace=True)


sns.countplot(data=df_y, x='diagnosed_diabetes')
plt.show()


df_x['hypertension_history'].value_counts()


numeric_cols = [
    col for col in train_df.drop(['id', 'diagnosed_diabetes'], axis=1).columns
    if train_df[col].dtype != 'object' and train_df[col].nunique() > 6
]

plt.figure(figsize=(15, 18))

i = 1
for col in numeric_cols:
    plt.subplot(5, 3, i)
    plt.boxplot(train_df[col], vert=True)
    plt.title(col, fontsize=11)
    plt.grid(alpha=0.3)
    i += 1

plt.tight_layout()
plt.show()


outlier_report = {}

for col in df_x.select_dtypes(include=['int64', 'float64']).columns:

    # Skip low-variance features
    if df_x[col].nunique() <= 5:
        continue

    Q1 = df_x[col].quantile(0.25)
    Q3 = df_x[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Count outliers
    outliers = ((df_x[col] < lower) | (df_x[col] > upper)).sum()
    outlier_report[col] = outliers

    # Cap the outliers
    df_x[col] = np.where(df_x[col] < lower, lower, df_x[col])
    df_x[col] = np.where(df_x[col] > upper, upper, df_x[col])

# Display report
for i, (col, cnt) in enumerate(outlier_report.items(), start=1):
    print(f"{i}. {col} â†’ {cnt} outliers capped")

    


plt.figure(figsize=(15, 18))

i = 1
for col in df_x.select_dtypes(include=['int64','float64']).columns:
    if df_x[col].nunique() > 5:
        plt.subplot(5, 3, i)
        plt.boxplot(df_x[col])
        plt.title(col)
        i += 1

plt.tight_layout()
plt.show()



# Load your training dataset
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')

# Define features and target
target = 'diagnosed_diabetes'

df_x = train_df.drop(columns=[target])
df_y = train_df[target]


import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

X = df_x.copy()
y = df_y.copy()

num_col = X.select_dtypes(include=['int64','float64']).columns
cat_col = X.select_dtypes(include='object').columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocess = ColumnTransformer([
    ('num', num_pipeline, num_col),
    ('cat', cat_pipeline, cat_col)
])

scale = (y == 0).sum() / (y == 1).sum()

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=scale
)



model = Pipeline([
    ('preprocess', preprocess),
    ('classifier', xgb_model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


param_dist = {
    'classifier__n_estimators': randint(200, 500),
    'classifier__max_depth': randint(3, 8),
    'classifier__learning_rate': uniform(0.03, 0.08),
    'classifier__subsample': uniform(0.75, 0.25),
    'classifier__colsample_bytree': uniform(0.7, 0.3),
    'classifier__min_child_weight': randint(1, 6),
    'classifier__gamma': uniform(0, 0.6)
}



search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=40,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

search.fit(X_train, y_train)


best_model = search.best_estimator_

y_pred = best_model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_pred)

print("\nğŸ”¥ Optimized ROC-AUC:", roc)
print("\nBest Parameters:\n", search.best_params_)


# Load inference (test) dataset
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Make predictions
test_preds = best_model.predict_proba(test_df)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': test_preds
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("âœ… submission.csv created successfully!")
submission.head()


