import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
inference_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


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


plt.figure(figsize=(4, 12))
i=1

for col in train_df.drop(['id', 'diagnosed_diabetes'], axis=1).columns:
    if train_df[col].dtype != 'object' and len(train_df[col].unique()) > 6:
        plt.subplot(5, 3, i)
        plt.boxplot(train_df[col])
        plt.title(f'plot for\n{col}', fontsize=10)
        i+=1
    else:
        continue
        
plt.tight_layout(rect=[0, 0.03, 2, 0.95])
plt.show()


i=1

for col in df_x.columns:
    if df_x[col].dtype != 'object' and len(df_x[col].unique()) > 5:
        q1 = np.quantile(df_x[col], 0.25)
        q3 = np.quantile(df_x[col], 0.75)
        iqr = q3 - q1
        
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        
        values = df_x[col].to_numpy()
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        print("{}. for {} there were {} outliers.".format(i, col, len(outliers)))

        df_x[col] = df_x[col].astype('float')
        df_x.loc[df_x[col] < lower_bound, col] = lower_bound
        df_x.loc[df_x[col] > upper_bound, col] = upper_bound
        
        i+=1
    


num_col = [col for col in df_x.columns if df_x[col].dtype != 'object']
cat_col = [col for col in df_x.columns if df_x[col].dtype == 'object']


from sklearn.model_selection import StratifiedKFold
# Another is StratifiedShuffleSplit, where the shuffling happens before each split, so there may be overlap of test sets
# In our case, we will use StratifiedKFold; there is shuffling just once before splitting, so no overlap of test sets

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier


model_cat = CatBoostClassifier(
    verbose=0
)              # other params are defined in param_grid


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),         # as there are no null values this wouldn't be needed but it's a good practice to include this
    ('scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),  # same here, good practice to include
    ('ord_enc', OrdinalEncoder(categories="auto")),
])

pre_processing = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_col),
    ('cat', cat_pipeline, cat_col),
])

model = Pipeline([
    ('preprocess', pre_processing),
    ('classifier', model_cat),
])


from sklearn.model_selection import train_test_split, GridSearchCV


X_train, X_test, y_train, y_test = train_test_split(df_x,
                                                    df_y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42)

param_grid = {
    'classifier__iterations': [50, 100, 150, 200, 250, 300],
    'classifier__learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    'classifier__depth': [5, 6, 7],
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=StratifiedKFold(5),
    scoring="roc_auc",
    n_jobs=-1,
)


grid.fit(X_train, y_train)

y_pred = grid.predict_proba(X_test)

roc_score = roc_auc_score(y_test, y_pred[:, 1])

print(f"\nROC score: {roc_score}")


preds = grid.predict_proba(
    inference_df
)


data = pd.DataFrame({
    'id': inference_id,
    'diagnosed_diabetes': preds[:, 1]
})

data.head()


data.to_csv(
    'submission.csv',
    index=False
)

