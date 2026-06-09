import seaborn as sns
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold


sns.set_theme(style="whitegrid", palette="Set2")
warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
df_datasert = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_train.describe()


df_datasert.describe()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].set_title("Train Dataset")
df_train_personality_group = df_train["Personality"].value_counts().reset_index()
df_train_personality_group.columns = ["Personality", "Count"]
sns.barplot(data=df_train_personality_group, x="Personality", y="Count", ax=axes[0])

axes[1].set_title("Datasert Dataset")
df_datasert_personality_group = df_datasert["Personality"].value_counts().reset_index()
df_datasert_personality_group.columns = ["Personality", "Count"]
sns.barplot(data=df_datasert_personality_group, x="Personality", y="Count", ax=axes[1])

plt.tight_layout()
plt.show()


def make_mi_score(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores


X = df_train.copy()
X = X.drop(columns='id')
X = X.dropna()
y = X.pop("Personality")


for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

discrete_features = X.dtypes == int

mi_scores = make_mi_score(X, y, discrete_features=discrete_features)

mi_scores = mi_scores.reset_index()

mi_scores.columns = ["feature", "mi_score"]

plt.figure(figsize=(14, 4))
sns.barplot(data=mi_scores, y="mi_score", x="feature", palette="Set2")
plt.tight_layout()
plt.xticks(rotation=90)
plt.show()


X = df_train.copy()
X = X.dropna()

for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

y = X.pop("Personality")

lda = LDA(n_components=1)

X_lda = lda.fit_transform(X, y)
loadings = lda.scalings_

loadings_df = pd.DataFrame({"feature": X.columns, "LD1": loadings[:, 0]})
loadings_df = loadings_df.sort_values("LD1", key=abs, ascending=False)

loadings_df.reset_index()

plt.figure(figsize=(14, 3))
sns.barplot(data=loadings_df, y='LD1', x='feature')
plt.xticks(rotation=90)
plt.show()


df_datasert = (
    df_datasert
    .rename(columns={
        "Personality": "match_p"
    })
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency'])
)

merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']


df_test = df_test.merge(df_datasert, how='left', on=merge_cols)
df_train = df_train.merge(df_datasert, how='left', on=merge_cols)

target_col = "Personality"

X_train = df_train.drop(columns=["Personality", "id"])
y_train = df_train["Personality"]

X_test = df_test.drop(columns=["id"])

X_train.drop_duplicates()


numerical_cols = X_train.select_dtypes(["number"]).columns
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

print("numerical_cols", numerical_cols)
print("categorical_cols", categorical_cols)


 numercal_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=82,
                max_depth=24,
                min_samples_split=6,
                min_samples_leaf=7,
                max_features='sqrt',
                random_state=42,
            ),
            max_iter=20,
            tol=1e-4,
            imputation_order='ascending',
            n_nearest_features=10,
            skip_complete=True,
            random_state=42
        )),
        ('scaler', RobustScaler()),
    ])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numercal_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols),
])


pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('clsf', RandomForestClassifier(
        n_estimators=210,
        max_depth=16,
        min_samples_split=13,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
    ))
])


pipeline.fit(X_train, y_train)


y_preds = pipeline.predict(df_test)

submission_df["Personality"] = y_preds

submission_df.to_csv("submission.csv", index=False)

