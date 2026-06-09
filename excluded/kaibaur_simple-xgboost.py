# ============================================
# load library
# ============================================
import pandas as pd

# ============================================
# load data
# ============================================

# load train data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv", index_col='id')
df_test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv" , index_col='id')


df_train.head()


# ============================================
# EDA
# ============================================

from ydata_profiling import ProfileReport
report = ProfileReport(df_train, title='diabetespredict')
report.to_notebook_iframe()


# Remove rows with missing target, separate target from predictors
df_train.dropna(axis=0, subset=['diagnosed_diabetes'], inplace=True)
y = df_train.diagnosed_diabetes
df_train.drop(['diagnosed_diabetes'], axis=1, inplace=True)

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in df_train.columns if
                    df_train[cname].nunique() < 10 and 
                    df_train[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in df_train.columns if 
                df_train[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X      = df_train[my_cols].copy()
X_test = df_test[my_cols].copy()


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ============================================
# Preprocessing
# ============================================

# 1. numerical missing: median
numerical_transformer = SimpleImputer(strategy='median')

# 2. categorical missing with OneHot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 3. merge
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# ============================================
# Model
# ============================================

model = XGBClassifier(
    n_estimators=1000,   # 木の数（適当な多めの数）
    learning_rate=0.05,  # 学習率
    n_jobs=-1,           # CPUフル稼働
    random_state=42      # 再現性確保
)

# ============================================
# make Pipeline
# ============================================

my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])



from sklearn.model_selection import cross_val_score

scores = cross_val_score(my_pipeline, X, y,
                         cv=5,
                         scoring='roc_auc')

print("AUC scores:", scores)
print("Mean AUC:", scores.mean())


# learning
my_pipeline.fit(X, y)

# predict
preds_proba = my_pipeline.predict_proba(X_test)[:, 1]

# submit
output = pd.DataFrame({'id': df_test.index, 'diagnosed_diabetes': preds_proba})
output.to_csv('submission.csv', index=False)

