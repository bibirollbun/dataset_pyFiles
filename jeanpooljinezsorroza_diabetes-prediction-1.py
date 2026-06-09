import numpy as np
import pandas as pd

# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


#----------------------------------------------------------------------------------------------------------------------------
def grab_col_names(df, target=None, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtype in ["O", "category", "bool"]]
    num_but_cat = [col for col in df.columns 
                   if df[col].nunique() < cat_th and df[col].dtype in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns 
                   if df[col].nunique() > car_th and df[col].dtype in ["O", "category"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    if target:
        for col_list in [cat_cols, num_cols, cat_but_car, num_but_cat]:
            if target in col_list:
                col_list.remove(target)
    cat_cols = [col for col in cat_cols if col not in num_but_cat]
    print("-" * 20)
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print("-" * 20)
    print('cat_cols:\n',cat_cols)
    print('num_cols:\n',num_cols)
    print('cat_but_car:\n',cat_but_car)
    print('num_but_cat:\n',num_but_cat)
    print("-" * 20)    
    return cat_cols, num_cols, cat_but_car, num_but_cat

'''cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df=df, target='target')'''


df_diabetes = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
df_diabetes.head()


Target = 'diagnosed_diabetes'
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df=df_diabetes, target=Target)


from sklearn.model_selection import train_test_split

X = df_diabetes.drop([Target], axis=1)
y = df_diabetes[Target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69, stratify=y)


from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ===============================
# MODELOS LINEALES
# ===============================
logistic = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    random_state=69
)

ridge = RidgeClassifier(
    random_state=69
)

# ===============================
# ÁRBOLES
# ===============================
decision_tree = DecisionTreeClassifier(
    random_state=69
)

random_forest = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    random_state=69
)

extra_trees = ExtraTreesClassifier(
    n_estimators=200,
    n_jobs=-1,
    random_state=69
)

gradient_boosting = GradientBoostingClassifier(
    random_state=69
)

# ===============================
# SUPPORT VECTOR MACHINES
# ===============================
linear_svm = LinearSVC(
    max_iter=2000,
    random_state=69
)

# ===============================
# NAIVE BAYES
# ===============================
gaussian_nb = GaussianNB()

multinomial_nb = MultinomialNB()

# ===============================
# KNN
# ===============================
knn = KNeighborsClassifier()

# ===============================
# RED NEURONAL (MLP)
# ===============================
mlp = MLPClassifier(
    max_iter=300,
    random_state=69
)

# ===============================
# XGBOOST
# ===============================
xgboost = XGBClassifier(
    eval_metric="logloss",
    random_state=69
)

# ===============================
# LIGHTGBM
# ===============================
lightgbm = LGBMClassifier(
    random_state=69
)

# ===============================
# CATBOOST
# ===============================
catboost = CatBoostClassifier(
    loss_function="Logloss",
    verbose=0,
    random_state=69
)

# ===============================
# ENSEMBLES BÁSICOS
# ===============================
voting = VotingClassifier(
    estimators=[
        ("lr", logistic),
        ("rf", random_forest),
        ("gb", gradient_boosting)
    ],
    voting="soft"
)

stacking = StackingClassifier(
    estimators=[
        ("rf", random_forest),
        ("gb", gradient_boosting),
        ("lr", logistic)
    ],
    final_estimator=LogisticRegression(max_iter=1000)
)

# ===============================
# COLECCIÓN DE MODELOS (útil para loops)
# ===============================
models = {
    "logistic": logistic,
    "ridge": ridge,
    "decision_tree": decision_tree,
    "random_forest": random_forest,
    "extra_trees": extra_trees,
    "gradient_boosting": gradient_boosting,
    "linear_svm": linear_svm,
    "gaussian_nb": gaussian_nb,
    "multinomial_nb": multinomial_nb,
    "knn": knn,
    "mlp": mlp,
    "xgboost": xgboost,
    "lightgbm": lightgbm,
    "catboost": catboost,
    "voting": voting,
    "stacking": stacking
}


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

ordinal_features = ["education_level", "income_level"]

ordinal_categories = [
    ["No formal", "Highschool", "Graduate", "Postgraduate"],
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]
]

nominal_features = [
    "gender",
    "ethnicity",
    "smoking_status",
    "employment_status"
]

numeric_scaled = ["alcohol_consumption_per_week"]

preprocessor = ColumnTransformer(
    transformers=[
        ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_features),
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse=False), nominal_features),
        ("num", RobustScaler(), num_cols),
        ("num_scaled", StandardScaler(), numeric_scaled)
    ],
    remainder="passthrough"
)


# from sklearn.pipeline import Pipeline
# from sklearn.metrics import roc_auc_score

# results = []

# for name, model in models.items():

#     try:
#         # Construir pipeline (si no usas preprocessor, reemplaza por model directamente)
#         pipe = Pipeline(
#             steps=[
#                 ("preprocessor", preprocessor),
#                 ("model", model)
#             ]
#         )

#         # Entrenamiento
#         pipe.fit(X_train, y_train)

#         # Predicción de scores
#         if hasattr(pipe.named_steps["model"], "predict_proba"):
#             y_score = pipe.predict_proba(X_test)[:, 1]
#         elif hasattr(pipe.named_steps["model"], "decision_function"):
#             y_score = pipe.decision_function(X_test)
#         else:
#             # Fallback (no ideal, pero seguro)
#             y_score = pipe.predict(X_test)

#         # Métrica
#         roc_auc = roc_auc_score(y_test, y_score)

#         results.append({
#             "model": name,
#             "roc_auc": roc_auc
#         })

#         print(f"{name:<20} ROC AUC: {roc_auc:.4f}")

#     except Exception as e:
#         print(f"{name:<20} ERROR -> {str(e)}")

# # Resultados ordenados
# results_df = pd.DataFrame(results).sort_values(
#     by="roc_auc",
#     ascending=False
# )

# print("\n=== RANKING DE MODELOS ===")
# print(results_df)


