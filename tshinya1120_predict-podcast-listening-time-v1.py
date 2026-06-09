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


#å‰�å‡¦ç�†ãƒ©ã‚¤ãƒ–ãƒ©ãƒª
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel

#è©•ä¾¡ãƒ©ã‚¤ãƒ–ãƒ©ãƒª
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'MS Gothic'  # Windowsã�®å ´å�ˆ
import seaborn as sns
from scipy.stats import randint,uniform
import keras_tuner as kt
from keras_tuner import RandomSearch
import optuna
from sklearn.metrics import mean_squared_error
import shap
from sklearn.model_selection import RandomizedSearchCV


#ãƒ¢ãƒ‡ãƒ«
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

#èª­ã�¿è¾¼ã�¿ï¼†è¨“ç·´ãƒ‡ãƒ¼ã‚¿ä½œæˆ�å‡¦ç�†
Xt=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
Y=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

target="Listening_Time_minutes"


#ãƒ‡ãƒ¼ã‚¿ç¢ºèª�
print(Xt.info())
print(Xt.isna().sum())
print(Y.info())
print(Y.isna().sum())

def plot_column_distributions(df, max_cols=4, figsize=(16, 4), bins=30):
    """
    ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ã�®ã�™ã�¹ã�¦ã�®åˆ—ã�®åˆ†å¸ƒã‚’ä¸€æ‹¬ã�§ã‚°ãƒªãƒƒãƒ‰è¡¨ç¤ºã�™ã‚‹ã€‚
    æ•°å€¤åˆ—ã�¯ãƒ’ã‚¹ãƒˆã‚°ãƒ©ãƒ ã€�ã‚«ãƒ†ã‚´ãƒªåˆ—ã�¯æ£’ã‚°ãƒ©ãƒ•ã€‚
    """
    sns.set(style="whitegrid")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns
    
    all_cols = list(num_cols) + list(cat_cols)
    total = len(all_cols)
    rows = (total + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(rows, max_cols, figsize=(figsize[0], figsize[1] * rows))
    axes = axes.flatten()

    for i, col in enumerate(all_cols):
        ax = axes[i]
        if col in num_cols:
            sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax)
        else:
            sns.countplot(x=col, data=df, order=df[col].value_counts().index, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(col)
    
    # ä¸�è¦�ã�ªè»¸ã‚’é��è¡¨ç¤º
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

plot_column_distributions(Xt)


y = Xt[target]
X=Xt
X=X.drop([target], axis=1)

xtr,xval,ytr,yval=model_selection.train_test_split(X,y,test_size=0.2,random_state=0)


# ==== ç‰¹å¾´é‡�åˆ†é¡� ====
cat_cols3=[cname for cname in xtr.columns if xtr[cname].dtype == "object"]
num_cols =[cname for cname in xtr.columns if xtr[cname].dtype in ['int64', 'float64']]

imp=SimpleImputer(strategy="mean")


# ==== å‰�å‡¦ç�† ====
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

prep = ColumnTransformer(transformers=[
        ('num',numeric_transformer , num_cols),
        ('cat',categorical_transformer , cat_cols3)
    ])

# ==== ç‰¹å¾´é‡�é�¸æŠ�å™¨ ====
feature_selector = SelectFromModel(
    XGBRegressor(n_estimators=100, random_state=0),
    threshold=0.005  # importance > 0.01 ã‚’é�¸æŠ�
)

pip = Pipeline(steps=[
    ("preprocessor", prep),
    ("feature_selector", feature_selector),
    ("model", XGBRegressor(n_estimators=100, random_state=0))
])

# ==== å­¦ç¿’ ====
pip.fit(xtr, ytr)


ypred=pip.predict(xval)
rmse = mean_squared_error(yval, ypred, squared=False)
print(f"ğŸ“� RMSEï¼ˆRoot Mean Squared Errorï¼‰: {rmse:.4f}")


# === é–¢æ•°ï¼šé�¸ã�°ã‚Œã�Ÿç‰¹å¾´é‡�ã�®å��å‰�ã‚’å�–å¾— ===
def get_selected_features_from_pipeline(pipeline, num_cols, cat_cols):
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"]["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    all_feature_names = list(num_cols) + list(cat_feature_names)
    selector = pipeline.named_steps["feature_selector"]
    selected_mask = selector.get_support()
    selected_features = [name for name, keep in zip(all_feature_names, selected_mask) if keep]
    return selected_features

# === é–¢æ•°å‘¼ã�³å‡ºã�—ï¼šãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³ã�‹ã‚‰ç‰¹å¾´é‡�å�–å¾— ===
selected_features = get_selected_features_from_pipeline(pip, num_cols, cat_cols3)

print("\nâœ… é�¸ã�°ã‚Œã�Ÿé‡�è¦�ç‰¹å¾´é‡�ä¸€è¦§:")
for feat in selected_features:
    print(f"ãƒ»{feat}")

# === ãƒ¢ãƒ‡ãƒ«ã�®ç‰¹å¾´é‡�é‡�è¦�åº¦ã�¨çµ„ã�¿å�ˆã‚�ã�›ã�¦å�¯è¦–åŒ– ===
model = pip.named_steps["model"]
importances = model.feature_importances_

# DataFrame åŒ–
feat_imp_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

# === å�¯è¦–åŒ– ===
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x="Importance", y="Feature", palette="viridis")
plt.title("âœ… é�¸ã�°ã‚Œã�Ÿç‰¹å¾´é‡�ã�®é‡�è¦�åº¦")
plt.tight_layout()
plt.show()

#SHAPå€¤ã‚’ç”¨ã�„ã�Ÿç‰¹å¾´é‡�ã�®æŠŠæ�¡

# Step 1: ãƒ‡ãƒ¼ã‚¿ã‚’å‰�å‡¦ç�† + ç‰¹å¾´é‡�é�¸æŠ�å¾Œã�®å½¢å¼�ã�§å�–å¾—
# ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³ã�®é€”ä¸­ã�¾ã�§æ‰‹å‹•ã�§é�©ç”¨
preprocessor = pip.named_steps["preprocessor"]
selector = pip.named_steps["feature_selector"]
model = pip.named_steps["model"]
    
# X_train ã‚’ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³å‰�å‡¦ç�†å¾Œã�® NumPy é…�åˆ—ã�«å¤‰æ�›
X_transformed = preprocessor.transform(xtr)
    
# SelectFromModel ã�§ç‰¹å¾´é‡�é�¸æŠ�ã�•ã‚Œã�Ÿãƒ‡ãƒ¼ã‚¿
if hasattr(X_transformed, "toarray"):  # sparse â†’ dense å¤‰æ�›
    X_transformed = X_transformed.toarray()
    
X_selected = selector.transform(X_transformed)
    
# é�¸ã�°ã‚Œã�Ÿç‰¹å¾´é‡�å��ã‚’å�–å¾—
selected_features = get_selected_features_from_pipeline(pip, num_cols, cat_cols3)

# Step 2: SHAP explainer ä½œæˆ�ï¼ˆXGBoostå°‚ç”¨ï¼‰
explainer = shap.Explainer(model)
    
# Step 3: SHAPå€¤è¨ˆç®—
shap_values = explainer(X_selected)
    
# Step 4: å�¯è¦–åŒ–ï¼ˆé‡�è¦�åº¦ + æ–¹å�‘æ€§ï¼‰
shap.summary_plot(shap_values, features=X_selected, feature_names=selected_features)




#ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ãƒ�ãƒ¥ãƒ¼ãƒ‹ãƒ³ã‚°

    # ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã�®ç¯„å›²
param_dist = {
    "model__n_estimators": randint(100, 500),
    "model__max_depth": randint(3, 10),
    "model__learning_rate": uniform(0.01, 0.3),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.5, 0.5),
    "model__min_child_weight": randint(1, 10),
    "model__gamma": uniform(0, 0.3)}

    
    # ãƒ©ãƒ³ãƒ€ãƒ ã‚µãƒ¼ãƒ�
random_search = RandomizedSearchCV(
    estimator=pip,
    param_distributions=param_dist,
    n_iter=100,  # è©¦è¡Œå›�æ•°
    cv=4,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=0,
    random_state=0
)

# æ�¢ç´¢å®Ÿè¡Œ
random_search.fit(xtr, ytr)

# æœ€é�©ã�ªãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã�¨ã‚¹ã‚³ã‚¢
print("æ�¢ç´¢çµ�æ�œ:", random_search.best_params_)
print("ãƒ™ã‚¹ãƒˆã‚¹ã‚³ã‚¢:", random_search.best_score_)
param=random_search.best_params_
rf_params = {key.replace('model__', ''): value for key, value in param.items()}


model = XGBRegressor(**rf_params,random_state=0)
pip = Pipeline(steps=[
    ("preprocessor", prep),
    ("feature_selector", feature_selector),
    ("model",XGBRegressor(**rf_params,random_state=0))
])

pip.fit(xtr,ytr)
preds2=pip.predict(Y)

#æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã�®ä¿�å­˜
output = pd.DataFrame({'id': Y.id,
                       'Listening_Time_minutes': preds2})
output.to_csv('submission.csv', index=False)

