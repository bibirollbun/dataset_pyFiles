import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import KBinsDiscretizer
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df = df.drop('id',axis = 1)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test = df_test.drop('id',axis=1)
df = df.drop_duplicates()


df.describe()


fig, axs = plt.subplots(nrows=7 ,ncols= 1, figsize = (8,18))

for i, col in enumerate(df.select_dtypes(include=[np.number]).columns):
    sns.histplot(df[col],bins=30,kde=True,ax=axs[i])
    axs[i].set_title(f'HISTOGRAM_{col}')
    axs[i].grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


nrows, ncols = 4, 2
total_plots = nrows * ncols

fig, axs = plt.subplots(nrows= 4,ncols=ncols, figsize = (12,16),dpi = 100)


for i , col in enumerate(df.select_dtypes(include=[np.number]).columns):
    row = i // 2
    col_idx = i % 2
    sns.violinplot(x=df['Sex'], y=df[col], palette='Set2', ax=axs[row, col_idx], showfliers=False)
    axs[row, col_idx].set_title(f'Sex vs {col}')
    axs[row, col_idx].grid(axis='y', linestyle='--', alpha=0.7)

for i in range(7, total_plots):
    row = i // ncols
    col_idx = i % ncols
    axs[row, col_idx].set_visible(False)
plt.tight_layout()
plt.show()


sns.heatmap(df.corr(numeric_only=True), annot = True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


sns.pairplot(df)
plt.tight_layout()
plt.show()


x_pca = df.drop(['Calories','Sex'], axis = 1)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(x_pca)


## EXPLAINED VARIANCE AFTER PCA 
pca.explained_variance_ratio_.sum()


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_pca)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$")
plt.ylabel("Inertia")
plt.grid()
plt.show()


kmean_model = KMeans(n_clusters= 3, random_state = 42)
kmean_model.fit_predict(X_pca)
df_pca = pd.DataFrame(X_pca, columns=['PCA1','PCA2','PCA3'])
pca_df = pd.concat([df_pca, pd.DataFrame({'cluster': kmean_model.labels_})], axis = 1)

##The 'Calories' feature was divided into tertiles using pd.qcut.
pca_df['group'] = pd.qcut(df['Calories'], q=3, labels=False)

(pca_df['group'] == pca_df['cluster']).mean()


##PLOT #
fig = plt.figure(figsize= (10,8))
ax = fig.add_subplot(111,projection='3d')

scatter = ax.scatter(
    pca_df['PCA1'],
    pca_df['PCA2'],
    pca_df['PCA3'],
    c = pca_df['group'],
    cmap = 'rainbow',
    marker = 'o',
    alpha = 0.6
)

ax.set_title('KMeans - 3D PCA Projection')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.colorbar(scatter)
plt.show()


numeric_cols = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','BMI','Intensity']
def feature_engineering(df : pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
   
    # sqrt for all features 
    for i in range(len(numeric_cols)):
        df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
        df['Intensity'] = df['Heart_Rate'] / df['Duration']
        feature_1 = numeric_cols[i]
        for j in range(i+1,len(numeric_cols)):
            feature_2 = numeric_cols[j]
            df[f'{feature_1}_x_{feature_2}'] = df[feature_1] * df[feature_2]
           
    return df


label_enc = LabelEncoder()
df['Sex'] = label_enc.fit_transform(df['Sex'])
df_test['Sex'] = label_enc.transform(df_test['Sex'])


train = feature_engineering(df,numeric_cols)
test = feature_engineering(df_test,numeric_cols)
train["Sex"] = train["Sex"].astype("category")
test["Sex"] = test["Sex"].astype("category")


X = train.drop(['Calories'], axis = 1 )
y = np.log1p(train["Calories"])


FOLDS = 40
KF = KFold(n_splits=FOLDS, shuffle = True, random_state = 42)
cat_features = ['Sex']
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

 # CATBOOST MODEL
cat_model = CatBoostRegressor(
    iterations= 3500,
    learning_rate= 0.02,
    depth= 12,
    loss_function= 'RMSE',
    l2_leaf_reg= 3,
    random_seed= 42,
    eval_metric= 'RMSE',
    early_stopping_rounds = 200,
    verbose= 1000,
    task_type= 'GPU')

 ## XGBOOST
xgb_model = XGBRegressor(
    max_depth=10,
    colsample_bytree=0.75,
    subsample=0.9,
    n_estimators=2000,
    learning_rate=0.01,
    gamma=0.01,
    max_delta_step=2,
    early_stopping_rounds=100,
    eval_metric="rmse",
    enable_categorical=True,
    device = 'cuda')

for i, (train_idx,valid_idx) in enumerate(KF.split(X,y)):
    print('#' * 15, i+1, '#' *15)
    ## SPLIT DS 
    X_train,y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
 
    ## CATBOOST fit
    cat_model.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],cat_features=cat_features,
            use_best_model=True,verbose=0)
    ## XGB FIR
    xgb_model.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],verbose=0)
    ## PREDICTION CATBOOST
    oof_cat[valid_idx] = cat_model.predict(X_valid)
    pred_cat += cat_model.predict(test)
    ## PREDICTION XGB
    oof_xgb[valid_idx] = xgb_model.predict(X_valid)
    pred_xgb += xgb_model.predict(test)
    
    cat_rmse = mean_squared_error(y_valid,oof_cat[valid_idx]) ** 0.5
    xgb_rmse = mean_squared_error(y_valid, oof_xgb[valid_idx]) ** 0.5
    
    print(f'FOLD {i+1} CATBOOST_RMSE = {cat_rmse:.4f} <=> XGB_RMSE = {xgb_rmse:.4f}')


# Average predictions from folds
pred_cat /= FOLDS
pred_xgb /= FOLDS




y_preds = np.expm1(pred_cat) * 0.30 + np.expm1(pred_xgb)*0.70
y_preds = np.clip(y_preds, 1, 314)

# Save submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission["Calories"] = y_preds
submission.to_csv("submission.csv", index=False)
print('submission saved')
submission.head()


plt.figure(figsize = (10,8),dpi=100)
sns.histplot(np.expm1(oof_cat),alpha=0.2,bins=30)
sns.histplot(np.expm1(oof_xgb),alpha=0.2,bins=30)
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()




