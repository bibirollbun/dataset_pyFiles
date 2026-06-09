import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, RandomizedSearchCV
import warnings; warnings.filterwarnings('ignore')



train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df.head()


test_df.head()


df = pd.concat([train_df,test_df],ignore_index=True)
df.head()





def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())


check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)



cat_cols



num_cols


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=col_name, data=dataframe, 
                      order=dataframe[col_name].value_counts().index)
        plt.xticks(rotation=45)  
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df,col,plot=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df,col,plot=True)


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)
correlation_matrix(df,num_cols)


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


for col in num_cols:
    if col != "Calories":
        replace_with_thresholds(df, col)


for col in num_cols:
    print(col, check_outlier(df, col))


df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 35, 50, 65, 100],
                         labels=['Teen', 'YoungAdult', 'Adult', 'MiddleAge', 'Senior'])
df['Intensity'] = df['Heart_Rate'] * df['Duration']
df['Temp_Weight_Ratio'] = df['Body_Temp'] / df['Weight']
df['BMR_approx'] = 10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] + 5 




df.head()


check_df(df)


cat_cols, num_cols, cat_but_car = grab_col_names(df)



def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_column


missing_values_table(df)



def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Calories", cat_cols)


# Nadir sınıfların  Object veriler üzerinden tespit edilmesi
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df = rare_encoder(df,0.01)


df.head()


cat_cols, cat_but_car, num_cols = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)


train_df = df[df['Calories'].notnull()]
test_df = df[df['Calories'].isnull()]


y = np.log1p(train_df['Calories'])
X = train_df.drop(columns=['Calories', 'id'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [
            # ('LR', LinearRegression()),
            # ("Ridge", Ridge()),
            # ("Lasso", Lasso()),
            # ("ElasticNet", ElasticNet()),
            # ('KNN', KNeighborsRegressor()),
            # ('CART', DecisionTreeRegressor()),
            # ('RF', RandomForestRegressor()),
            # ('SVR', SVR()), #Support Vector Machines(regressor)
            # ('GBM', GradientBoostingRegressor()),
            # ("XGBoost", XGBRegressor(objective='reg:squarederror')),
            # ("CatBoost", CatBoostRegressor(verbose=0)),
            ("LightGBM", LGBMRegressor(verbose=0))
        ]

for name, regressor in models:
    scores = cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_log_error")
    rmsle = np.mean(np.sqrt(-scores))
    print(f"RMSLE: {round(rmsle, 4)} ({name})")


# rf_params = {"max_depth": [8, 15, None],
#              "max_features": [5, 7, "sqrt"],
#              "min_samples_split": [15, 20],
#              "n_estimators": [200, 300,500]}

# xgboost_params = {"learning_rate": [0.1, 0.01],
#                   "max_depth": [5, 8],
#                   "n_estimators": [100,200,500],
#                   "colsample_bytree": [0.5, 1],}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

# gbm_params = {"learning_rate": [0.01, 0.1],
#               "max_depth": [3, 8],
#               "n_estimators": [500, 1000],
#               "subsample": [1, 0.5, 0.7]}


regressors = [#('KNN', KNeighborsClassifier(), knn_params),
               #("CART", DecisionTreeRegressor(random_state=42), cart_params),
               # ("RF", RandomForestRegressor(random_state=42), rf_params),
               # ('XGBoost', xgboost.XGBRegressor(eval_metric='logloss',random_state=42), xgboost_params),
               # ('GBM', GradientBoostingRegressor(random_state=1),gbm_params),

               ('LightGBM', LGBMRegressor(random_state=42,verbose=-1), lightgbm_params),]
                #('CatBoost', CatBoostClassifier(verbose=False),catboost_params)]
best_models = {}




best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")

    rmsle = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_log_error")))
    print(f"RMSLE (Before): {round(rmsle, 4)} ({name})")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False, scoring="neg_mean_squared_log_error").fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)

    rmsle = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_log_error")))
    print(f"RMSLE (After): {round(rmsle, 4)} ({name})")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    
    best_models[name] = final_model



# En iyi model eğitimi
final_model.fit(X_train, y_train)

# Tahmin alma
y_pred = final_model.predict(X_test)



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(x=np.expm1(y_test), y=np.expm1(y_pred))
plt.xlabel("Gerçek Kalori")
plt.ylabel("Tahmin Edilen Kalori")
plt.title("Gerçek vs Tahmin (LightGBM)")
plt.plot([0, max(np.expm1(y_test))], [0, max(np.expm1(y_test))], 'r--')  # 45 derece çizgisi
plt.show()



# Train ve test hatası farkı
train_preds = final_model.predict(X_train)
test_preds = final_model.predict(X_test)

# Log'tan çıkalım
train_preds_inv = np.expm1(train_preds)
y_train_inv = np.expm1(y_train)
test_preds_inv = np.expm1(test_preds)
y_test_inv = np.expm1(y_test)

train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_preds_inv))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_preds_inv))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Fark:", abs(train_rmse - test_rmse))



# Modelin eğitimde kullandığı sütunları kullanarak test setini yeniden oluştur
X_test_final = test_df[X_train.columns]  # Sadece aynı 15 feature'ı al

# Tahmin yap
test_preds = final_model.predict(X_test_final)
test_preds_inv = np.expm1(test_preds)

# Submission dosyası oluştur
submission = pd.DataFrame({
    "id": test_df.index,  # test_df'de id sütunu yoksa index kullanılabilir
    "Calories": test_preds_inv
})

submission["Calories"] = submission["Calories"].round(2)
submission.to_csv("submission.csv", index=False)




submission.head()



submission.info()


submission.to_csv("submission.csv", index=False)





