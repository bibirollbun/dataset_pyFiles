import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import shap
import random
from sklearn.metrics import mean_squared_error,roc_auc_score
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer, KNNImputer
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor
# import warnings
# warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def missing(df):
	missing_number = df.isnull().sum().sort_values(ascending=False)
	missing_percent = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
	missing_values = pd.concat([missing_number, missing_percent], axis=1)
	return missing_values


df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
samples_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv',index_col='id')


print("Number of entries in training data:",len(df_train))
print("Number of entries in test data:",len(df_test))
print("Number of columns in training data:",len(df_train.columns))
print("Number of columns in test data:",len(df_test.columns))
print("Columns in training data:", df_train.columns)


df_train.head()


df_train.describe()


df_train.info()
print(missing(df_train))


cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 
            'Color']
num_cols = ['Compartments', 'Weight Capacity (kg)']


def NullValuefit(df):
    df = df.copy()
    for col in cat_cols:
        df.fillna({col: 'None'},inplace=True)

    df.fillna({'Weight Capacity (kg)': -1},inplace=True)
    return df

df_train = NullValuefit(df_train)
df_test = NullValuefit(df_test)


feature_cols = df_test.columns.to_list()
print("Number of duplicates in train data:", len(df_train[df_train[feature_cols].duplicated()]), "\n")
print("Number of duplicates in test data:", len(df_test[df_test.duplicated()]), "\n")


for c in cat_cols:
    # compare test and train cat values
    A=df_train[c].astype('str').unique()
    B=df_test[c].astype('str').unique()
    C = np.setdiff1d(B,A)
    print(f"{c}: Test has categories {C} which are not in train.")
    if len(C)>0:
        print(f" => {len(df_test.loc[df_test[c].astype(str).isin(C)])} rows" )


# 通过这个可以看见每个特征的类别分布情况，得到每个特征的类别总数、平均类别频率、最小类别频率和最大类别频率的描述性统计结果
feat_stat = {}
for feature in df_train.columns:
    feat_stat[feature] = np.round(df_train[
        feature].value_counts().describe())[['count','mean', 'min', 'max']].astype(int)
feat_stat_df = pd.DataFrame(feat_stat)
feat_stat_df.T.sort_values(by=['count'])


# print(df_train['Price'].value_counts().sort_index().to_string())


fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sns.histplot(df_train["Price"], kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Price Distribution")
axes[0].set_xlabel("Price")
axes[0].set_ylabel("Frequency")
sns.histplot(df_train["Price"], kde=True, ax=axes[1], color='orange', log_scale =True)
axes[1].set_title("Price Log Distribution")
axes[1].set_xlabel("Log Price")
axes[1].set_ylabel("Frequency")
sns.boxplot(x=df_train["Price"], ax=axes[2], color="lightcoral")
axes[2].set_title("Boxplot of Price")
axes[2].set_xlabel("Price")
plt.tight_layout()
plt.show()
price_summary = df_train["Price"].describe()
price_summary


plt.figure(figsize=(10, 6))
for i , col in enumerate(cat_cols,1):
    plt.subplot(3,3,i)
    sns.boxplot(x=col, y = "Price", data=df_train, palette="Dark2")
    plt.title(col)
    plt.tight_layout()


def display_categories(df, feature, sort_feature='median price'):
    """
    displays the full list of categories of a categorical feature together with their
    properties count and mean and median price for each category
    """
    mean_price_per_category = np.round(df_train.groupby(feature)[
        'Price'].mean()).reset_index().rename(columns={'Price': 'mean price'})
    median_price_per_category = df_train.groupby(feature)[
        'Price'].median().reset_index().rename(columns={'Price': 'median price'})
    category_counts = pd.DataFrame(df_train[feature].value_counts())
    merger = pd.merge(mean_price_per_category, category_counts, on=feature)
    merger = pd.merge(median_price_per_category,merger, on = feature).sort_values(by=sort_feature)
    print(merger.to_string(index = False))
    
    sns.scatterplot(x=merger['mean price'], y=merger['median price'],
                color='grey', s = 100)
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            # if point['x'] > 150000:
                ax.text(point['x']+.02, point['y'], str(point['val']))
    
    label_point(merger['mean price'], merger['median price'], merger[feature], plt.gca()) 
    plt.show()
    return merger
    
merger = display_categories(df_train, 'Brand')
merger = display_categories(df_train, 'Material')
merger = display_categories(df_train, 'Size')
merger = display_categories(df_train, 'Style')
merger = display_categories(df_train, 'Color')
merger = display_categories(df_train, 'Laptop Compartment')
merger = display_categories(df_train, 'Waterproof')


# num_cols = ['Compartments', 'Weight Capacity (kg)']

def set_boxplot_color(x):
    colors = []
    for bin in x.value_counts().index:
        count = x.value_counts()[bin]
        color = 'orange' if count > 30000 else 'grey'
        colors.append(color)
    return colors

sns.set(rc={'figure.figsize': (20, 5)})
compartments = df_train.Compartments
colors = set_boxplot_color(compartments)
sns.boxplot(x=compartments, y=df_train['Price'], palette = colors)
plt.ylim((0,160))
plt.xticks(rotation=45)
plt.show()


def set_boxplot_color(x):
    colors = []
    for bin in x.value_counts().index:
        count = x.value_counts()[bin]
        color = 'orange' if count > 30000 else 'grey'
        colors.append(color)
    return colors

sns.set(rc={'figure.figsize': (20, 5)})
kg = (df_train['Weight Capacity (kg)']//10)*10
colors = set_boxplot_color(kg)
sns.boxplot(x=kg, y=df_train['Price'], palette = colors)
plt.ylim((0,160))
plt.xticks(rotation=45)
plt.show()


def fea_eng(df):
    cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 
            'Color']
    df = df.copy()
    
    for col in cat_cols:
        df.fillna({col: 'None'},inplace=True)
        df[col] = df[col].astype('category')

    df.fillna({'Weight Capacity (kg)': -1},inplace=True)
    df['Weight Capacity (kg)'] = (df['Weight Capacity (kg)']//10*10).astype('category')
    return df

train = fea_eng(pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id'))
test = fea_eng(pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id'))
samples_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv',index_col='id')


cat_cols.append('Weight Capacity (kg)')


def add_new_feature(data, name, feature, smoothing_param):

    # 只考虑数据中实际出现的类别，因为我们知道 train 和 test 中的类都是一模一样的
    agg = data.groupby(feature, observed=False)['Price'].agg(['count', 'mean','median','std'])
    counts = agg['count']
        
    if name == 'mean':
        target_value = data['Price'].mean()
        target_fun = agg['mean']        
    elif name == 'median':        
        target_value = data['Price'].median()
        target_fun = agg['median']
    elif name == 'std':
        target_value = data['Price'].std()
        target_fun = agg['std']    
    else:
        print("error: target function not implemented")
    
    price_per_category = (counts * target_fun + smoothing_param * target_value
                         ) / (counts + smoothing_param) 
        
    num_col = data[feature].map(price_per_category
                               ).astype(float).fillna(target_value).astype(int)
    return num_col

def target_encoding(X_train, X_val, cat_cols, new_cols = ['median'], smoothing_param = 0.0001):
    """
    Associates mean, median and standard deviation of prices 
    with categories for each categorical feature.
    The parameter smoothing_param is implemented to prevent overfitting on rare categories
    and is tunable.
    The new features are returned as integers
    """
    X_train_target = X_train.copy()
    X_val_target = X_val.copy()
    
    for feature in cat_cols:
        for name in new_cols:
            feature_name = feature + '_' + name + '_price'
            X_train_target[feature_name] = add_new_feature(X_train, name, feature, smoothing_param)
            X_val_target[feature_name] = add_new_feature(X_train, name, feature, smoothing_param)
    
    return X_train_target, X_val_target


# X_train, X_val = target_encoding(train, test, cat_cols)
# X_train.head()


def flag_price_outliers(data):
    df = data.copy()
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(df['Price'], 25)
    Q3 = np.percentile(df['Price'], 75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df['Price'] > upper_bound) | (df['Price'] < lower_bound)]
    df['Price'] = df['Price'].apply(
        lambda x: 0 if (x > upper_bound) | (x < lower_bound) else 1)
    
    return df, outliers


df, outliers = flag_price_outliers(train)
df['Price'].value_counts()


SEED = 42
NSPLITS = 5
NBAGS = 3


def fea_eng(df):
    cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 
            'Color']
    df = df.copy()
    
    for col in cat_cols:
        df.fillna({col: 'None'},inplace=True)
        df[col] = df[col].astype('category')

    df.fillna({'Weight Capacity (kg)': -1},inplace=True)
    df['Weight Capacity (kg)'] = (df['Weight Capacity (kg)']//10*10).astype('category')
    return df


def select_subset(X, target, subset_size=0.8):
    """
    随机抽取 X 与 target 中的子集，比例为 subset_size。
    """
    X_subset = X.sample(frac=subset_size, random_state=42)
    target_subset = target.loc[X_subset.index]
    return X_subset, target_subset

def augment_data_with_nans(X, target, threshold=0.1, subset_size=0.2):
    """
    随机引入 NaN 值来增强数据集，模拟数据缺失的情况。
    这种方法可以帮助模型更好地学习如何处理缺失数据，从而提高模型在实际应用中的鲁棒性。
    """
    
    X_subset, target_subset = select_subset(X, target, subset_size)
    # 重置索引并创建副本防止对原数据的改动
    X_augmented = X_subset.reset_index(drop=True).copy()  

    # 识别哪些列中包含 NAN 值
    columns_with_nan = [col for col in X.columns if X[col].isna().sum() > 0] 
    
    # 创建掩码标注哪些不是 NAN 值
    non_nan_mask = X_augmented[columns_with_nan].notna()

    # 基于阈值生成一个随机掩码,决定该列的哪些行被设置会 NAN值
    for col in columns_with_nan:
        random_mask = np.random.rand(len(X_augmented)) < threshold  
        
        X_augmented.loc[random_mask, col] = np.nan
    
    return X_augmented, target_subset

def gaussian_noise_injection(X, target, noise_level, subset_size=0.2):
    """
    向数值列注入高斯噪声来增强数据集，同时保留非数值列不变。
    这种方法可以帮助模型更好地处理噪声数据，从而提高模型的鲁棒性。
    """
    
    X_subset, target_subset = select_subset(X, target, subset_size)

    # 划分 数值列 和 非数值列
    numeric_cols = X_subset.select_dtypes(include=['float64', 'int64'])
    non_numeric_cols = X_subset.select_dtypes(exclude=['float64', 'int64'])

    # 对数值列缺失值进行插入
    imputer = SimpleImputer(strategy='mean')
    numeric_imputed = pd.DataFrame(imputer.fit_transform(numeric_cols), 
                                   columns=numeric_cols.columns, 
                                   index=numeric_cols.index)

    # 在数值列加入噪音
    augmented_numeric = numeric_imputed
    for col in augmented_numeric.columns:
        std_dev = augmented_numeric[col].std()
        if std_dev > 0:  # 如果标准差大于0（即该列存在变异性），则生成高斯噪声
            noise = np.random.normal(0, noise_level * std_dev, size=len(augmented_numeric))
            augmented_numeric[col] += noise

    # 与非数值列合并
    augmented_df = pd.concat([augmented_numeric, non_numeric_cols], axis=1)

    # 确保列名顺序与原先数据一致
    augmented_df = augmented_df[X_subset.columns]
    return augmented_df, target_subset


def crossvalidate(model, X, Y, test, n_bags = 5, n_splits = 5, seed = 42):
    # beginer time
    start_time = datetime.datetime.now()

    scores = []
    test_pred = np.zeros(len(test))
    oof = np.zeros(len(X))

    
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_index, val_index) in enumerate(cv.split(X, Y)):
            
        # Split the data into training and validation sets for the current fold
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        Y_train, Y_val = Y.iloc[train_index], Y.iloc[val_index]

        for i in range(n_bags):
            
            cols_temp = X_train.columns.to_list()
            random.Random(i).shuffle(cols_temp)
            X_train = X_train[cols_temp]
            X_val = X_val[cols_temp]
            test = test[cols_temp]
            
            m = clone(model)
            m.set_params(random_state = i + seed)
            m.fit(X_train, Y_train,eval_set=[(X_val, Y_val)])

            y_val_pred = m.predict(X_val)
            y_test_pred = m.predict(test)
            
            oof[val_index] += y_val_pred/n_bags
            test_pred += y_test_pred/cv.get_n_splits()/n_bags

        # 得到每 Fold 中，在 Bagging 上验证集的平均分数
        val_score = np.sqrt(mean_squared_error(Y_val, oof[val_index]))
        scores.append(val_score)

    elapsed_time = datetime.datetime.now() - start_time
    
    print(f"#RMSE: {np.array(scores)}, mean: {np.array(scores).mean():.7f} (+- {np.array(scores).std():.7f})")
    print(f"#OOf score: {np.sqrt(mean_squared_error(Y, oof)):.7f} ({int(np.round(elapsed_time.total_seconds() / 60))} min)")

    return test_pred, oof, scores


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
val = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
samples_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv',index_col='id')


# 训练集准备
X_train = train.drop(columns = "Price")
Y_train = train["Price"]

X_train_nan,Y_train_nan = augment_data_with_nans(X_train,Y_train)
X_tr_combined = pd.concat([X_train, X_train_nan], ignore_index=True).reset_index(drop=True)

X = fea_eng(X_tr_combined)
Y = pd.concat([Y_train, Y_train_nan], ignore_index=True).reset_index(drop=True)


# 验证集准备
X_val = val.drop(columns = "Price")
Y_val = val["Price"]

X_val = fea_eng(X_val)


# 测试集准备
test = fea_eng(test)


model = LGBMRegressor(verbose = -1)
test_pred, oof, scores = crossvalidate(model, X, Y, test)


model.fit(X,Y,eval_set=[(X_val, Y_val)])
y_val_pred = model.predict(X_val)
val_score = np.sqrt(mean_squared_error(Y_val, y_val_pred))
val_score


samples_df['Price'] = test_pred
samples_df.to_csv("submission.csv",index='id')
samples_df.head()

