import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest  
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
print("Train shape", train.shape )
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print("Test shape:", test.shape)
test.head()


train.describe()


sns.heatmap(train.isnull(),cbar = False, cmap = 'viridis')
plt.title('Missing Values Heatmap')
plt.show()


sns.heatmap(test.isnull(),cbar = False, cmap = 'viridis')
plt.title('Missing Values Heatmap')
plt.show()


train.drop(columns = 'id', inplace = True)
test.drop(columns = 'id', inplace = True)
n_cols = len(train.columns)

n_rows = (n_cols // 3)  + (1 if n_cols % 3 else 0)

plt.figure(figsize=(15, 5 * n_rows))  

for i, col in enumerate(train.columns, 1):
    plt.subplot(n_rows, 3, i)  
    sns.boxplot(y=train[col])  
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


def drop_outliers(df, col,  threshold = 1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

before_drop_train = len(train)
before_drop_test= len(test)

for col in train.select_dtypes('number'):
    if col != 'rainfall':
        train = drop_outliers(train,col)
        
for col in test.select_dtypes('number'):
    test = drop_outliers(test,col)

print(f'Number of removed rows from train dataset: {before_drop_train - len(train)}')
print(f'Number of removed rows from test dataset: {before_drop_test - len(test)}')


sns.countplot(data=train, x='rainfall', order=train['rainfall'].value_counts().index)
plt.title('Number of samples per class')
plt.show()


train.hist(figsize = (12,5), bins = 20)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
filtered_corr = train.corr().where(np.abs(train.corr()) >= 0.6)
sns.heatmap(filtered_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


train.columns


X = train.drop('rainfall', axis = 1)
y = train.rainfall
X_train, X_test, y_train, y_test = train_test_split(X,y , random_state = 42)


gbc = GradientBoostingClassifier(max_depth = 4, random_state = 42)
model = gbc.fit(X_train,y_train)
predict = model.predict(X_test)
f1 = round(f1_score(y_test,predict, average = 'weighted'), 3)
print(f1)


fig, ax = plt.subplots(figsize=(10, 6))
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(X_train) 
scaled_train_df = pd.DataFrame(scaled_train, columns=X_train.columns)
variances = scaled_train_df.var(axis=0)

bars = ax.bar(variances.index, variances.values, width=0.2)
ax.bar_label(bars, labels=[f"{v:.2f}" for v in variances.values], fontsize=14, padding=2)

plt.title('Variance of Scaled Features (MinMaxScaler)')
plt.xlabel('Features')
plt.ylabel('Variance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


X_train_v1, X_test_v1, y_train_v1, y_test_v1 = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()


var_train = train.copy()

high_variance_cols = variances[variances > 0.05].index
high_variance_values = variances[high_variance_cols]


fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(high_variance_values.index, high_variance_values.values, width=0.2)
ax.bar_label(bars, labels=[f"{v:.2f}" for v in high_variance_values.values], fontsize=14, padding=2)

plt.title('Variance of Selected Scaled Features (MinMaxScaler)')
plt.xlabel('Features')
plt.ylabel('Variance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


del_X_train = X_train_v1[high_variance_cols]
del_X_test = X_test_v1[high_variance_cols]


gbc = GradientBoostingClassifier(max_depth = 4, random_state = 42)
model = gbc.fit(del_X_train,y_train_v1)
predict = model.predict(del_X_test)
var_f1 = round(f1_score(y_test_v1,predict), 3)
print(var_f1)


fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(['Base Model - F1', 'Variance Selection - F1'], [f1, var_f1], width=0.2)
ax.bar_label(bars, labels=[f"{v:.2f}" for v in [f1, var_f1]], fontsize=14, padding=2)

plt.title('Variance of Scaled Features (MinMaxScaler)')
plt.xlabel('Features')
plt.ylabel('Variance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


X_train_v2, X_test_v2, y_train_v2, y_test_v2 = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()


f1_score_list = []

for k in range(1,12):
    selector = SelectKBest(mutual_info_classif,k = k)
    selector.fit(X_train_v2, y_train_v2)

    sel_X_train_v2 = selector.transform(X_train_v2)
    sel_X_test_v2 = selector.transform(X_test_v2)

    gbc.fit(sel_X_train_v2, y_train_v2)
    kbest_preds = gbc.predict(sel_X_test_v2)

    f1_score_kbest = round(f1_score(y_test_v2, kbest_preds ,average = 'weighted'), 3)

    f1_score_list.append(f1_score_kbest)


fig, ax = plt.subplots()

x = np.arange(1, 12)
y = f1_score_list

ax.bar(x, y, width=0.2)
ax.set_xlabel('Number of features selected using mutual information')
ax.set_ylabel('F1-Score (weighted)')
ax.set_xticks(np.arange(1, 12))
for i, v in enumerate(y):
    plt.text(x=i+1, y=v+0.05, s=str(v), ha='center')
    
plt.tight_layout()


selector = SelectKBest(mutual_info_classif, k=3)
selector.fit(X_train_v2, y_train_v2)

selected_feature_mask = selector.get_support()

selected_features = X_train_v2.columns[selected_feature_mask]

selected_features


X_train_v3, X_test_v3, y_train_v3, y_test_v3 = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()



from sklearn.feature_selection import RFE

rfe_f1_score_list = []

for k in range(1, 12):
    RFE_selector = RFE(estimator=gbc, n_features_to_select=k, step=1)
    RFE_selector.fit(X_train_v3, y_train_v3)
    
    sel_X_train_v3 = RFE_selector.transform(X_train_v3)
    sel_X_test_v3 = RFE_selector.transform(X_test_v3)
    
    gbc.fit(sel_X_train_v3, y_train_v3)
    RFE_preds = gbc.predict(sel_X_test_v3)
    
    f1_score_rfe = round(f1_score(y_test_v3, RFE_preds, average='weighted'), 3)
    
    rfe_f1_score_list.append(f1_score_rfe)


fig, ax = plt.subplots()

x = np.arange(1, 12)
y = rfe_f1_score_list

ax.bar(x, y, width=0.2)
ax.set_xlabel('Number of features selected using RFE')
ax.set_ylabel('F1-Score (weighted)')
ax.set_xticks(np.arange(1, 12))

for i, v in enumerate(y):
    plt.text(x=i+1, y=v+0.05, s=str(v), ha='center')
    
plt.tight_layout()


RFE_selector = RFE(estimator=gbc, n_features_to_select=3, step=10)
RFE_selector.fit(X_train_v3, y_train_v3)

selected_features_mask = RFE_selector.get_support()

selected_features = X_train_v3.columns[selected_features_mask]
selected_features


X_train_v4, X_test_v4, y_train_v4, y_test_v4 = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()



from boruta import BorutaPy

boruta_selector = BorutaPy(gbc, random_state=42)

boruta_selector.fit(X_train_v4.values, y_train_v4.values.ravel())

sel_X_train_v4 = boruta_selector.transform(X_train_v4.values)
sel_X_test_v4 = boruta_selector.transform(X_test_v4.values)

gbc.fit(sel_X_train_v4, y_train_v4)

boruta_preds = gbc.predict(sel_X_test_v4)

boruta_f1_score = round(f1_score(y_test_v4, boruta_preds, average='weighted'), 3)


selected_features_mask = boruta_selector.support_

selected_features = X_train_v4.columns[selected_features_mask]
selected_features

