import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# file dealy và not_delay_7_9
data_df=pd.read_csv("delay_and_not_7_9.csv")
print(data_df.dtypes)


data_df.shape


data_df.columns


data_df.head()


data_df.sample(100)


data_df.info()


data_df.isnull().sum()


data_df.duplicated().sum()


data_df['Order date']=pd.to_datetime(data_df['Order date'], errors= 'coerce')


numeric_data_df= data_df.select_dtypes(include=["float64","int64"])
corr_matrix = numeric_data_df.corr()
corr_matrix


plt.figure(figsize=(10,8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=90)
plt.yticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns)
plt.title('Correlation Matrix')
plt.show()


corr_matrix = numeric_data_df.corr()
corr_unstack= corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
high_corr=corr_unstack.stack().reset_index()
high_corr.columns=['feature 1','feature 2','Correlation']
high_corr=high_corr[high_corr['Correlation'].abs()>0.5]
high_corr


data_df = data_df.rename(columns={'Order date': 'Order_date'})
data_df = data_df.rename(columns={'DIRECT SHIP FLG': 'DIRECT_SHIP_FLG'})
data_df = data_df.rename(columns={'Sales order line number': 'Sales_order_line_number'})
data_df = data_df.rename(columns={'SO QTY': 'SO_QTY'})
data_df = data_df.rename(columns={'Consider count hodiday Saturday': 'Consider_count_hodiday_Saturday'})


data_df=data_df.drop(['PRODUCT ATTRIBUTION','ALLOCATION QTY', 'SPECIAL DIV','PURCHASE AMOUNT','CLASSIFY_CD','SUPPLIER_DIV'],axis=1)
print(data_df.shape)


data_df.isnull().sum()


data_df=data_df.drop(['SOUF_RCV_NO','QTUF_RCV_NO','OTHER AREA SHIP DIV'],axis=1)


categorical_data_df= data_df.select_dtypes(include= ['object'])
categorical_data_df = categorical_data_df.dropna() 


categorical_data_df.dtypes


for column in categorical_data_df.columns:
    categorical_data_df[column] = categorical_data_df[column].astype('category')


categorical_data_df.dtypes


from scipy.stats import chi2_contingency

results = []
selected_columns=[column for column in categorical_data_df.columns if categorical_data_df[column].nunique()<50]
chi2_matrix=pd.DataFrame(np.zeros((len(selected_columns), len(selected_columns))),
                           columns=selected_columns,
                           index=selected_columns)
for i, column1 in enumerate(selected_columns):
    for j, column2 in enumerate(selected_columns[i+1:]):
        frequency=pd.crosstab(categorical_data_df[column1], categorical_data_df[column2])
        chi2, p, dof, expected = chi2_contingency(frequency)
        if p < 0.05:
                results.append((column1, column2, chi2, p))
        chi2_matrix.iloc[i, j+i+1] = chi2
        chi2_matrix.iloc[j+i+1, i] = chi2  
results_df=pd.DataFrame(results, columns=['Column1', 'Column2', 'Chi2', 'p_value'])
significant_relations = results_df.sort_values('p_value')
significant_relations


plt.figure(figsize=(12, 10))
sns.heatmap(chi2_matrix, cmap='coolwarm', fmt='.2f', xticklabels=chi2_matrix.columns, yticklabels=chi2_matrix.columns)
plt.title('Chi2 Square Matrix')
plt.tight_layout()
plt.show()


ship_mode=data_df['Ship Mode'].mode()[0]
data_df['Ship Mode'].replace(['NaN', ' ', ''], np.nan, inplace=True)
data_df['Ship Mode']=data_df['Ship Mode'].fillna(ship_mode)


data_df.isnull().sum()


data_df['SHIP DECISION NO']


data_df=data_df.drop(columns='REASON_CD', axis=1)


median_ship_decision=data_df['SHIP DECISION NO'].median()
data_df['SHIP DECISION NO']=data_df['SHIP DECISION NO'].fillna(median_ship_decision)


data_df.isnull().sum()


data_df = data_df.rename(columns={'WEIGHT PER PIECE': 'WEIGHT_PER_PIECE'})
data_df = data_df.rename(columns={'PACK QTY': 'PACK_QTY'})
data_df = data_df.rename(columns={'Stock class': 'Stock_class'})
data_df = data_df.rename(columns={'SUPPLIER INV AMOUNT': 'SUPPLIER_INV_AMOUNT'})
data_df = data_df.rename(columns={'PACKING RANK': 'PACKING_RANK'})
data_df = data_df.rename(columns={'LOGICAL PLANT': 'LOGICAL_PLANT'})
data_df = data_df.rename(columns={'Ship Mode': 'Ship_Mode'})
data_df = data_df.rename(columns={'SHIP DECISION NO': 'SHIP_DECISION_NO'})
data_df = data_df.rename(columns={'PACK QTY': 'PACK_QTY'})
data_df = data_df.rename(columns={'WEIGHT PER PIECE': 'WEIGHT_PER_PIECE'})


numeric_data_df=numeric_data_df.drop(columns='label')


outlier={}
numeric_data_df=[col for col in numeric_data_df if col in data_df.columns]

for column in numeric_data_df:
    Q1=data_df[column].quantile(0.25)
    Q3=data_df[column].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR

    outliers=data_df[(data_df[column]<lower_bound) | (data_df[column]>upper_bound)]
    outlier_percentage=len(outliers)/len(data_df)*100
    outlier[column]={
        'Number Outliers':len(outliers),
        'Outlier Percentage':outlier_percentage
    }
outlier_summary=pd.DataFrame(outlier).T
print(outlier_summary.sort_values('Number Outliers', ascending=False))


plt.figure(figsize=(12, 10))
sns.barplot(data=outlier_summary.sort_values('Number Outliers', ascending=False),
            x='Number Outliers', y=outlier_summary.index,
            palette='Reds_r')
plt.title('Số lượng giá trị ngoại lai theo thuộc tính')
plt.xlabel('Số lượng ngoại lai')
plt.ylabel('Thuộc tính')
plt.tight_layout()
plt.show()


for column in numeric_data_df:
    Q1=data_df[column].quantile(0.25)
    Q3=data_df[column].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers=data_df[(data_df[column]<lower_bound) | (data_df[column]>upper_bound)]
    outlier_percentage=len(outliers)/len(data_df)*100
    if outlier_percentage>0.1:
        lower_threshold = data_df[column].quantile(0.05)
        upper_threshold = data_df[column].quantile(0.95)
        data_df[column] = data_df[column].clip(lower=lower_threshold, upper=upper_threshold)
    elif outlier_percentage >0.05 and outlier_percentage<0.1:
        data_df[column]=np.log1p(data_df[column])
print(data_df.head())


data_df['label'].value_counts()


from sklearn.preprocessing import MinMaxScaler
#chuan hoa du lieu nhieu ngoai lai
scaler=MinMaxScaler() 
numeric_data_df=data_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
data_df[numeric_data_df]=scaler.fit_transform(data_df[numeric_data_df])
print(data_df.head())


categorical_data_df.columns


data_df['label'].value_counts()


data_df.columns


from sklearn.preprocessing import OrdinalEncoder

categorical_column= ['SUBSIDIARY_CD', 'GLOBAL_NO', 'BRAND_CD', 'INNER_CD', 'SUPPLIER_CD',
                     'PACKING_RANK', 'PRODUCT_CD', 'VSD', 'DELI_DIV', 'Ship_Mode'
]
data_df[categorical_column]=data_df[categorical_column].astype(str)
encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
data_df[categorical_column]=encoder.fit_transform(data_df[categorical_column])
data_encoded=pd.concat([data_df[categorical_column]], axis=1)
print(data_encoded.head())


data_df.describe()


data_df=data_df.drop(columns=['SPECIAL_DIV','SUBSIDIARY_CD','SPECIAL_DIV','SUBSIDIARY_CD','Stock_class','SHIP_DECISION_NO','PACK_QTY','SPECIAL_DIV'], axis=1)


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


drop_columns = ['label','ID', 'Order_date', 'GLOBAL_NO', 'WEIGHT_UNIT','HAZARD_FLG','SPECIFY_SHIP_DAYS','PRODUCT_ASSORT']  
x = data_df.drop(columns=drop_columns, errors='ignore')
y = data_df['label']

le_dict = {}
for col in x.select_dtypes(include='object').columns:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col].astype(str))
    le_dict[col] = le

# chia train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
print('Train:', x_train.shape)
print('Test:', x_test.shape)


print("Train label distribution:")
print(y_train.value_counts(normalize=True)) 
print("\test label distribution:")
print(y_test.value_counts(normalize=True))


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)


neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
ratio = neg / pos
model = XGBClassifier(scale_pos_weight=ratio, random_state=42)
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print("F1-Score train:", f1_score(y_train, y_train_pred))
print("F1-Score val  :", f1_score(y_test, y_test_pred))


print("Total samples:", len(x))
print("Unique labels:", y.nunique())


train_columns = x.columns.tolist()


df_test = pd.read_csv("PILOT_10.csv")
df_test['Order date']=pd.to_datetime(df_test['Order date'], errors= 'coerce')


df_test = df_test.rename(columns={'Order date': 'Order_date'})
df_test = df_test.rename(columns={'DIRECT SHIP FLG': 'DIRECT_SHIP_FLG'})
df_test = df_test.rename(columns={'Sales order line number': 'Sales_order_line_number'})
df_test = df_test.rename(columns={'SO QTY': 'SO_QTY'})
df_test = df_test.rename(columns={'Consider count hodiday Saturday': 'Consider_count_hodiday_Saturday'})
df_test=df_test.drop(['PRODUCT ATTRIBUTION','ALLOCATION QTY', 'SPECIAL DIV','PURCHASE AMOUNT','CLASSIFY_CD','SUPPLIER_DIV'],axis=1)
df_test=df_test.drop(['SOUF_RCV_NO','QTUF_RCV_NO','OTHER AREA SHIP DIV'],axis=1)


categorical_df_test= df_test.select_dtypes(include= ['object'])
categorical_df_test = categorical_df_test.dropna() 
categorical_df_test.dtypes
for column in categorical_df_test.columns:
    categorical_df_test[column] = categorical_df_test[column].astype('category')
categorical_df_test.dtypes
results = []
selected_columns=[column for column in categorical_df_test.columns if categorical_df_test[column].nunique()<50]

chi2_matrix=pd.DataFrame(np.zeros((len(selected_columns), len(selected_columns))),
                           columns=selected_columns,
                           index=selected_columns)
for i, column1 in enumerate(selected_columns):
    for j, column2 in enumerate(selected_columns[i+1:]):
        frequency=pd.crosstab(categorical_df_test[column1], categorical_df_test[column2])
        chi2, p, dof, expected = chi2_contingency(frequency)
        if p < 0.05:
                results.append((column1, column2, chi2, p))
        chi2_matrix.iloc[i, j+i+1] = chi2
        chi2_matrix.iloc[j+i+1, i] = chi2  
results_df=pd.DataFrame(results, columns=['Column1', 'Column2', 'Chi2', 'p_value'])
significant_relations = results_df.sort_values('p_value')
significant_relations


ship_mode=df_test['Ship Mode'].mode()[0]
df_test['Ship Mode'].replace(['NaN', ' ', ''], np.nan, inplace=True)
df_test['Ship Mode']=df_test['Ship Mode'].fillna(ship_mode)
df_test=df_test.drop(columns='REASON_CD', axis=1)
median_ship_decision=df_test['SHIP DECISION NO'].median()
df_test['SHIP DECISION NO']=df_test['SHIP DECISION NO'].fillna(median_ship_decision)


df_test = df_test.rename(columns={'WEIGHT PER PIECE': 'WEIGHT_PER_PIECE'})
df_test = df_test.rename(columns={'PACK QTY': 'PACK_QTY'})
df_test = df_test.rename(columns={'Stock class': 'Stock_class'})
df_test = df_test.rename(columns={'SUPPLIER INV AMOUNT': 'SUPPLIER_INV_AMOUNT'})
df_test = df_test.rename(columns={'PACKING RANK': 'PACKING_RANK'})
df_test = df_test.rename(columns={'LOGICAL PLANT': 'LOGICAL_PLANT'})
df_test = df_test.rename(columns={'Ship Mode': 'Ship_Mode'})
df_test = df_test.rename(columns={'SHIP DECISION NO': 'SHIP_DECISION_NO'})
df_test = df_test.rename(columns={'PACK QTY': 'PACK_QTY'})
df_test = df_test.rename(columns={'WEIGHT PER PIECE': 'WEIGHT_PER_PIECE'})

outlier={}
numeric_data_df=[col for col in numeric_data_df if col in df_test.columns]
for column in numeric_data_df:
    Q1=df_test[column].quantile(0.25)
    Q3=df_test[column].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers=df_test[(df_test[column]<lower_bound) | (df_test[column]>upper_bound)]
    outlier_percentage=len(outliers)/len(df_test)*100
    outlier[column]={
        'Number Outliers':len(outliers),
        'Outlier Percentage':outlier_percentage
    }
outlier_summary=pd.DataFrame(outlier).T
print(outlier_summary.sort_values('Number Outliers', ascending=False))


for column in numeric_data_df:
    Q1=df_test[column].quantile(0.25)
    Q3=df_test[column].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers=df_test[(df_test[column]<lower_bound) | (df_test[column]>upper_bound)]
    outlier_percentage=len(outliers)/len(df_test)*100
    if outlier_percentage>0.1:
        lower_threshold = df_test[column].quantile(0.05)
        upper_threshold = df_test[column].quantile(0.95)
        df_test[column] = df_test[column].clip(lower=lower_threshold, upper=upper_threshold)
    elif outlier_percentage >0.05 and outlier_percentage<0.1:
        df_test[column]=np.log1p(df_test[column])
print(df_test.head())


scaler=MinMaxScaler() 
numeric_data_df = df_test.select_dtypes(include=['float64', 'int64']).drop(columns=['ID'], errors='ignore').columns.tolist()
df_test[numeric_data_df]=scaler.fit_transform(df_test[numeric_data_df])
print(df_test.head())


categorical_column= ['SUBSIDIARY_CD', 'GLOBAL_NO', 'BRAND_CD', 'INNER_CD', 'SUPPLIER_CD',
                     'PACKING_RANK', 'PRODUCT_CD', 'VSD', 'DELI_DIV', 'Ship_Mode'
]
df_test[categorical_column]=df_test[categorical_column].astype(str)
encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_test[categorical_column]=encoder.fit_transform(df_test[categorical_column])
data_encoded=pd.concat([df_test[categorical_column]], axis=1)
print(data_encoded.head())


df_test = df_test.rename(columns={'Order date': 'Order_date'})
df_test = df_test.rename(columns={'DIRECT SHIP FLG': 'DIRECT_SHIP_FLG'})
df_test = df_test.rename(columns={'Sales order line number': 'Sales_order_line_number'})
df_test = df_test.rename(columns={'SO QTY': 'SO_QTY'})
df_test = df_test.rename(columns={'Consider count hodiday Saturday': 'Consider_count_hodiday_Saturday'})


df_test=df_test.drop(columns=['SPECIAL_DIV','SUBSIDIARY_CD','Stock_class','SHIP_DECISION_NO','PACK_QTY','SPECIAL_DIV'], axis=1)


drop_columns = ['ID', 'Order_date', 'GLOBAL_NO', 'WEIGHT_UNIT','HAZARD_FLG','SPECIFY_SHIP_DAYS','PRODUCT_ASSORT']  
x_pilot = df_test.drop(columns=[col for col in drop_columns if col in df_test.columns], errors='ignore').copy()
for col in x_pilot.select_dtypes(include='object').columns:
    if col in le_dict:
        le = le_dict[col]
        x_pilot[col] = x_pilot[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
    else:
        x_pilot[col] = -1  
for col in train_columns:
    if col not in x_pilot.columns:
        x_pilot[col] = -1
x_pilot = x_pilot[train_columns]  
y_pred_pilot = model.predict(x_pilot).astype(int)
result = pd.DataFrame({
    'ID': df_test['ID'],
    'label': y_pred_pilot
})
result


result.to_csv("submission.csv", index=False)

