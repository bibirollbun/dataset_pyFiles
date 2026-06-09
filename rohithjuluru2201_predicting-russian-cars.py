# import packages
import numpy as np 
import pandas as pd
import seaborn as sns
import supplemental_english as supplement
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


pd.set_option(('display.max_rows') ,None)
pd.set_option(('display.max_columns') ,None)
pd.option_context('mode.use_inf_as_na', True)


# load data
train=pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test=pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
train.head()


print(train.shape
,test.shape)


train.isnull().sum()


train.info(),test.info()


gov_rows = []
for (letters, (num_from, num_to), region_code), (desc, forbidden, advantage, significance) in supplement.GOVERNMENT_CODES.items():
    gov_rows.append({
        'letters': letters,
        'number_from': num_from,
        'number_to': num_to,
        'region_code': region_code,
        'forbidden_to_buy': bool(forbidden),
        'road_advantage': bool(advantage),
        'significance_level': significance
    })
region_rows = []
for region, codes in supplement.REGION_CODES.items():
    for code in codes:
        region_rows.append({'region_code': code})

df_regions = pd.DataFrame(region_rows)
df_govs = pd.DataFrame(gov_rows)
df_gov = pd.merge(df_govs,on='region_code',how='left',right=df_regions)
df_gov.replace(to_replace=True,value=1,inplace=True)
df_gov.replace(to_replace=False,value=0,inplace=True)
df_gov.shape



new_train=train
new_test=test
new_train.head()


def extract_plate_features(df):
    df = df.copy()
    
    df['plate_str'] = df['plate'].astype(str)
    
    # region code: last 2–3 digits
    
    df['region_code'] = df['plate_str'].str.extract(r'(\d{2,3})$')[0]
    
    # prefix letters: first 1–3 characters
    
    df['prefix'] = df['plate_str'].str.extract(r'^([A-ZА-Я]{1,3})')[0]
    
    # numeric block: three digits
    
    df['number'] = df['plate_str'].str.extract(r'([0-9]{3})')[0]
#     df['mid_prefix'] = df['plate_str'].str.extract(r'^([A-ZА-Я]{2,3})')[0]
    # date parts
    
    df['date'] = pd.to_datetime(df['date'])
    df['year']    = df['date'].dt.year
    df['month']   = df['date'].dt.month
    df['day']     = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    
    
    # government‐plate flag
    
    df['is_gov'] = df['prefix'].isin(supplement.GOVERNMENT_CODES).astype(int)
    
    # numeric region code directly
    
    df['region_num'] = pd.to_numeric(df['region_code'], errors='coerce').fillna(0).astype(int)
    
    
    return df

new_train = extract_plate_features(new_train)
new_test = extract_plate_features(new_test)


new_train.info()


sns.pairplot(new_test)


# find object columns
obj_data=new_train.select_dtypes(include=['O'])
obj_data
obj_data.columns


le_pref = LabelEncoder()
le_reg  = LabelEncoder()
ohe_num = OneHotEncoder()

new_train['pref_enc'] = le_pref.fit_transform(new_train['prefix'])
new_train['plate_enc'] = le_pref.fit_transform(new_train['plate'])
new_train['plate_str_enc'] = le_pref.fit_transform(new_train['plate_str'])
new_train['reg_enc'] = le_reg.fit_transform(new_train['region_code'])
# new_train['letters_enc'] = le_pref.fit_transform(new_train['letters'])


new_test['pref_enc'] = le_pref.fit_transform(new_test['prefix'])
new_test['plate_enc'] = le_pref.fit_transform(new_test['plate'])
new_test['plate_str_enc'] = le_pref.fit_transform(new_test['plate_str'])
new_test['reg_enc'] = le_reg.fit_transform(new_test['region_code'])
# new_test['letters_enc'] = le_pref.fit_transform(new_test['letters'])


new_train.head()


sns.pairplot(new_train)


X = new_train.drop(columns=['plate', 'plate_str', 'region_code', 'prefix','date','price',],axis=1)
y = new_train['price']
test_X = new_test.drop(columns=['plate', 'plate_str', 'region_code', 'prefix','date','price',],axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



st= MinMaxScaler()
st.fit(X_train)


sc_x_train=st.transform(X_train)
sc_x_test=st.transform(X_test)


sns.distplot(sc_x_test)


# Decision Tree
tree_clf = DecisionTreeClassifier(criterion='gini',ccp_alpha=1,splitter='best')
tree_clf.fit(sc_x_train, y_train)



tree_clf.score(sc_x_test,y_test)


predicted=tree_clf.predict(test_X)


predicted


submission = pd.DataFrame({
    'id': test['id'],
    'price': predicted
})
submission.to_csv('/kaggle/working/submission.csv',index=False)

