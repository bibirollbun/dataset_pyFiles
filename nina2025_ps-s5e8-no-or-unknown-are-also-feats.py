#run hv_blend function

%run /kaggle/input/4-august-2025-ps-s5e8/hv_blend.py


import numpy as np
import pandas as pd
import warnings; warnings.simplefilter('ignore')

test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
subm  = pd.read_csv('../input/playground-series-s5e8/sample_submission.csv')


# micro-EDA:
display(train.head(11))
print('info: \n')
print(f"{train.info(show_counts=True)}")
cat_feats = test.select_dtypes(include="object").columns
num_feats = test.select_dtypes(include="int64") .columns
print('\ncategorical uniques:\n')
def tt(n): return "\t\t" if len(n) <= 5 else '\t'
for cat_col in cat_feats[1:]: 
    lu = test[cat_col].unique()
    print(f"{cat_col}:{tt(cat_col)}{len(lu)}\t{lu}")
print(f"{'job'}:{tt('job')}12\t{test['job'].unique()}")


def micro_fe(df):
    
    df = df.copy()
    
    def f2(x):
        if x['default']=='no' and x['housing']=='no' and x['loan']=='no':
            return 21
        if x['default']=='no' and x['housing']=='no'\
        or x['default']=='no' and x['loan']=='no'\
        or x['housing']=='no' and x['loan']=='no':
            return 7
        if x['default']=='no' or x['housing']=='no' or x['loan']=='no':
            return 3
        return 0
        
    def f1(x):
        if x['education']=='unknown' and x['contact'] =='unknown' and x['poutcome']=='unknown':
            return 21
        if x['education']=='unknown' and x['contact'] =='unknown'\
        or x['education']=='unknown' and x['poutcome']=='unknown'\
        or x['contact']  =='unknown' and x['poutcome']=='unknown':
            return 7
        if x['education']=='unknown' or x['contact']=='unknown' or x['poutcome']=='unknown':
            return 3
        return 0
    
    df['unknowns'] = df.apply(lambda x: f1(x), axis=1)
    df['many_no']  = df.apply(lambda x: f2(x), axis=1)
    
    #df['log_dur']  = np.log1p(df['duration'])
    #df['log_age']  = np.log1p(df['age'])

    return df


def fen(df):
    marital_map   = {'married':1.01, 'single':1.02, 'divorced':1.03}
    education_map =	{'secondary':2.01, 'tertiary':2.02, 'primary':2.03, 'unknown':2.04}
    default_map   =	{'no':3.01, 'yes':3.02}
    housing_map   =	{'no':4.01, 'yes':4.02}
    loan_map      =	{'no':5.01, 'yes':5.02}
    contact_map   = {'unknown':6.01, 'cellular':6.02, 'telephone':6.03}
    month_map     = {'jan': 7.01, 'feb': 7.02, 'mar': 7.03, 'apr': 7.04, 'may': 7.05, 'jun': 7.06,
                     'jul': 7.07, 'aug': 7.08, 'sep': 7.09, 'oct': 7.10, 'nov': 7.11, 'dec': 7.12}
    poutcome_map  =	{'unknown':8.01, 'other':8.02, 'failure':8.03, 'success':8.04}
    job_map       = {'blue-collar':9.01, 'management':9.02, 'self-employed':9.03, 'technician':9.04,
                     'services':9.05,'retired':9.06, 'entrepreneur':9.07, 'admin.':9.08,
                     'housemaid':9.09, 'unemployed':9.10, 'student':9.11,'unknown':9.12}
    
    df['x1']      = df['marital']  .map(marital_map)
    df['x2']      = df['education'].map(education_map)
    df['x3']      = df['default']  .map(default_map)
    df['x4']      = df['housing']  .map(housing_map)
    df['x5']      = df['loan']     .map(loan_map)
    df['x6']      = df['contact']  .map(contact_map)
    df['x7']      = df['month']    .map(month_map)
    df['x8']      = df['poutcome'] .map(poutcome_map)
    df['x9']      = df['job']      .map(job_map)
    # df['x10']     = df['unknowns']
    # df['x11']     = df['many_no']
    # df['x12']     = df['log_dur']
    # df['x13']     = df['log_age']
    # df['x14']     = df['age']
    # df['x15']     = df['balance']
    # df['x16']     = df['day']
    # df['x17']     = df['duration']
    # df['x18']     = df['campaign']
    # df['x19']     = df['pdays']
    # df['x20']     = df['previous']

    import polars as pl

    polar_df = pl.from_pandas(df)
    polar_df = polar_df.with_columns(
        # _2_1 = ((pl.col('x1')-pl.col('x3'))**2+(pl.col('x2')-pl.col('x4'))**2).sqrt(),
        # _2_2 = ((pl.col('x1')-pl.col('x5'))**2+(pl.col('x2')-pl.col('x6'))**2).sqrt(),
        # _2_3 = ((pl.col('x1')-pl.col('x7'))**2+(pl.col('x2')-pl.col('x8'))**2).sqrt(),
        # _2_4 = ((pl.col('x3')-pl.col('x5'))**2+(pl.col('x4')-pl.col('x6'))**2).sqrt(),
        # _2_5 = ((pl.col('x3')-pl.col('x7'))**2+(pl.col('x4')-pl.col('x8'))**2).sqrt(),
        # _2_6 = ((pl.col('x5')-pl.col('x7'))**2+(pl.col('x6')-pl.col('x8'))**2).sqrt(),
        # _3_1 = ((pl.col('x1')-pl.col('x4'))**2+(pl.col('x2')-pl.col('x5'))**2+(pl.col('x3')-pl.col('x6'))**2).sqrt(),
        # _3_2 = ((pl.col('x1')-pl.col('x7'))**2+(pl.col('x2')-pl.col('x8'))**2+(pl.col('x3')-pl.col('x9'))**2).sqrt(),
        # _3_3 = ((pl.col('x4')-pl.col('x7'))**2+(pl.col('x5')-pl.col('x8'))**2+(pl.col('x6')-pl.col('x9'))**2).sqrt(),
        # _4_1 = ((pl.col('x1')-pl.col('x5'))**2+(pl.col('x2')-pl.col('x6'))**2+(pl.col('x3')-pl.col('x7'))**2+(pl.col('x4')-pl.col('x8'))**2).sqrt(),
        # _2_1 = ((pl.col('x1')-pl.col('x3'))**2+(pl.col('x2')-pl.col('x4'))**2).sqrt(),
        # _2_2 = ((pl.col('x1')-pl.col('x5'))**2+(pl.col('x2')-pl.col('x6'))**2).sqrt(),
        # _2_3 = ((pl.col('x1')-pl.col('x7'))**2+(pl.col('x2')-pl.col('x8'))**2).sqrt(),
        # _2_4 = ((pl.col('x1')-pl.col('x9'))**2+(pl.col('x2')-pl.col('x10'))**2).sqrt(),
        # _2_5 = ((pl.col('x1')-pl.col('x11'))**2+(pl.col('x2')-pl.col('x12'))**2).sqrt(),
        # _2_6 = ((pl.col('x1')-pl.col('x13'))**2+(pl.col('x2')-pl.col('x14'))**2).sqrt(),
        # _2_7 = ((pl.col('x1')-pl.col('x15'))**2+(pl.col('x2')-pl.col('x16'))**2).sqrt(),
        # _2_8 = ((pl.col('x1')-pl.col('x17'))**2+(pl.col('x2')-pl.col('x18'))**2).sqrt(),
        # _2_9 = ((pl.col('x1')-pl.col('x19'))**2+(pl.col('x2')-pl.col('x20'))**2).sqrt(),
        # _2_10 = ((pl.col('x3')-pl.col('x5'))**2+(pl.col('x4')-pl.col('x6'))**2).sqrt(),
        # _2_11 = ((pl.col('x3')-pl.col('x7'))**2+(pl.col('x4')-pl.col('x8'))**2).sqrt(),
        # _2_12 = ((pl.col('x3')-pl.col('x9'))**2+(pl.col('x4')-pl.col('x10'))**2).sqrt(),
        # _2_13 = ((pl.col('x3')-pl.col('x11'))**2+(pl.col('x4')-pl.col('x12'))**2).sqrt(),
        # _2_14 = ((pl.col('x3')-pl.col('x13'))**2+(pl.col('x4')-pl.col('x14'))**2).sqrt(),
        # _2_15 = ((pl.col('x3')-pl.col('x15'))**2+(pl.col('x4')-pl.col('x16'))**2).sqrt(),
        # _2_16 = ((pl.col('x3')-pl.col('x17'))**2+(pl.col('x4')-pl.col('x18'))**2).sqrt(),
        # _2_17 = ((pl.col('x3')-pl.col('x19'))**2+(pl.col('x4')-pl.col('x20'))**2).sqrt(),
        # _2_18 = ((pl.col('x5')-pl.col('x7'))**2+(pl.col('x6')-pl.col('x8'))**2).sqrt(),
        # _2_19 = ((pl.col('x5')-pl.col('x9'))**2+(pl.col('x6')-pl.col('x10'))**2).sqrt(),
        # _2_20 = ((pl.col('x5')-pl.col('x11'))**2+(pl.col('x6')-pl.col('x12'))**2).sqrt(),
        # _2_21 = ((pl.col('x5')-pl.col('x13'))**2+(pl.col('x6')-pl.col('x14'))**2).sqrt(),
        # _2_22 = ((pl.col('x5')-pl.col('x15'))**2+(pl.col('x6')-pl.col('x16'))**2).sqrt(),
        # _2_23 = ((pl.col('x5')-pl.col('x17'))**2+(pl.col('x6')-pl.col('x18'))**2).sqrt(),
        # _2_24 = ((pl.col('x5')-pl.col('x19'))**2+(pl.col('x6')-pl.col('x20'))**2).sqrt(),
        # _2_25 = ((pl.col('x7')-pl.col('x9'))**2+(pl.col('x8')-pl.col('x10'))**2).sqrt(),
        # _2_26 = ((pl.col('x7')-pl.col('x11'))**2+(pl.col('x8')-pl.col('x12'))**2).sqrt(),
        # _2_27 = ((pl.col('x7')-pl.col('x13'))**2+(pl.col('x8')-pl.col('x14'))**2).sqrt(),
        # _2_28 = ((pl.col('x7')-pl.col('x15'))**2+(pl.col('x8')-pl.col('x16'))**2).sqrt(),
        # _2_29 = ((pl.col('x7')-pl.col('x17'))**2+(pl.col('x8')-pl.col('x18'))**2).sqrt(),
        # _2_30 = ((pl.col('x7')-pl.col('x19'))**2+(pl.col('x8')-pl.col('x20'))**2).sqrt(),
        # _2_31 = ((pl.col('x9')-pl.col('x11'))**2+(pl.col('x10')-pl.col('x12'))**2).sqrt(),
        # _2_32 = ((pl.col('x9')-pl.col('x13'))**2+(pl.col('x10')-pl.col('x14'))**2).sqrt(),
        # _2_33 = ((pl.col('x9')-pl.col('x15'))**2+(pl.col('x10')-pl.col('x16'))**2).sqrt(),
        # _2_34 = ((pl.col('x9')-pl.col('x17'))**2+(pl.col('x10')-pl.col('x18'))**2).sqrt(),
        # _2_35 = ((pl.col('x9')-pl.col('x19'))**2+(pl.col('x10')-pl.col('x20'))**2).sqrt(),
        # _2_36 = ((pl.col('x11')-pl.col('x13'))**2+(pl.col('x12')-pl.col('x14'))**2).sqrt(),
        # _2_37 = ((pl.col('x11')-pl.col('x15'))**2+(pl.col('x12')-pl.col('x16'))**2).sqrt(),
        # _2_38 = ((pl.col('x11')-pl.col('x17'))**2+(pl.col('x12')-pl.col('x18'))**2).sqrt(),
        # _2_39 = ((pl.col('x11')-pl.col('x19'))**2+(pl.col('x12')-pl.col('x20'))**2).sqrt(),
        # _2_40 = ((pl.col('x13')-pl.col('x15'))**2+(pl.col('x14')-pl.col('x16'))**2).sqrt(),
        # _2_41 = ((pl.col('x13')-pl.col('x17'))**2+(pl.col('x14')-pl.col('x18'))**2).sqrt(),
        # _2_42 = ((pl.col('x13')-pl.col('x19'))**2+(pl.col('x14')-pl.col('x20'))**2).sqrt(),
        # _2_43 = ((pl.col('x15')-pl.col('x17'))**2+(pl.col('x16')-pl.col('x18'))**2).sqrt(),
        # _2_44 = ((pl.col('x15')-pl.col('x19'))**2+(pl.col('x16')-pl.col('x20'))**2).sqrt(),
        # _2_45 = ((pl.col('x17')-pl.col('x19'))**2+(pl.col('x18')-pl.col('x20'))**2).sqrt(),
        # _3_1 = ((pl.col('x1')-pl.col('x4'))**2+(pl.col('x2')-pl.col('x5'))**2+(pl.col('x3')-pl.col('x6'))**2).sqrt(),
        # _3_2 = ((pl.col('x1')-pl.col('x7'))**2+(pl.col('x2')-pl.col('x8'))**2+(pl.col('x3')-pl.col('x9'))**2).sqrt(),
        # _3_3 = ((pl.col('x1')-pl.col('x10'))**2+(pl.col('x2')-pl.col('x11'))**2+(pl.col('x3')-pl.col('x12'))**2).sqrt(),
        # _3_4 = ((pl.col('x1')-pl.col('x13'))**2+(pl.col('x2')-pl.col('x14'))**2+(pl.col('x3')-pl.col('x15'))**2).sqrt(),
        # _3_5 = ((pl.col('x1')-pl.col('x16'))**2+(pl.col('x2')-pl.col('x17'))**2+(pl.col('x3')-pl.col('x18'))**2).sqrt(),
        # _3_6 = ((pl.col('x4')-pl.col('x7'))**2+(pl.col('x5')-pl.col('x8'))**2+(pl.col('x6')-pl.col('x9'))**2).sqrt(),
        # _3_7 = ((pl.col('x4')-pl.col('x10'))**2+(pl.col('x5')-pl.col('x11'))**2+(pl.col('x6')-pl.col('x12'))**2).sqrt(),
        # _3_8 = ((pl.col('x4')-pl.col('x13'))**2+(pl.col('x5')-pl.col('x14'))**2+(pl.col('x6')-pl.col('x15'))**2).sqrt(),
        # _3_9 = ((pl.col('x4')-pl.col('x16'))**2+(pl.col('x5')-pl.col('x17'))**2+(pl.col('x6')-pl.col('x18'))**2).sqrt(),
        # _3_10 = ((pl.col('x7')-pl.col('x10'))**2+(pl.col('x8')-pl.col('x11'))**2+(pl.col('x9')-pl.col('x12'))**2).sqrt(),
        # _3_11 = ((pl.col('x7')-pl.col('x13'))**2+(pl.col('x8')-pl.col('x14'))**2+(pl.col('x9')-pl.col('x15'))**2).sqrt(),
        # _3_12 = ((pl.col('x7')-pl.col('x16'))**2+(pl.col('x8')-pl.col('x17'))**2+(pl.col('x9')-pl.col('x18'))**2).sqrt(),
        # _3_13 = ((pl.col('x10')-pl.col('x13'))**2+(pl.col('x11')-pl.col('x14'))**2+(pl.col('x12')-pl.col('x15'))**2).sqrt(),
        # _3_14 = ((pl.col('x10')-pl.col('x16'))**2+(pl.col('x11')-pl.col('x17'))**2+(pl.col('x12')-pl.col('x18'))**2).sqrt(),
        # _3_15 = ((pl.col('x13')-pl.col('x16'))**2+(pl.col('x14')-pl.col('x17'))**2+(pl.col('x15')-pl.col('x18'))**2).sqrt(),
        # _4_1 = ((pl.col('x1')-pl.col('x5'))**2+(pl.col('x2')-pl.col('x6'))**2+(pl.col('x3')-pl.col('x7'))**2+(pl.col('x4')-pl.col('x8'))**2).sqrt(),
        # _4_2 = ((pl.col('x1')-pl.col('x9'))**2+(pl.col('x2')-pl.col('x10'))**2+(pl.col('x3')-pl.col('x11'))**2+(pl.col('x4')-pl.col('x12'))**2).sqrt(),
        # _4_3 = ((pl.col('x1')-pl.col('x13'))**2+(pl.col('x2')-pl.col('x14'))**2+(pl.col('x3')-pl.col('x15'))**2+(pl.col('x4')-pl.col('x16'))**2).sqrt(),
        # _4_4 = ((pl.col('x1')-pl.col('x17'))**2+(pl.col('x2')-pl.col('x18'))**2+(pl.col('x3')-pl.col('x19'))**2+(pl.col('x4')-pl.col('x20'))**2).sqrt(),
        # _4_5 = ((pl.col('x5')-pl.col('x9'))**2+(pl.col('x6')-pl.col('x10'))**2+(pl.col('x7')-pl.col('x11'))**2+(pl.col('x8')-pl.col('x12'))**2).sqrt(),
        # _4_6 = ((pl.col('x5')-pl.col('x13'))**2+(pl.col('x6')-pl.col('x14'))**2+(pl.col('x7')-pl.col('x15'))**2+(pl.col('x8')-pl.col('x16'))**2).sqrt(),
        # _4_7 = ((pl.col('x5')-pl.col('x17'))**2+(pl.col('x6')-pl.col('x18'))**2+(pl.col('x7')-pl.col('x19'))**2+(pl.col('x8')-pl.col('x20'))**2).sqrt(),
        # _4_8 = ((pl.col('x9')-pl.col('x13'))**2+(pl.col('x10')-pl.col('x14'))**2+(pl.col('x11')-pl.col('x15'))**2+(pl.col('x12')-pl.col('x16'))**2).sqrt(),
        # _4_9 = ((pl.col('x9')-pl.col('x17'))**2+(pl.col('x10')-pl.col('x18'))**2+(pl.col('x11')-pl.col('x19'))**2+(pl.col('x12')-pl.col('x20'))**2).sqrt(),
        # _4_10 = ((pl.col('x13')-pl.col('x17'))**2+(pl.col('x14')-pl.col('x18'))**2+(pl.col('x15')-pl.col('x19'))**2+(pl.col('x16')-pl.col('x20'))**2).sqrt(),
        # _5_1 = ((pl.col('x1')-pl.col('x6'))**2+(pl.col('x2')-pl.col('x7'))**2+(pl.col('x3')-pl.col('x8'))**2+(pl.col('x4')-pl.col('x9'))**2+(pl.col('x5')-pl.col('x10'))**2).sqrt(),
        # _5_2 = ((pl.col('x1')-pl.col('x11'))**2+(pl.col('x2')-pl.col('x12'))**2+(pl.col('x3')-pl.col('x13'))**2+(pl.col('x4')-pl.col('x14'))**2+(pl.col('x5')-pl.col('x15'))**2).sqrt(),
        # _5_3 = ((pl.col('x1')-pl.col('x16'))**2+(pl.col('x2')-pl.col('x17'))**2+(pl.col('x3')-pl.col('x18'))**2+(pl.col('x4')-pl.col('x19'))**2+(pl.col('x5')-pl.col('x20'))**2).sqrt(),
        # _5_4 = ((pl.col('x6')-pl.col('x11'))**2+(pl.col('x7')-pl.col('x12'))**2+(pl.col('x8')-pl.col('x13'))**2+(pl.col('x9')-pl.col('x14'))**2+(pl.col('x10')-pl.col('x15'))**2).sqrt(),
        # _5_5 = ((pl.col('x6')-pl.col('x16'))**2+(pl.col('x7')-pl.col('x17'))**2+(pl.col('x8')-pl.col('x18'))**2+(pl.col('x9')-pl.col('x19'))**2+(pl.col('x10')-pl.col('x20'))**2).sqrt(),
        # _5_6 = ((pl.col('x11')-pl.col('x16'))**2+(pl.col('x12')-pl.col('x17'))**2+(pl.col('x13')-pl.col('x18'))**2+(pl.col('x14')-pl.col('x19'))**2+(pl.col('x15')-pl.col('x20'))**2).sqrt(),
        )
    
    df_pandas = polar_df.to_pandas()

    for i in range(1,10): del df_pandas[f'x{i}']
        
    return df_pandas


%%time
train = micro_fe (train)
test  = micro_fe (test)


X = train.drop(["y", "id"], axis=1)
y = train["y"]
X_test = test.drop(["id"], axis=1)


from sklearn.preprocessing import LabelEncoder

for col_name in cat_feats:
    le = LabelEncoder()
    X[col_name] = le.fit_transform(X[col_name])
    X_test[col_name] = le.transform(X_test[col_name])


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
y_probs = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=20000,
        learning_rate=0.06,
        num_leaves=100,
        max_depth=10,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.79,
        reg_lambda=3.0,
        max_bin=4523,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    y_probs += model.predict_proba(X_test)[:, 1] / n_splits


submission = pd.DataFrame({"id": test["id"], "y": y_probs})
submission.to_csv("submission no_data.csv", index=False)


if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind='barh')


'''
five straight blends:

1.                  0.97538  + 0.97539  -> x1_data
2.                  0.975941 + 0.975942 -> x2_data
3.                  no_data  + 0.97677  -> x3_data
4.                  0.97642  + 0.96648  -> x4_data                                  
5.                  0.97681  + 0.97689  -> x5_data
'''


%%time

path3 = '/kaggle/input/3-august-2025-ps-s5e8/'
path7 = '/kaggle/input/7-august-2025-ps-s5e8/'

subm_x11 = pd.read_csv(path3 + 'submission 0.97538.csv' )
subm_x12 = pd.read_csv(path3 + 'submission 0.97539.csv' )
subm_x21 = pd.read_csv(path3 + 'submission 0.975941.csv')
subm_x22 = pd.read_csv(path3 + 'submission 0.975942.csv')
subm_x31 = pd.read_csv(        'submission no_data.csv' )
subm_x32 = pd.read_csv(path7 + 'submission 0.97677.csv' )
subm_x41 = pd.read_csv(path3 + 'submission 0.97642.csv' )
subm_x42 = pd.read_csv(path3 + 'submission 0.97648.csv' )
subm_x51 = pd.read_csv(path3 + 'submission 0.97681.csv' )
subm_x52 = pd.read_csv(path3 + 'submission 0.97689.csv' )

def straight_blend(df1,df2, file_name, wts=[0.50,0.50]):
    df1['y'] = df1['y'] *wts[0] + wts[1]* df2['y']
    df1.to_csv(file_name, index=False)
    return df1

straight_blend(subm_x11, subm_x12, 'x1_data.csv')
straight_blend(subm_x21, subm_x22, 'x2_data.csv')
straight_blend(subm_x31, subm_x32, 'x3_data.csv',wts=[0.15,0.85])
straight_blend(subm_x41, subm_x42, 'x4_data.csv')
straight_blend(subm_x51, subm_x52, 'x5_data.csv')


path ='/kaggle/working/'

file_short_names = ['x1_data','x2_data','x3_data','x4_data','x5_data']


params = {
      'path'   : path,
      'id'     : 'id',
      'target' : "y",
      'desc'   : 0.70,
      'asc'    : 0.30,
      'subwts' : [+0.09, +0.03, -0.01, -0.04, -0.07],        # Lb = ?
      'subm'   : [
         { 'name':file_short_names[0],'weight':0.03, },
         { 'name':file_short_names[1],'weight':0.04, },
         { 'name':file_short_names[2],'weight':0.05, },
         { 'name':file_short_names[3],'weight':0.07, },
         { 'name':file_short_names[4],'weight':0.81, },
      ]
    }


df = hv_blend ( path, file_short_names, params )

df.to_csv('submission.csv', index=False)

display(df)

