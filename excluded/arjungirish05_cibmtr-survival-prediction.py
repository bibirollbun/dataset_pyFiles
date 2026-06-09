import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def load_data(url):
    chunk_size=100000
    chunks=[]
    for chunk in pd.read_csv(url,chunksize=chunk_size):
        int_cols=chunk.select_dtypes(include=['int']).columns
        float_cols=chunk.select_dtypes(include=['float']).columns
        chunk[int_cols]=chunk[int_cols].apply(pd.to_numeric,downcast='integer')
        chunk[float_cols]=chunk[float_cols].apply(pd.to_numeric,downcast='float')
        chunks.append(chunk)

    df=pd.concat(chunks,ignore_index=True)
    return df


train_data=load_data('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_data=load_data('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
data_dictionary=load_data('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
sample_submission=load_data('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')


train_data.head()


test_data.head(10)


data_dictionary


train_data.describe()


train_data.info()


import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = train_data.select_dtypes(include=[np.number]).columns[1:-2]

num_cols = len(numerical_cols)
rows = (num_cols // 5) + 1 
fig, axes = plt.subplots(rows, 5, figsize=(20, 4 * rows))  

# Flatten axes array for easy iteration
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(train_data[col], kde=True, ax=axes[i])
    axes[i].set_title(col)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



train_data.isna().sum()


import os
print(os.listdir("/kaggle/input/"))



!pip install --no-index --find-links=/kaggle/input/lifelines-internetoff/lifelines_and_dependencies lifelines autograd-gamma



from lifelines import KaplanMeierFitter

def transform_survival(df,time_col='efs_time',event_col='efs'):
    kfm=KaplanMeierFitter()
    kfm.fit(df[time_col],df[event_col])
    y=kfm.survival_function_at_times(df[time_col]).values
    return y


train_data['survival']=transform_survival(train_data)


categorical_cols=train_data.select_dtypes(include='object').columns


for col in categorical_cols:
    print(f"col:{train_data[col].value_counts()}")


corrmatrix=train_data[numerical_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corrmatrix,annot=True,cmap='coolwarm',fmt=".2f")
plt.title("Correlation HeatMap")
plt.show()

high_corr_pairs=[(col1,col2) for col1 in numerical_cols for col2 in numerical_cols
                    if col1!=col2 and abs(corrmatrix.loc[col1,col2])>0.9]

print(f"Highly correlated pairs {high_corr_pairs}")


#removing redudant columns
train_data.drop(columns=['hla_high_res_8',
                         'hla_high_res_10',
                        'hla_low_res_8',
                        'hla_low_res_10'],inplace=True)

test_data.drop(columns=['hla_high_res_8',
                         'hla_high_res_10',
                        'hla_low_res_8',
                        'hla_low_res_10'],inplace=True)




numerical_cols=train_data.select_dtypes(include=[np.number]).columns[1:-2]


train_data['dri_score'].value_counts()

def map_dri_score(value):
    value=str(value).strip().lower()
    if value.startswith('intermediate'):
        return 0.5
    elif value.startswith('n/a'):
        return 0.5
    elif value.startswith('high'):
        return 1
    elif value.startswith('low'):
        return 0
    elif value.startswith('missing'):
        return 0.5
    elif value.startswith('very high'):
        return 1
    else:
        return 0.5

train_data['dri_score']=train_data['dri_score'].apply(map_dri_score)
test_data['dri_score']=test_data['dri_score'].apply(map_dri_score)


def binary_coding(value):
    value_map = {
        'No': 0,
        'Negative': 0,
        'Permissive': 0,
        'Fully matched': 0,
        'M-M': 0,
        'F-F': 0,
        'N/A, Mel not given': 0,
        'MEL': 1,
        'Yes': 1,
        'Positive': 1,
        'GvH non-permissive': 0.5,
        'HvG non-permissive': 0.5,
        'M-F': 1,
        'F-M': 1,
        'Intermediate':0.5,
        'Poor':1,
        'TBD':0.5,
        'related':0,
        'unrelated':1,
        'Bi-directional non-permissive':1,
        
        
    }
    return value_map.get(value,0.5)

bin_cols=['psych_disturb',
          'diabetes',
         'arrhythmia',
         'vent_hist',
         'renal_issue',
         'pulm_severe',
         'rituximab',
          'obesity',
          'mrd_hct',
          'in_vivo_tcd',
          'tce_match',
          'hepatic_severe',
          'prior_tumor',
          'peptic_ulcer',
          'rheum_issue',
          'sex_match',
          'hepatic_mild',
          'tce_div_match',
          'donor_related',
          'melphalan_dose',
          'cardiac',
          'pulm_moderate',
          'cyto_score',
          "cyto_score_detail"
         ]
train_data[bin_cols]=train_data[bin_cols].apply(lambda col:col.map(binary_coding))
test_data[bin_cols]=test_data[bin_cols].apply(lambda col: col.map(binary_coding))


train_data.head()


train_data['tbi_status'].value_counts()


#tbi status categorizing
import re
def categorize_tbi(status):
    status=str(status)
    if 'No TBI' in status:
        return 0
    elif re.search(r'<=\d+\s*cGy', status):
        return 1
    elif re.search(r'>\d+\s*cGy', status):
        return 2
    elif 'single' in status.lower():
        return 1
    elif 'fractionated' in status.lower():
        return 2
    else:
        return 1

train_data['tbi_category'] = train_data['tbi_status'].map(categorize_tbi)
test_data['tbi_category']=test_data['tbi_status'].map(categorize_tbi)


train_data['cmv_status'].value_counts()


#cmv status encoding
cmv_mapping = {'+/+': 2, '-/+': 3, '+/-': 3, '-/-': 1}  
train_data['cmv_status_encoded'] = train_data['cmv_status'].map(cmv_mapping)
test_data['cmv_status_encoded']=test_data["cmv_status"].map(cmv_mapping)


#T-cell Epitope Immunological Matching
risk_map={
    'P/P':0,'TBD':0, 'No drugs reported':0, 'N/A':0,
    'P/H':1,'P/B':1,'P/G':1,'NMA':1,
    'G/G':2,'H/H':2,'RIC':2,
    'G/B':3,'H/B':3,'MAC':3

}

train_data['tce_imm_match']=train_data['tce_imm_match'].map(risk_map)
test_data['tce_imm_match']=test_data['tce_imm_match'].map(risk_map)
train_data['conditioning_intensity']=train_data['conditioning_intensity'].map(risk_map)
test_data['conditioning_intensity']=test_data['conditioning_intensity'].map(risk_map)


train_data.drop(columns=['tbi_status','cmv_status'],inplace=True)


cat_cols=list(train_data.select_dtypes(include='object').columns)
num_cols=list(train_data.drop(columns=['efs','efs_time','survival']).select_dtypes(include=[np.number]).columns[1:])
target='survival'


num_cols


#lets handle the missing values using imputer
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="mean")
imputerCat=SimpleImputer(strategy="most_frequent")
imputer.fit(train_data[num_cols])
imputerCat.fit(train_data[cat_cols])
train_data[num_cols]=imputer.transform(train_data[num_cols])
test_data[num_cols]=imputer.transform(test_data[num_cols])
train_data[cat_cols]=imputerCat.transform(train_data[cat_cols])
test_data[cat_cols]=imputerCat.transform(test_data[cat_cols])


#OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
onehot=OneHotEncoder(sparse_output=False,handle_unknown='ignore').fit(train_data[cat_cols])
encoded_cols=onehot.get_feature_names_out(cat_cols)
train_encoded_df=pd.DataFrame(onehot.transform(train_data[cat_cols]),columns=encoded_cols,index=train_data.index)
test_encoded_df=pd.DataFrame(onehot.transform(test_data[cat_cols]),columns=encoded_cols,index=test_data.index)
train_en=pd.concat([train_data[num_cols],train_encoded_df],axis=1)
test_en=pd.concat([test_data[num_cols],test_encoded_df],axis=1)


train_en


train_en.describe()


#baseline model
def baseline(df):
    return np.full(1,len(df))


from sklearn.model_selection import train_test_split
train_inputs,valid_inputs=train_test_split(train_en,test_size=0.2,random_state=42)
target_inputs,valid_target=train_test_split(train_data[target],test_size=0.2,random_state=42)


print(target)


from sklearn.model_selection import KFold,cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.linear_model import LinearRegression

def model_to_stack(): 
    kf=KFold(n_splits=6,shuffle=True)
    base_models = [
    ('lr', LinearRegression()),  
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),  
    ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))  
    ]
    for name,model in base_models:
        scores=cross_val_score(model,train_inputs,target_inputs,cv=kf)
        plt.figure(figsize=(10,7))
        sns.barplot(x=np.arange(1,len(scores)+1),y=scores)
        plt.axhline(scores.mean(),linestyle='--',c='r')
        plt.title(f"performance{name}")
        plt.xlabel("Folds")
        plt.ylabel("Score")
        plt.show()

    stacking_model=StackingRegressor(estimators=base_models,final_estimator=LinearRegression(),cv=5)
    stacking_model.fit(train_inputs,target_inputs)
    return stacking_model.predict(test_en)
    print(f"Final Prediction:{final_prediction}")



final_prediction=model_to_stack()
print(final_prediction)


output=pd.DataFrame({
    'ID':test_data['ID'],
    'prediction':final_prediction
})


output.to_csv("submission.csv",index=False)

