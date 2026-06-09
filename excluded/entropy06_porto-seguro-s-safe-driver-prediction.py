import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set(style='white',context='notebook',palette='deep')

import warnings
warnings.filterwarnings('ignore')


from sklearn.impute import SimpleImputer

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel

from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier

# 데이터프레임을 출력할 때, 최대 100개의 열까지 표시하도록 설정하는 코드
pd.set_option('display.max_columns',100)


train=pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
test=pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')


# 각 열의 data type 확인 
train.dtypes


test.dtypes


train.head()


train.tail()


train.shape


train.drop_duplicates()
train.shape


test.shape


train.info()


data=[]

for f in train.columns:

    # role 정의
    
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
        
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == float:
        level = 'interval'
    elif train[f].dtype == int:
        level = 'ordinal'

    
    # id만 제외하고 keep을 True로 설정
    # keep은 변수가 분석에서 사용될지 여부를 나타냄
    
    keep=True
    
    if f == 'id':
        keep = False

    # data type 정의 
    dtype = train[f].dtype

    # 메타데이터 딕셔너리 생성
    
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)

# 메타데이터 프레임 생성

meta=pd.DataFrame(data, columns=['varname','role','level','keep','dtype'])
meta.set_index('varname',inplace=True)



meta


meta[(meta.level == 'nominal') & (meta.keep)].index


pd.DataFrame({'count': meta.groupby(['role','level'])
             ['role'].size()}).reset_index()


v = meta[(meta.level == 'interval') & (meta.keep)].index

# train[v] 값에 대한 설명 
train[v].describe()


v = meta[(meta.level =='ordinal') & (meta.keep)].index

train[v].describe()


v = meta[(meta.level == 'binary') & (meta.keep)].index

train[v].describe()


# apriori - 사전 확률

desired_apriori=0.10

idx_0=train[train.target==0].index
idx_1=train[train.target==1].index

nb_0=len(train.loc[idx_0])
nb_1=len(train.loc[idx_1])

# 0 클래스에서 얼만큼 데이터를 줄여야 할지 계산하는 부분 
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)

print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of recrds with target=0 after undersampling: {}'.format(undersampled_nb_0))

# 언더샘플링 수행 (랜덤 선택)
undersampled_idx = shuffle(idx_0, random_state=37, n_samples = undersampled_nb_0)

# 언더샘플링 후 인덱스 결합
idx_list = list(undersampled_idx) +list(idx_1)

# 최종 데이터셋 생성
train = train.loc[idx_list].reset_index(drop=True)


vars_with_missing = []

for f in train.columns:
    missings = train[train[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings / train.shape[0]
        print('Variable {} has {} records ({:.2%}) with missing values'.
             format(f,missings,missings_perc))

print('In total, there are {} variables with missing values'.
      format(len(vars_with_missing)))


# 변수 삭제 
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[(vars_to_drop),'keep'] = False  

# SimpleImputer를 사용해 결측값 대체 
mean_imp = SimpleImputer(missing_values=-1, strategy='mean')
mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')

# SimpleImputer는 데이터를 처리할 때
# 기본적으로 2차원 배열을 반환함 
# 그래서 ravel()을 사용해서 1차원 배열로 변환한 뒤, 
# 다시 원래의 df 열에 할당하는 것
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()


v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    dist_values = train[f].value_counts().shape[0]
    print('Variable {} has {} distinct values'.format(f,dist_values))


# 랜덤 노이즈 추가

def add_noise(series, noise_level):
    return series * (1+noise_level * np.random.randn(len(series)))

# 타겟 인코딩을 구현한 함수
# - 범주형 변수 -> 숫자형으로 바꾸는 기법 
# - 범주형 변수에 대해 타겟 변수의 평균을 사용하여 해당 범주를 인코딩함
# - 여기에 추가적으로 노이즈를 더하는 기능도 포함됨 
# - 모델이 과적합되는 것을 방지하고, 데이터에서 약간의 변화를 더해주는 효과를 줌 

def target_encode(trn_series=None, 
                 tst_series=None,
                 target=None,
                 min_samples_leaf=1,
                 smoothing=1,
                 noise_level=0):
    """
    trn_series : training categorical feature as a pd.Series - 훈련 데이터의 범주형 변수
    tst_series : test categorical feature as a pd.Series - 테스트 데이터의 범주형 변수
    target: target data as a pd.Series - 타겟 변수
    min_samples_leaf (int) : minimum samples to take category average into account - 최소 샘플 수
    -> 특정 범주가 이 값보다 적은 샘플을 가지면, 그 범주의 평균을 덜 신뢰하고 전체 타겟 평균을 더 반영하도록 조정
    smoothing (int) : smopthing effect to balance categorical average ve prior - 스무딩 효과 
    -> 범주의 평균값과 전체 평균 사이의 균형을 조절하는 역할 
    """

    # assert - 단정문 -> 밑의 코드는 둘이 무조건 맞아야 한다.
    #                   아니면 오류 발생
    
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    
    temp = pd.concat([trn_series, target], axis=1)

    # .agg: 집계함수 (aggregation function)
    averages=temp.groupby(by=trn_series.name)[target.name].agg(['mean','count'])

    # 로지스틱 함수 (sigmoid) 를 활용하여,
    # 범주별 데이터 샘플 수와 최소 샘플 수를 비교하고
    # 이를 기반으로 smoothing값을 계산하는 역할을 함 
    smoothing = 1/ (1 + np.exp(-(averages['count'] - min_samples_leaf)/ smoothing))

    # 각 범주의 평균값 조정 
    # 이렇게 하면, 샘플이 적은 범주는 전체 평균을 더 반영 
    prior = target.mean()
    averages[target.name] = prior * (1 - smoothing) + averages['mean'] * smoothing
    averages.drop(['mean','count'],axis=1, inplace=True)

    # 훈련 데이터에 타겟 평균 적용 
    ft_trn_series=pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={
            'index':target.name,target.name:'average'}),
        on = trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    ft_trn_series.index = trn_series.index
    
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={
            'index': target.name,target.name:'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


train_encoded, test_encoded = target_encode(train['ps_car_11_cat'],
                                            test['ps_car_11_cat'],
                                            target=train.target,
                                            min_samples_leaf=100,
                                            smoothing=10,
                                            noise_level=0.01)
train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat','keep'] = False
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat',axis=1, inplace = True)


v = meta[(meta.level =='nominal') & (meta.keep)].index

for f in v:
    plt.figure()
    fig,ax = plt.subplots(figsize=(20,10))
    cat_perc = train[[f,'target']].groupby([f],as_index=False).mean()
    cat_perc.sort_values(by='target',ascending=False,inplace=True)
    sns.barplot(ax=ax,x=f,y='target',data=cat_perc,order=cat_perc[f])
    plt.ylabel('% target',fontsize=18)
    plt.xlabel(f, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()
    


def corr_heatmap(v):
    correlations = train[v].corr()
    cmap = sns.diverging_palette(220,10,as_cmap=True)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0,center=0,fmt='.2f',
               square=True, linewidths=.5,annot=True, cbar_kws={'shrink':.75})
    plt.show()

v = meta[(meta.level =='interval') & (meta.keep)].index
corr_heatmap(v)


s = train.sample(frac=0.1)


sns.lmplot(x='ps_reg_02', y='ps_reg_03',data=s,
          hue='target',palette='Set1',scatter_kws={'alpha':0.3})
plt.show()


sns.lmplot(x='ps_car_12',y='ps_car_13',data=s,
          hue='target',palette='Set1',scatter_kws={'alpha':0.3})
plt.show()


sns.lmplot(x='ps_car_12',y='ps_car_14',data=s,hue='target',
          palette='Set1',scatter_kws={'alpha':0.3})
plt.show()


sns.lmplot(x='ps_car_15',y='ps_car_13',data=s,
          hue='target',palette='Set1',scatter_kws={'alpha':0.3})
plt.show()


v = meta[(meta.level =='ordinal') & (meta.keep)].index
corr_heatmap(v)


v = meta[(meta.level=='nominal') & (meta.keep)].index

print('Before dummification we have {} variables in train'.format(train.shape[1]))

train=pd.get_dummies(train,columns=v,drop_first=True)
print('After dummification we have {} variables in train'.format(train.shape[1]))


v = meta[(meta.level =='interval')&(meta.keep)].index

poly=PolynomialFeatures(degree=2, interaction_only=False,
                       include_bias=False)
interactions =pd.DataFrame(data=poly.fit_transform(train[v]),
                          columns=poly.get_feature_names_out(v))
interactions.drop(v,axis=1,inplace=True)

print('Before creating interactions we have {} variables in train'.format(train.shape[1]))
train=pd.concat([train,interactions],axis=1)

print('After creating interactions we have {} variables in train'.format(train.shape[1]))


selector = VarianceThreshold(threshold=.01)
selector.fit(train.drop(['id','target'],axis=1))

f = np.vectorize(lambda x: not x)

v = train.drop(['id','target'],axis=1).columns[f(selector.get_support())]
print('{} variables have too low variance.'.format(len(v)))
print('These variables are {}'.format(list(v)))


X_train = train.drop(['id','target'],axis=1)
y_train = train['target']

feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators=1000, random_state=0,n_jobs=-1)

rf.fit(X_train,y_train)
importances = rf.feature_importances_

indices = np.argsort(rf.feature_importances_)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1,30,feat_labels[indices[f]],importances[indices[f]]))




sfm = SelectFromModel(rf, threshold='median',prefit=True)
print('Number of features before selection: {}'.format(X_train.shape[1]))

n_features = sfm.transform(X_train).shape[1]
print('Number of features after selection: {}'.format(n_features))
selected_vars=list(feat_labels[sfm.get_support()])




