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







import pandas as pd

app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv') #å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ï¼ˆæ­£è§£ãƒ©ãƒ™ãƒ«ã�‚ã‚Šï¼‰

print(app_train.shape)#å‰�å‡¦ç�†ã�®ã�Ÿã‚�ã�«ã€�å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ã�«ã�©ã‚Œã��ã‚‰ã�„ã�®æƒ…å ±ã�Œã�‚ã‚‹ã�‹ç¢ºèª�ï¼ˆè¡Œã�¨åˆ—ï¼‰

app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')#ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ï¼ˆæ­£è§£ãƒ©ãƒ™ãƒ«ç„¡ï¼‰

print(app_test.shape)#å‰�å‡¦ç�†ã�®ã�Ÿã‚�ã�«ã€�ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�«ã�©ã‚Œã��ã‚‰ã�„ã�®æƒ…å ±ã�Œã�‚ã‚‹ã�‹ç¢ºèª�ï¼ˆè¡Œã�¨åˆ—ï¼‰

app_train.head()#å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ã�®åˆ—ã�®è©³ç´°ã‚’ï¼•è¡Œã� ã�‘ç¢ºèª�


app_train['TARGET'].value_counts()#å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ã�®TARGETåˆ—ã�®å‡ºç�¾å›�æ•°ã‚’èª¿ã�¹ã‚‹



app_train['TARGET'].plot.hist()#ãƒ’ã‚¹ãƒˆã‚°ãƒ©ãƒ è¡¨ç¤ºé–¢æ•°ã�§è¡¨ç¤º


app_train.isnull()#å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ã�®ã�†ã�¡æ¬ æ��ãƒ‡ãƒ¼ã‚¿ã�Œã�‚ã‚‹ã�ªã‚‰ï¼ˆTrueï¼‰ã€�ã�ªã�„ã�ªã‚‰ï¼ˆFalseï¼‰ã�¨è¡¨ç¤º


app_train.isnull().sum()#å�„åˆ—ã�®æ¬ æ��ãƒ‡ãƒ¼ã‚¿ã�®ç·�æ•°ã‚’è¡¨ç¤º


app_train.isnull().sum()/len(app_train)#len(app_train)ã�¯å�„ãƒ‡ãƒ¼ã‚¿ã�®è¡Œæ•°


(app_train.isnull().sum()/len(app_train)).sort_values(ascending=False).head(20)


app_train.select_dtypes('object').apply(pd.Series.nunique,axis=0)#æ–‡å­—ï¼ˆobjectå�‹ï¼‰ã�®åˆ—ã� ã�‘ã‚’æŠ½å‡ºã�—ã€�ã�•ã‚‰ã�«nuniqueï¼ˆé‡�è¤‡ã�—ã�ªã�„ï¼‰ã‚‚ã�®ã‚’é�¸ã�¶


from sklearn.preprocessing import LabelEncoder

"""
ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°
"""

le = LabelEncoder()#ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ€ãƒ¼ã�®ã‚¤ãƒ³ã‚¹ã‚¿ãƒ³ã‚¹ã‚’å¤‰æ•°leã�«æ ¼ç´�
le_count=0#ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‰ã�—ã�Ÿåˆ—æ•°ã‚’æ•°ã�ˆã‚‹ã�Ÿã‚�ã�®ã‚«ã‚¦ãƒ³ã‚¿å¤‰æ•°

for col in app_train.columns:#å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ã�®åˆ—å��ä¸€è¦§ã‚’é †ç•ªã�«å�–ã‚Šå‡ºã�—ã€�å�„ãƒ«ãƒ¼ãƒ—ã�®è¦�ç´ ã‚’colã�«ä»£å…¥
        if app_train[col].dtype == 'object':#å�„åˆ—ã�®dtype(ã�¤ã�¾ã‚Šãƒ‡ãƒ¼ã‚¿å�‹)å±�æ€§ã�Œã‚«ãƒ†ã‚´ãƒªå¤‰æ•°ï¼ˆobject)ã�ªã‚‰å‡¦ç�†ã‚’å§‹ã‚�ã‚‹
            if app_train[col].nunique() <= 2:#ã�•ã‚‰ã�«åˆ—ã�®ãƒ¦ãƒ‹ãƒ¼ã‚¯å€¤ï¼ˆé‡�è¤‡ã�—ã�ªã�„ï¼‰ã‚«ãƒ†ã‚´ãƒªå¤‰æ•°ã�®æ•°ã�Œï¼’ä»¥ä¸‹ã�ªã‚‰ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°ã‚’é–‹å§‹
                le.fit(app_train[col])#ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ€ãƒ¼ã�«å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ï¼ˆä¾‹ã�ˆã�°['Y','N']ã�¯[0,1]ã�«å¯¾å¿œ
                app_train[col]= le.transform(app_train[col])#å­¦ç¿’ã�—ã�Ÿã‚¨ãƒ³ã‚³ãƒ¼ãƒ€ã‚’ä½¿ã�„å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ã�¨ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®ä¸¡æ–¹ã�«ã�Šã�‘ã‚‹è©²å½“åˆ—ã‚’æ•´æ•°ãƒ©ãƒ™ãƒ«ã�«ç½®ã��æ�›ã�ˆã�Ÿ
                app_test[col] = le.transform(app_test[col])
                le_count += 1
                print(f'{le_count}åˆ—ã‚’ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°ã�—ã�¾ã�—ã�Ÿ')


"""
ãƒ¯ãƒ³ãƒ›ãƒƒãƒˆã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°
"""
app_train = pd.get_dummies(app_train)#pandasãƒ©ã‚¤ãƒ–ãƒ©ãƒªã�®get_dummiesé–¢æ•°ã‚’ä½¿ç”¨ã�™ã‚‹ã�“ã�¨ã�§è‡ªå‹•çš„ã�«ãƒ‡ãƒ¼ã‚¿ã�®ã‚«ãƒ†ã‚´ãƒªå¤‰æ•°åˆ—ã‚’è‡ªå‹•çš„ã�«ãƒ¯ãƒ³ãƒ›ãƒƒãƒˆåŒ–ã�™ã‚‹
app_test = pd.get_dummies(app_test)




train_labels = app_train['TARGET']#å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ã�®TARGETåˆ—ã‚’å¤‰æ•°ã�«æ ¼ç´�ã�—ã€�align()é–¢æ•°ã�§å‰Šé™¤ã�•ã‚Œã�ªã�„ã‚ˆã�†ã�«ä¸€æ™‚çš„ã�«é�¿é›£ã�•ã�›ã‚‹

app_train,app_test = app_train.align(app_test,join='inner',axis=1)#align()é–¢æ•°ã�§äºŒã�¤ã�®ãƒ‡ãƒ¼ã‚¿ã�®åˆ—ã‚’ã��ã‚�ã�ˆã€�ä¸¡æ–¹ã�«å…±é€šã�™ã‚‹åˆ—ã� ã�‘æ®‹ã�™

app_train['TARGET'] = train_labels#align()é–¢æ•°ã�§å‡¦ç�†ã�—ã�Ÿãƒ‡ãƒ¼ã‚¿ã�«æœ€åˆ�ã�«é�¿é›£ã�•ã�›ã�ŸTARGETåˆ—ã‚’æˆ»ã�™




correlations = app_train.corr()['TARGET'].sort_values()#TARGETã�¨å�„ç‰¹å¾´é‡�ï¼ˆåˆ—ï¼‰ã�®ç›¸é–¢ä¿‚æ•°ã‚’å°�ã�•ã�„é †ã�«ä¸¦ã�¹ã‚‹

print('è² ã�®ç›¸é–¢ã�Œé«˜ã�„ç‰¹å¾´é‡�:\n',correlations.head(15))

print('æ­£ã�®ç›¸é–¢ã�Œé«˜ã�„ç‰¹å¾´é‡�:\n',correlations.tail(15))



app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])


app_train['DAYS_BIRTH'].corr(app_train['TARGET'])


import matplotlib.pyplot as plt #matplotlib.pyplot ãƒ¢ã‚¸ãƒ¥ãƒ¼ãƒ«ã‚’çŸ­ç¸®ã�—ã�¦ plt ã�¨ã�„ã�†å��å‰�ã�«ã�™ã‚‹
plt.style.use('fivethirtyeight')#ã‚°ãƒ©ãƒ•ã�®ã‚¹ã‚¿ã‚¤ãƒ«ã‚’æ±ºå®š

plt.hist(app_train['DAYS_BIRTH']/365,edgecolor = 'k',bins = 25)#plt.hist(...)ï¼šãƒ’ã‚¹ãƒˆã‚°ãƒ©ãƒ ï¼ˆåº¦æ•°åˆ†å¸ƒå›³ï¼‰ã‚’æ��ã��é–¢æ•°ã€‚app_train['DAYS_BIRTH']ã�§å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ã�‹ã‚‰DAYS_BIRTHã‚’å�–ã‚Šå‡ºã�—ã€�365ã�§ã��ã�®æ—¥æ•°ã‚’å‰²ã�£ã�¦å¹´æ•°ã�«å¤‰æ›´ã�™ã‚‹ã€‚edgecoloreã�§æ� ã‚’é»’ã�«ã€�binsã�§åŒºåˆ‡ã‚‹æ•°ã‚’æ±ºã‚�ã‚‹

plt.xlabel('Age')
plt.ylabel('Count')

plt.show()


import matplotlib.pyplot as plt #matplotlib.pyplot ãƒ¢ã‚¸ãƒ¥ãƒ¼ãƒ«ã‚’çŸ­ç¸®ã�—ã�¦ plt ã�¨ã�„ã�†å��å‰�ã�«ã�™ã‚‹
import seaborn as sns #seaborn ãƒ¢ã‚¸ãƒ¥ãƒ¼ãƒ«ã‚’çŸ­ç¸®ã�—ã�¦ sns ã�¨ã�„ã�†å��å‰�ã�«ã�™ã‚‹
#ãƒ­ãƒ¼ãƒ³ã‚’è¿”æ¸ˆã�—ã�Ÿäººï¼ˆTARGET=0ï¼‰ã�¨è¿”æ¸ˆã�—ã�ªã�‹ã�£ã�Ÿäººï¼ˆTARGET=1ï¼‰ã�®å¹´é½¢åˆ†å¸ƒã‚’KDEï¼ˆæ»‘ã‚‰ã�‹ã�ªç¢ºç�‡å¯†åº¦ï¼‰ã�§æ��ã��ã€�è»¸ãƒ©ãƒ™ãƒ«ã�¨ã‚¿ã‚¤ãƒˆãƒ«ã‚’è¨­å®šã�™ã‚‹ã€‚

plt.figure(figsize = (10,8))#KDEã‚°ãƒ©ãƒ•ã�®ã‚µã‚¤ã‚ºã‚’æŒ‡å®š

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0,'DAYS_BIRTH'] / 365 ,label = 'target == 0')#DataFrame.loc[è¡Œã�®æŒ‡å®š, åˆ—ã�®æŒ‡å®š]ã�¨ã�„ã�†æ§‹æ–‡ã‚’å�–ã‚‹ã€‚ã�“ã�“ã�§ã�¯è¿”æ¸ˆè€…ï¼ˆï¼�ï¼‰ã�®DAYS_BIRTHã‚’å�–ã‚Šå‡ºã�—ã€�ã��ã‚Œã‚’365æ—¥ã�§å‰²ã‚‹ã�“ã�¨ã�§å¹´é½¢ã‚’ç®—å‡ºã�—ã�¦ã�„ã‚‹

sns.kdeplot(app_train.loc[app_train['TARGET'] == 1,'DAYS_BIRTH'] / 365 ,label = 'target == 1')#DataFrame.loc[è¡Œã�®æŒ‡å®š, åˆ—ã�®æŒ‡å®š]ã�¨ã�„ã�†æ§‹æ–‡ã‚’å�–ã‚‹ã€‚ã�“ã�“ã�§ã�¯æœªè¿”æ¸ˆè€…ï¼ˆ1ï¼‰ã�®DAYS_BIRTHã‚’å�–ã‚Šå‡ºã�—ã€�ã��ã‚Œã‚’365æ—¥ã�§å‰²ã‚‹ã�“ã�¨ã�§å¹´é½¢ã‚’ç®—å‡ºã�—ã�¦ã�„ã‚‹

plt.xlabel('Age(years)')
plt.ylabel('Density')
plt.title('Distribution of Ages')


#æ–°ã�—ã�„ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ age_dataã‚’ä½œã‚Šã€�ã‚‚ã�¨ã�®app_trainã�‹ã‚‰'TARGET'åˆ—ã�¨'DAYS_BIRTH'åˆ—ã� ã�‘ã‚’æŠœã��å‡ºã�—ã€�ä»£å…¥ã�—ã�¦ã�„ã‚‹

age_data = app_train[['TARGET','DAYS_BIRTH']]#app_train = {'TARGET': [0, 1, 0, 0, 1, ...], 'DAYS_BIRTH': [-10000, -12000, -8000, ...], 'INCOME': [150000, 200000, 180000, ...]}ã�¨ã�—ã�¦ã€�ä¸€ã�¤ã�®åˆ—ã�®ã�™ã�¹ã�¦ã�®å€¤ã‚’ãƒªã‚¹ãƒˆã�¨ã�—ã�¦ä¿�å­˜ã�—ã�Ÿã�„ã€‚ã�“ã�“ã�§[]ãƒªã‚¹ãƒˆã‚’ä½¿ã�„ã€�è¤‡æ•°ã�®ãƒªã‚¹ãƒˆã‚’ä½¿ã�ˆã‚‹ã‚ˆã�†ã�«[[]]ã�¨äºŒé‡�ã�«ãƒªã‚¹ãƒˆã‚’é‡�ã�­ã‚‹

age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH']/365 #age_dataã�®YEARS_BIRTHãƒªã‚¹ãƒˆã�«age_data['DAYS_BIRTH']/365ã‚’å…¥ã‚Œã‚‹


#pd.cut(ãƒ“ãƒ³åˆ†ã�‘ã�®å¯¾è±¡ã�¨ã�ªã‚‹åˆ—,bins = åˆ†å‰²æ•°)ã�¯pandasã�®é–¢æ•°ã�§ã€�é€£ç¶šå¤‰æ•°ã‚’åŒºé–“ã�”ã�¨ã�®ãƒ“ãƒ³ã�«åˆ†ã�‘ã‚‹
#numpyã�®linspace()ã�§20ã�‹ã‚‰70æ­³ã‚’11å€‹ã€�ã�¤ã�¾ã‚Š5æ­³ã�”ã�¨ã�®åŒºé–“ã�«åˆ†å‰²
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'],bins = np.linspace(20,70,num = 11))

age_data.head(10)


#groupbyã�§åŒºé–“ã�”ã�¨ã�«ã‚°ãƒ«ãƒ¼ãƒ—ã‚’ä½œã‚‹ã€‚ã�•ã‚‰ã�«.mean()ã�§å�„ã‚°ãƒ«ãƒ¼ãƒ—ã�®ä¸­ã�§ã€�æ•°å€¤ã�®å¹³å�‡ã‚’å�–ã‚‹
age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups


plt.figure(figsize = (8,8))
#astype(str)ã�§indexã�®å�‹ã‚’æ–‡å­—åˆ—ã�«å¤‰æ�›ã�™ã‚‹
plt.bar(age_groups.index.astype(str),100*age_groups['TARGET'])

plt .xticks(rotation = 75)

plt.xlabel('Age Group(years)')
plt.ylabel('Failure to Repay(%)')
plt.title('Failure to Repay by Age Group')



#æ–°ã�—ã�„ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ext_dataã�«app_train(è¨“ç·´ãƒ‡ãƒ¼ã‚¿)ã�‹ã‚‰TARGETå¤‰æ•°ã€�DAYS_BIRTHå¤‰æ•°ã€�EXT_SOURCE_3,EXT_SOURCE_2,EXT_SOURCE_1å¤‰æ•°ã‚’å�–ã‚Šå‡ºã�™ã€‚
ext_data = app_train[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]

#å¤‰æ•°ext_data_corrsã�«å�„åˆ—å�Œå£«ã�®ç›¸é–¢ä¿‚æ•°ã‚’è¨ˆç®—ã�—ã�Ÿã‚‚ã�®ã‚’è¿½åŠ 
#corr()ã�¯pandasã�®ãƒ¡ã‚½ãƒƒãƒ‰ã�§ã�‚ã‚Šã€�å�„åˆ—å�Œå£«ã�®ç›¸é–¢ä¿‚æ•°ã‚’è‡ªå‹•çš„ã�«è¨ˆç®—ã�™ã‚‹
ext_data_corrs = ext_data.corr()

ext_data_corrs


plt.figure(figsize = (8,6))

#ç›¸é–¢è¡Œåˆ—ã‚’è‰²ã�§è¡¨ç�¾ã�™ã‚‹seabornã�®heatmapé–¢æ•°ã‚’ä½¿ç”¨ã�™ã‚‹
#cmapã�§è‰²ã�®ç¨®é¡�ã‚’é�¸æŠ�ã�™ã‚‹
sns.heatmap(ext_data_corrs,cmap=plt.cm.RdYlBu_r,vmin = -0.25,vmax = 0.6,annot = True)#vminã�§è‰²ã�®æœ€å°�å€¤ã€�vmaxã�§è‰²ã�®æœ€å¤§å€¤ã‚’ã€�annotã�§å�„ã‚»ãƒ«ã�«æ•°å€¤ã‚’è¡¨ç¤º
plt.title('Correlation Heatmap')


plt.figure(figsize = (10,12))

# for i,source in enumerate([])ã�§ç•ªå�·ã�¨è¦�ç´ ã‚’å�Œæ™‚ã�«å�–ã‚Šå‡ºã�™é–¢æ•°ã€‚i=0ã�§EXT_SOURCE1ã�Œå�–ã‚Šå‡ºã�•ã‚Œã€�å¾Œã�®ã‚‚ã�®ã‚‚ã�“ã‚Œã�«å€£ã�†ã€‚ã�“ã‚Œã�§EXT_SOURCE1,2,3ã�®ã‚°ãƒ©ãƒ•ã‚’é †ã�«ç”Ÿæˆ�ã�—ã�¦ã�„ã��

for i,source in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):#sourceå¤‰æ•°ã�«EXT_SOURCEï¼Ÿã�Œå…¥ã‚‹

    plt.subplot(3,1,i+1)#ã‚°ãƒ©ãƒ•ã‚’ç¸¦ï¼“è¡Œã�§æ¨ªï¼‘åˆ—ã�«ã�—ã€�1â†’2â†’3ã�®ä½�ç½®ã�«æ��å†™ã�—ã�¦ã�„ã��ã€‚
    #ã‚«ãƒ¼ãƒ�ãƒ«å¯†åº¦æ�¨å®šã‚’æ›¸ã�„ã�¦ã�„ã��
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0,source],label = 'target == 0')#DataFrame.loc[è¡Œã�®æŒ‡å®š, åˆ—ã�®æŒ‡å®š]ã�¨ã�„ã�†æ§‹æ–‡ã‚’å�–ã‚‹ã€‚ã�“ã�“ã�§ã�¯è¿”æ¸ˆè€…ï¼ˆï¼�ï¼‰ã�®EXT_SOURCEã‚’å�–ã‚Šå‡ºã�™ã€‚
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1,source],label = 'target == 1')#DataFrame.loc[è¡Œã�®æŒ‡å®š, åˆ—ã�®æŒ‡å®š]ã�¨ã�„ã�†æ§‹æ–‡ã‚’å�–ã‚‹ã€‚ã�“ã�“ã�§ã�¯è¿”æ¸ˆè€…ï¼ˆ1ï¼‰ã�®EXT_SOURCEã‚’å�–ã‚Šå‡ºã�™ã€‚

    plt.xlabel('%s'% source)
    plt.ylabel('Density')




# ==========================================
# 1. ç’°å¢ƒã‚»ãƒƒãƒˆã‚¢ãƒƒãƒ—ã�¨ãƒ©ã‚¤ãƒ–ãƒ©ãƒª
# ==========================================
!pip install shap > /dev/null

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.special import expit # Sigmoid
import ipywidgets as widgets
from IPython.display import display, clear_output

# å†�ç�¾æ€§ç¢ºä¿�
np.random.seed(42)

print("âœ… ç’°å¢ƒæ§‹ç¯‰å®Œäº†")

# ==========================================
# 2. ãƒ‡ãƒ¼ã‚¿ç”Ÿæˆ� (å±�æ€§ãƒ‡ãƒ¼ã‚¿ã�®ã�¿èª­ã�¿è¾¼ã�¿)
# ==========================================
def load_financial_data(app_path='/kaggle/input/home-credit-default-risk/application_train.csv', n_samples=None):
    print("ãƒ‡ãƒ¼ã‚¿ã‚’èª­ã�¿è¾¼ã‚“ã�§ã�„ã�¾ã�™...")
    
    # ãƒ•ã‚¡ã‚¤ãƒ«ã�Œå­˜åœ¨ã�—ã�ªã�„å ´å�ˆã�®ãƒ€ãƒŸãƒ¼å‡¦ç�†ç”¨try-catchã�¯å®Ÿè¡Œæ™‚ã�«è¡Œã�„ã�¾ã�™
    if app_path is None: 
        raise FileNotFoundError
        
    df_app = pd.read_csv(app_path)
    
    # ãƒ‡ãƒ�ãƒƒã‚°ç”¨ã�«ã‚µãƒ³ãƒ—ãƒ«æ•°ã‚’çµ�ã‚‹å ´å�ˆ
    if n_samples:
        df_app = df_app.iloc[:n_samples]
        
    # --- é‡‘è��ç‰¹å¾´é‡�ã�®å‡¦ç�† (application_train) ---
    feature_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_EMPLOYED']
    target_col = 'TARGET'
    
    # å¿…è¦�ã�ªåˆ—ã‚’æŠ½å‡º
    X_df = df_app[feature_cols].copy()
    y = df_app[target_col].values

    # ã€�æ¬ æ��å€¤å‡¦ç�†ã€‘å¹³å�‡å€¤ã�§åŸ‹ã‚�ã‚‹
    for col in feature_cols:
        mean_val = X_df[col].mean()
        X_df[col] = X_df[col].fillna(mean_val)
        
    # numpyé…�åˆ—åŒ–
    X_financial = X_df.values

    return X_financial, y, feature_cols

# --- å®Ÿè¡Œ ---
try:
    # å®Ÿéš›ã�®ãƒ‡ãƒ¼ã‚¿ã�Œã�‚ã‚‹å ´å�ˆã�¯ãƒ‘ã‚¹ã‚’æŒ‡å®šã€�ã�ªã�„å ´å�ˆã�¯ãƒ€ãƒŸãƒ¼ãƒ‡ãƒ¼ã‚¿ç”Ÿæˆ�ã�ªã�©ã‚’æƒ³å®š
    # ã�“ã�“ã�§ã�¯Kaggleç’°å¢ƒç­‰ã�®ãƒ‘ã‚¹ã‚’æŒ‡å®šã�—ã�¦ã�„ã�¾ã�™ã�Œã€�ã‚¨ãƒ©ãƒ¼æ™‚ã�¯ãƒ€ãƒŸãƒ¼ãƒ¢ãƒ¼ãƒ‰ã�«ç§»è¡Œã�—ã�¾ã�™
    X_fin, y_true, feat_names = load_financial_data(
        app_path='/kaggle/input/home-credit-default-risk/application_train.csv', 
        n_samples=1000
    )
    print("âœ… å®Ÿãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿å®Œäº†")
    print(f"Financial Shape: {X_fin.shape}")

except FileNotFoundError:
    print("âš ï¸� CSVãƒ•ã‚¡ã‚¤ãƒ«ã�Œè¦‹ã�¤ã�‹ã‚Šã�¾ã�›ã‚“ã€‚ãƒ€ãƒŸãƒ¼ãƒ‡ãƒ¼ã‚¿ãƒ¢ãƒ¼ãƒ‰ã�§èµ·å‹•ã�—ã�¾ã�™ã€‚")
    # ãƒ€ãƒŸãƒ¼ãƒ‡ãƒ¼ã‚¿ã�®ç”Ÿæˆ� (5ç‰¹å¾´é‡� x 100ã‚µãƒ³ãƒ—ãƒ«)
    feat_names = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_EMPLOYED']
    X_fin = np.random.rand(100, 5)
    # å��å…¥ã‚„ã‚¯ãƒ¬ã‚¸ãƒƒãƒˆé¡�ã‚‰ã�—ã��ã‚¹ã‚±ãƒ¼ãƒªãƒ³ã‚°
    X_fin[:, 0] = X_fin[:, 0] * 200000 + 100000 # Income
    X_fin[:, 1] = X_fin[:, 1] * 500000 + 100000 # Credit
    X_fin[:, 4] = X_fin[:, 4] * -2000 # Days Employed (é€šå¸¸è² ã�®å€¤)
    y_true = np.random.randint(0, 2, 100)
    print("âœ… ãƒ€ãƒŸãƒ¼ãƒ‡ãƒ¼ã‚¿ç”Ÿæˆ�å®Œäº†")

# ==========================================
# 3. ãƒ¢ãƒ‡ãƒ«å®šç¾©: CSLBSystem (Pure Bandit)
# ==========================================
class CSLBSystem:
    def __init__(self, d, costs, lmbda=1.0):
        self.d = d
        self.costs = costs  # {'a': åˆ©æ�¯å��ç›Šä¿‚æ•°, 'c': ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆæ��å¤±ä¿‚æ•°}
        
        # ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿åˆ�æœŸåŒ– (Online Logistic Regressionç”¨)
        self.theta = np.zeros(d)      # å¹³å�‡ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿
        self.H = lmbda * np.eye(d)    # ãƒ˜ãƒƒã‚·ã‚¢ãƒ³ï¼ˆç²¾åº¦è¡Œåˆ—ï¼‰ã�®åˆ�æœŸåŒ–
        self.H_inv = np.linalg.inv(self.H) # è¨ˆç®—é«˜é€ŸåŒ–ã�®ã�Ÿã‚�é€†è¡Œåˆ—ã‚’æŒ�ã�¤
        
    def calculate_expected_profit(self, x, amount):
        """
        Thompson Samplingã‚’ç”¨ã�„ã�¦Î¸ã‚’ã‚µãƒ³ãƒ—ãƒªãƒ³ã‚°ã�—ã€�æœŸå¾…åˆ©ç›Šã‚’è¨ˆç®—
        """
        # 1. Thompson Sampling: äº‹å¾Œåˆ†å¸ƒ N(theta, H^-1) ã�‹ã‚‰ã‚µãƒ³ãƒ—ãƒªãƒ³ã‚°
        theta_sampled = np.random.multivariate_normal(self.theta, self.H_inv)
        
        # 2. ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆç¢ºç�‡ p = sigmoid(theta^T x)
        logit = np.dot(theta_sampled, x)
        p_default = expit(logit)
        
        # 3. å ±é…¬è¨­å®š (é‡‘é¡� w ã�«ä¾�å­˜)
        # æ­£å¸¸è¿”æ¸ˆæ™‚ã�®åˆ©ç›Š (interest)
        reward_tn = self.costs['a'] * amount 
        # ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆæ™‚ã�®æ��å¤± (loss)
        reward_fn = -self.costs['c'] * amount
        
        # 4. æœŸå¾…åˆ©ç›Š = (åˆ©ç›Š * æ­£å¸¸ç¢ºç�‡) + (æ��å¤± * ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆç¢ºç�‡)
        # æ­£å¸¸ç¢ºç�‡ = 1 - p_default
        expected_profit = (reward_tn * (1 - p_default)) + (reward_fn * p_default)
        
        return expected_profit, theta_sampled, p_default

    def update(self, x, y_observed):
        """
        ã‚ªãƒ³ãƒ©ã‚¤ãƒ³å­¦ç¿’: è¦³æ¸¬ã�•ã‚Œã�Ÿçµ�æ�œ(y)ã�«åŸºã�¥ã�„ã�¦Î¸ã�¨Hã‚’æ›´æ–°ã�™ã‚‹
        (ãƒ©ãƒ—ãƒ©ã‚¹è¿‘ä¼¼/ãƒ‹ãƒ¥ãƒ¼ãƒˆãƒ³æ³•ã�«åŸºã�¥ã��æ›´æ–°)
        y_observed: 1 (Default) or 0 (Paid)
        """
        # ç�¾åœ¨ã�®æ�¨å®šç¢ºç�‡
        p = expit(np.dot(self.theta, x))
        
        # ãƒ˜ãƒƒã‚·ã‚¢ãƒ³ã�®æ›´æ–°: H_new = H_old + p(1-p)xx^T
        weight = p * (1 - p)
        weight = max(weight, 1e-4) # æ•°å€¤å®‰å®šæ€§
        
        outer_prod = np.outer(x, x)
        self.H += weight * outer_prod
        
        # é€†è¡Œåˆ—ã�®æ›´æ–°
        self.H_inv = np.linalg.inv(self.H)
        
        # Thetaã�®æ›´æ–° (Newton Step)
        gradient = (p - y_observed) * x
        self.theta -= np.dot(self.H_inv, gradient)

    def explain_decision(self, x, theta_sampled, feature_names):
        """
        SHAPå€¤ã�®è¨ˆç®—ï¼ˆç·šå½¢ãƒ¢ãƒ‡ãƒ«ã�¨ã�—ã�¦è§£é‡ˆï¼‰
        """
        shap_values = x * theta_sampled
        
        # Explanationã‚ªãƒ–ã‚¸ã‚§ã‚¯ãƒˆã‚’æ§‹ç¯‰
        explainer = shap.Explanation(
            values=shap_values,
            base_values=0,
            data=x,
            feature_names=feature_names
        )
        return explainer

print("âœ… CSLBã‚·ã‚¹ãƒ†ãƒ å®šç¾©å®Œäº†")

# ==========================================
# 4. ã‚¤ãƒ³ã‚¿ãƒ©ã‚¯ãƒ†ã‚£ãƒ–ãƒ»ãƒ‡ãƒ¢ (å±�æ€§å¯©æŸ»ã�®ã�¿)
# ==========================================

# --- ã‚·ã‚¹ãƒ†ãƒ åˆ�æœŸåŒ– ---
# ç‰¹å¾´é‡�ã�¯5ã�¤ (Income, Credit, Ext1, Ext2, Employed)
# ã‚³ã‚¹ãƒˆè¨­å®š: é‡‘åˆ©10%, æ��å¤±ç�‡100%
cslb = CSLBSystem(d=5, costs={'a': 0.1, 'c': 1.0}) 

# --- UIéƒ¨å“�ã�®å®šç¾© ---
style = {'description_width': 'initial'}
layout_full = widgets.Layout(width='98%')
layout_half = widgets.Layout(width='48%')

# å…¥åŠ›ãƒ•ã‚©ãƒ¼ãƒ 
widgets.HTML("<b>ã€�é¡§å®¢å±�æ€§ãƒ‡ãƒ¼ã‚¿å…¥åŠ›ã€‘</b>")
input_income = widgets.FloatText(value=300000, description='å¹´é–“æ‰€å¾— (Income):', style=style, layout=layout_half)
input_credit = widgets.FloatText(value=500000, description='å¸Œæœ›è��è³‡é¡� (Credit):', style=style, layout=layout_half)
input_ext1 = widgets.FloatSlider(value=0.5, min=0, max=1, step=0.01, description='å¤–éƒ¨ã‚¹ã‚³ã‚¢1 (ä¿¡ç”¨åº¦):', style=style, layout=layout_full)
input_ext2 = widgets.FloatSlider(value=0.5, min=0, max=1, step=0.01, description='å¤–éƒ¨ã‚¹ã‚³ã‚¢2 (ä¿¡ç”¨åº¦):', style=style, layout=layout_full)
input_employed = widgets.FloatText(value=-1000, description='å‹¤ç¶šæ—¥æ•° (è² ã�®å€¤):', style=style, layout=layout_full)

# ã‚¢ã‚¯ã‚·ãƒ§ãƒ³ãƒœã‚¿ãƒ³
btn_run = widgets.Button(description="å¯©æŸ»å®Ÿè¡Œ (Run Analysis)", button_style='primary', icon='calculator', layout=layout_full)

# ãƒ•ã‚£ãƒ¼ãƒ‰ãƒ�ãƒƒã‚¯ç”¨ãƒœã‚¿ãƒ³
lbl_feedback = widgets.Label("ã€�çµ�æ�œãƒ•ã‚£ãƒ¼ãƒ‰ãƒ�ãƒƒã‚¯ã€‘ å®Ÿéš›ã�®è¿”æ¸ˆçµ�æ�œã‚’æ•™ã�ˆã�¦ã��ã� ã�•ã�„ï¼ˆãƒ¢ãƒ‡ãƒ«ã‚’æ›´æ–°ã�—ã�¾ã�™ï¼‰:")
btn_paid = widgets.Button(description="å®Œæ¸ˆã�—ã�Ÿ (Observed: 0)", button_style='success', icon='check')
btn_default = widgets.Button(description="ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆã�—ã�Ÿ (Observed: 1)", button_style='danger', icon='times')
feedback_box = widgets.VBox([lbl_feedback, widgets.HBox([btn_paid, btn_default])])
feedback_box.layout.display = 'none'

# å‡ºåŠ›ã‚¨ãƒªã‚¢
output_area = widgets.Output()

# --- ã‚¤ãƒ™ãƒ³ãƒˆãƒ�ãƒ³ãƒ‰ãƒ© ---
global last_x_features
last_x_features = None

def on_click_run(b):
    global last_x_features
    feedback_box.layout.display = 'none'
    output_area.clear_output()
    
    with output_area:
        # 1. å±�æ€§ãƒ‡ãƒ¼ã‚¿ã�®å�–å¾—ã�¨æ­£è¦�åŒ–ï¼ˆç°¡æ˜“çš„ã�ªæ­£è¦�åŒ–ï¼‰
        # ç‰¹å¾´é‡�ã�®é †åº�: ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_EMPLOYED']
        fin_features = np.array([
            input_income.value, 
            input_credit.value, 
            input_ext1.value, 
            input_ext2.value, 
            input_employed.value
        ])
        
        # æ­£è¦�åŒ–ä¿‚æ•° (å®Ÿãƒ‡ãƒ¼ã‚¿ã�®ã‚¹ã‚±ãƒ¼ãƒ«ã�«å�ˆã‚�ã�›ã‚‹ã�Ÿã‚�ã�®ç°¡æ˜“ã‚¹ã‚±ãƒ¼ãƒ©ãƒ¼)
        scale_factors = np.array([1000000, 1000000, 1, 1, 3000])
        x_norm = fin_features / scale_factors
        
        last_x_features = x_norm
        
        # 2. CSLBã�«ã‚ˆã‚‹æ„�æ€�æ±ºå®š
        amount = input_credit.value
        exp_profit, theta_sampled, p_default = cslb.calculate_expected_profit(x_norm, amount)
        
        decision = "æ‰¿èª� (APPROVED)" if exp_profit > 0 else "æ‹’å�¦ (REJECTED)"
        
        # çµ�æ�œè¡¨ç¤º
        print(f"--- Decision Engine Result ---")
        print(f"åˆ¤å®š: \033[1;3{1 if exp_profit <=0 else 2}m{decision}\033[0m")
        print(f"ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆç¢ºç�‡ (æ�¨å®š): {p_default:.1%}")
        print(f"æœŸå¾…å��ç›Š: {exp_profit:,.0f} JPY")
        
        # SHAPè¡¨ç¤º
        print("\nã€�åˆ¤æ–­æ ¹æ‹  (SHAP Values)ã€‘")
        shap_expl = cslb.explain_decision(x_norm, theta_sampled, feat_names)
        plt.figure(figsize=(8, 3))
        shap.plots.waterfall(shap_expl, show=False)
        plt.show()

        # æ‰¿èª�ã�•ã‚Œã�Ÿå ´å�ˆã�®ã�¿ãƒ•ã‚£ãƒ¼ãƒ‰ãƒ�ãƒƒã‚¯ã‚’å�—ã�‘ä»˜ã�‘ã‚‹
        if exp_profit > 0:
            print("æ‰¿èª�ã�•ã‚Œã�¾ã�—ã�Ÿã€‚çµ�æ�œãƒ•ã‚£ãƒ¼ãƒ‰ãƒ�ãƒƒã‚¯ã�«ã‚ˆã‚Šãƒ¢ãƒ‡ãƒ«ã‚’å­¦ç¿’å�¯èƒ½ã�§ã�™ã€‚")
            feedback_box.layout.display = 'block'
        else:
            print("æ‹’å�¦ã�•ã‚Œã�Ÿã�Ÿã‚�ã€�è��è³‡ã�¯å®Ÿè¡Œã�•ã‚Œã�¾ã�›ã‚“ï¼ˆå­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�ªã�—ï¼‰ã€‚")

def on_click_feedback(observed_outcome):
    global last_x_features
    with output_area:
        print("\n--- Feedback Loop ---")
        # ãƒ¢ãƒ‡ãƒ«ã�®æ›´æ–°
        cslb.update(last_x_features, observed_outcome)
        
        res_text = "å®Œæ¸ˆ" if observed_outcome == 0 else "ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆ"
        print(f"ã€Œ{res_text}ã€�ã�®çµ�æ�œã‚’å­¦ç¿’ã�—ã�¾ã�—ã�Ÿã€‚ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿Î¸ã�Œæ›´æ–°ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")
        feedback_box.layout.display = 'none'

# ã‚¤ãƒ™ãƒ³ãƒˆç´�ä»˜ã�‘
btn_run.on_click(on_click_run)
btn_paid.on_click(lambda b: on_click_feedback(0))
btn_default.on_click(lambda b: on_click_feedback(1))

# ç”»é�¢æ��ç”»
display(widgets.VBox([
    widgets.HTML("<h3>ğŸš€ Contextual Bandit Credit Model (No CNN)</h3>"),
    widgets.Label("é¡§å®¢å±�æ€§ã‚’å…¥åŠ›ã�—ã�¦ã��ã� ã�•ã�„:"),
    widgets.HBox([input_income, input_credit]),
    input_ext1,
    input_ext2,
    input_employed,
    widgets.HTML("<hr>"),
    btn_run,
    output_area,
    feedback_box
]))

