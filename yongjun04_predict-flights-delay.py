!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf


import matplotlib.font_manager as fm
', '.join(sorted([font.name for font in fm.fontManager.ttflist]))


fontpaths = "/kaggle/input/nanumfontsetup/NanumFontSetup_TTF_SQUARE"
font_list = fm.findSystemFonts(fontpaths = fontpaths, fontext='ttf')
for font_file in font_list:
    fm.fontManager.addfont(font_file)
fm._load_fontmanager(try_read_cache=False)


import matplotlib.pyplot as plt

plt.rc('font', family='NanumBarunGothic')


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='NanumSquare_ac')
import seaborn as sns

# ëª¨ë“  ì—´ì�„ ëª¨ë‘� displayí•˜ê¸° ìœ„í•¨.
pd.set_option('display.max_columns', 100)


train = pd.read_csv('/kaggle/input/flight-delay-data/flight_delays_train.csv')
train.info()
train.head()


col_name = ['Month','DayofMonth','DayOfWeek']

for col in col_name:
  lis = []
  for a in train[col]:
    if 'c' in a :
      lis .append(a.split('-')[1])
  train[col] = lis


col_name = {'Month':'ì›”','DayofMonth':'ì�¼','DayOfWeek':'ì£¼','DepTime':'ì¶œë°œì‹œê°„','UniqueCarrier':'í•­ê³µì‚¬ë²ˆí˜¸','Origin':'ì¶œë°œì§€','Dest':'ë�„ì°©ì§€','Distance':'ê±°ë¦¬','dep_delayed_15min':'ì§€ì—°ìœ ë¬´'}

train=train.rename(columns   = col_name)


train['ê±°ë¦¬'].duplicated()


df = train.pivot_table(index = 'ì¶œë°œì§€',columns='ì§€ì—°ìœ ë¬´',aggfunc = 'size').reset_index()
df['ì§€ì—°ìœ¨'] = df['Y']/(df['N']+df['Y'])
df = df.nlargest(10,'ì§€ì—°ìœ¨')



plt.figure(figsize=(12, 6))


colors = sns.color_palette("Blues_r", len(df))


sns.barplot(x='ì¶œë°œì§€', y='ì§€ì—°ìœ¨', data=df, palette=colors)


plt.title("ì¶œë°œì§€ë³„ í•­ê³µ ì§€ì—°ìœ¨ TOP 10", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("ì¶œë°œ ê³µí•­", fontsize=14, labelpad=10)
plt.ylabel("ì§€ì—°ìœ¨", fontsize=14, labelpad=10)


plt.xticks(rotation=45, ha='right', fontsize=12)


plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))


for index, value in enumerate(df["ì§€ì—°ìœ¨"]):
    plt.text(index, value + 0.005, f'{value*100:.1f}%', ha='center', fontsize=12, fontweight='bold')

sns.despine()


plt.show()




df = train.pivot_table(index = 'ì§€ì—°ìœ ë¬´',values = 'ê±°ë¦¬',aggfunc = 'mean').reset_index()
sns.barplot(x='ì§€ì—°ìœ ë¬´',y='ê±°ë¦¬',data = df)


#ê±°ë¦¬ë¶„í�¬ë¥¼ ë³¸ê²°ê³¼ ê±°ë¦¬ëŠ” í�¬ê²Œ ì§€ì—°ìœ ë¬´ì—� ì˜�í–¥ì�´ ì—†ì�Œ
plt.figure(figsize=(8, 6))
sns.boxplot(x='ì§€ì—°ìœ ë¬´', y='ê±°ë¦¬', data=train, palette="Blues")
plt.title('ì§€ì—°ìœ ë¬´ë³„ ê±°ë¦¬ë¶„í�¬')
plt.xlabel('Outcome')
plt.ylabel('ê±°ë¦¬')
plt.show()


from sklearn.preprocessing import MinMaxScaler
train_pivot = train.pivot_table(index = 'ì›”',columns = 'ì§€ì—°ìœ ë¬´',values = 'ê±°ë¦¬', aggfunc = 'count')

train_pivot = train_pivot.reset_index()
train_pivot.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in train_pivot.columns]
train_pivot = train_pivot.sort_values(by='ì›”')


train_pivot['ì—°ì°©ë¹„ìœ¨'] = train_pivot['Y']/(train_pivot['N']+train_pivot['Y'])
scaler = MinMaxScaler()


train_pivot[['N', 'Y']] = scaler.fit_transform(train_pivot[['N', 'Y']])
train_pivot['ì›”'] = train_pivot['ì›”'].astype(int)
train_pivot = train_pivot.sort_values(by='ì›”',ascending = False)


train_pivot



train_pivot.set_index('ì›”', inplace=True)

# í�ˆíŠ¸ë§µ ê·¸ë¦¬ê¸°
plt.figure(figsize=(8, 6))
sns.heatmap(train_pivot, cmap="Blues", annot=True)

# ì œëª© ë°� ë�¼ë²¨ ì„¤ì •
plt.title("ì›”ë³„ ì§€ì—° í˜„í™© í�ˆíŠ¸ë§µ")
plt.xlabel("ì§€ì—° ì—¬ë¶€")
plt.ylabel("ì›”")

# ê·¸ë�˜í”„ í‘œì‹œ
plt.show()


hour = []
minute = []

for x in train['ì¶œë°œì‹œê°„']:
    if len(str(x))>3:
        hour.append(str(x)[:2])
        minute.append(str(x)[2:])
    else:
        hour.append(str(x)[:1])
        minute.append(str(x)[1:])


train['ì¶œë°œì‹œê°�'] = hour
train['ì¶œë°œë¶„'] = minute

train = train[train['ì¶œë°œì‹œê°�'].astype(int)<=24]


from matplotlib.ticker import MaxNLocator


train_pivot = train.pivot_table(index = 'ì¶œë°œì‹œê°�',columns = 'ì§€ì—°ìœ ë¬´', aggfunc = 'size')
train_pivot['ì§€ì—°ìœ¨'] = (train_pivot['Y']/(train_pivot['N']+train_pivot['Y']))*100


train_pivot_reset = train_pivot.reset_index()
train_pivot_reset['ì¶œë°œì‹œê°�'] = train_pivot_reset['ì¶œë°œì‹œê°�'].astype(int)
train_pivot = train_pivot_reset.sort_values('ì¶œë°œì‹œê°�')


plt.figure(figsize=(10, 6))  
sns.lineplot(x=train_pivot['ì¶œë°œì‹œê°�'], y=train_pivot['ì§€ì—°ìœ¨'], color='red', marker='o', linestyle='-')

plt.title("ì¶œë°œ ì‹œê°�ì—� ë”°ë¥¸ ì§€ì—°ìœ¨ ë³€í™”", fontsize=15, fontweight='bold')
plt.xlabel("ì¶œë°œ ì‹œê°�", fontsize=12)
plt.ylabel("ì§€ì—°ìœ¨ (%)", fontsize=12)
plt.xticks(rotation=45) 


plt.axvspan(6, 12, color='#FFEB3B', alpha=0.3, label='ì˜¤ì „')  
plt.axvspan(12, 18, color='#00BCD4', alpha=0.3, label='ì˜¤í›„')
plt.axvspan(18, 24, color='#FF7043', alpha=0.3, label='ì €ë…�')
plt.axvspan(1, 6, color='#1A237E', alpha=0.3, label='ìƒˆë²½')

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=26))


plt.legend()


plt.show()


train_pivot


train['ê±°ë¦¬'].value_counts()








features = ['ì›”','ì�¼','ì£¼','í•­ê³µì‚¬ë²ˆí˜¸','ì¶œë°œì§€','ë�„ì°©ì§€','ê±°ë¦¬','ì¶œë°œì‹œê°�']
target = ['ì§€ì—°ìœ ë¬´']

x = train[features]
y = train[target]


x['í•­ê³µì‚¬ë²ˆí˜¸'] = x['í•­ê³µì‚¬ë²ˆí˜¸'].astype('category').cat.codes
x['ì¶œë°œì§€'] = x['ì¶œë°œì§€'].astype('category').cat.codes
x['ë�„ì°©ì§€'] = x['ê±°ë¦¬'].astype('category').cat.codes



x = x.astype(int)


from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                      test_size = 0.3,
                                                      stratify = y,
                                                      random_state = 51)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train, y_train)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.metrics import confusion_matrix

#ëª¨ë�¸ì„ ì •ê³¼ì •

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
     'KNN'  : KNeighborsClassifier()

}


for model_name, model in models.items():
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    pred = model.predict(x_test)


    print(f"model : {model_name}, Score : {score}")


    cm = confusion_matrix(y_test,pred)
    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.ylabel('True')
    plt.show()


model =  RandomForestClassifier(
    n_estimators=300,      
    min_samples_split=5,   
    min_samples_leaf=2,    
    max_features='sqrt',   
    bootstrap=True,        
    random_state=42,       
    n_jobs=-1              
)



model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print(score)


#!ì‹¤í–‰ì‹œê°„ ì˜¤ë�˜ê±¸ë¦¼!

from sklearn.model_selection import RandomizedSearchCV


param_dist = {
    'n_estimators': np.arange(50, 500, 50),  
    'max_depth': np.arange(5, 50, 5),  
    'min_samples_split': [2, 5, 10, 20],  
    'min_samples_leaf': [1, 2, 4, 8],  
}


rf = RandomForestClassifier(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(rf, param_dist, n_iter=30, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
random_search.fit(x_train, y_train)

# ìµœì �ì�˜ í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° ì¶œë ¥
print("ìµœì �ì�˜ íŒŒë�¼ë¯¸í„°:", random_search.best_params_) 
print("ìµœì � ëª¨ë�¸ ì •í™•ë�„:", random_search.best_score_)




model =  RandomForestClassifier(
    n_estimators=350,      
    min_samples_split=2,   
    min_samples_leaf=1,    
    
    random_state=42,       
    n_jobs=-1              
)



model.fit(x_train, y_train)
score = model.predict()

print(score)


test = pd.read_csv('/kaggle/input/flight-delay-data/flight_delays_test.csv')
test.info()
test.head()


col_name = ['Month','DayofMonth','DayOfWeek']

for col in col_name:
  lis = []
  for a in test[col]:
    if 'c' in a :
      lis .append(a.split('-')[1])
  test[col] = lis


features = ['Mo']


test


col_name


test = test.rename(columns = col_name)


test


test['í•­ê³µì‚¬ë²ˆí˜¸'] = test['í•­ê³µì‚¬ë²ˆí˜¸'].astype('category').cat.codes
test['ì¶œë°œì§€'] = test['ì¶œë°œì§€'].astype('category').cat.codes
test['ë�„ì°©ì§€'] = test['ê±°ë¦¬'].astype('category').cat.codes



test = test.astype(int)


test


hour = []
minute = []

for x in test['ì¶œë°œì‹œê°„']:
    if len(str(x))>3:
        hour.append(str(x)[:2])
        minute.append(str(x)[2:])
    else:
        hour.append(str(x)[:1])
        minute.append(str(x)[1:])


test['ì¶œë°œì‹œê°�'] = hour
test['ì¶œë°œë¶„'] = minute

test = test[test['ì¶œë°œì‹œê°�'].astype(int)<=24]


test = test[['ì›”','ì�¼','ì£¼','í•­ê³µì‚¬ë²ˆí˜¸','ì¶œë°œì§€','ë�„ì°©ì§€','ê±°ë¦¬','ì¶œë°œì‹œê°�']]


model =  RandomForestClassifier(
    n_estimators=350,      
    min_samples_split=2,   
    min_samples_leaf=1,    
    
    random_state=42,       
    n_jobs=-1              
)



model.fit(x_train, y_train)
test_pred = model.predict(test)

print(test_pred)


test['ì¶œë°œì§€ì—°ì˜ˆì¸¡'] = test_pred


test['ì¶œë°œì§€ì—°ì˜ˆì¸¡'].value_counts()

