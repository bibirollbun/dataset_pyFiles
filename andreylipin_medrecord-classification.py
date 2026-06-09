from collections import Counter
from IPython import display as D
from itertools import product
from matplotlib import pyplot as plt
from scipy import stats as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import mnlogit
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import gc
import joblib
import numpy as np
import pandas as pd
import re
import seaborn as sns
import time
import torch
import warnings
warnings.filterwarnings(action='ignore')



# Reading
sample_df = pd.read_csv('data/sampleResults.csv')
train_labels_df = pd.read_csv('data/trainLabels.csv',names=['Filename','Label'])

med_records_df = pd.DataFrame(columns = ['Filename','Record'])
path = 'data/Records/'
files = !ls $path
files = [path+i for i in list(files)]
for file in files:
    with open(file,'r',encoding='latin-1') as f:
        record = ''.join([i for i in f.readlines()])
    med_records_df.loc[-1] = [file.split('/')[-1],record]
    med_records_df.index = med_records_df.index + 1
med_records_df = med_records_df.reset_index(drop=True)
med_records_df.head()


first_sample = med_records_df.iloc[0,1]
D.display_markdown(first_sample,raw=True)


med_records_df.info()


X = pd.merge(med_records_df,train_labels_df,on='Filename')
X['Record'] = X['Record'].apply(lambda x: re.sub("CC:|<B>|</B>|\\n","",x))
med_records_df_dropped = X.loc[(X['Record'].str.len()==0),:].copy()
med_records_df_dropped


med_records_df['Record'] = med_records_df['Record'].apply(lambda x: re.sub("CC:|<B>|</B>|\\n","",x))
med_records_df_dropped = med_records_df.loc[(med_records_df['Record'].str.len()==0),:].copy()
med_records_df_dropped


med_records_df = med_records_df.reset_index(drop=True)
med_records_df['Record'] = med_records_df['Record'].apply(lambda x: re.sub("CC:|<B>|</B>|\\n","",x))

med_records_df.loc[med_records_df['Record'].str.len()==0,'Record'] = ' '


warnings.filterwarnings(action='ignore')
# TRAIN TEST SPLIT
X = pd.merge(med_records_df,train_labels_df,on='Filename').reset_index(drop=1)
random_state_split = 1
X_train, X_test, y_train, y_test = train_test_split(X,X['Label'],train_size=0.7,stratify=X['Label'],random_state=random_state_split) # 0,10


# TFIDF
## Here threshold is used to select only those words that do not appear in the records too frequently.
## Additionally, each threshold will by checked by f1 and accuracy scores.
results = []

for threshold in np.linspace(0,0.1,11):
    print(f'{threshold = }')
    X_train, X_test, y_train, y_test = train_test_split(X,X['Label'],train_size=0.7,stratify=X['Label'],random_state=10)
    vectorizer = TfidfVectorizer(analyzer= 'word', stop_words= 'english',lowercase=True, ngram_range = (1,2), norm=None)
    corpus = X_train['Record']
    tfidf_matrix = vectorizer.fit_transform(corpus)
    corpus_test = X_test['Record']
    tfidf_matrix_test = vectorizer.transform(corpus_test)
    vec_cols = list(np.hstack(vectorizer.get_feature_names_out()[np.argwhere(vectorizer.idf_<=np.quantile(vectorizer.idf_,threshold))]))
    vec_df = pd.DataFrame(tfidf_matrix.toarray(),columns=list(vectorizer.get_feature_names_out()))[vec_cols].fillna(0)
    X_train[vec_cols] = vec_df.values
    vec_df = pd.DataFrame(tfidf_matrix_test.toarray(),columns=list(vectorizer.get_feature_names_out()))[vec_cols].fillna(0)
    X_test[vec_cols] = vec_df.values
    lrc = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X_train.iloc[:,3:].fillna(0))
    lrc.fit(X_train_,y_train)
    X_test_ = scaler.transform(X_test.iloc[:,3:].fillna(0))
    y_hat = lrc.predict(X_test_)
    print(f'column counts: {len(vec_cols) = }')
    print('Test base model')
    f1 = f1_score(y_hat, y_test, average = "macro")
    print("Macro-Average F1: ", f1)
    accuracy = accuracy_score(y_hat, y_test)
    print("Accuracy: ", accuracy)
    results.append([threshold,f1,accuracy])
    X_train = X_train.drop(vec_cols,axis=1)
    X_test = X_test.drop(vec_cols,axis=1)
    ConfusionMatrixDisplay.from_predictions(y_hat, y_test)
    plt.show()


results_tfidf = pd.DataFrame(results)
results_tfidf = results_tfidf.set_index(results_tfidf[0].round(2))
results_tfidf.columns = ['threshold','f1','accuracy']
sns.lineplot(data = results_tfidf,x = 'threshold',y='f1',errorbar = None)
sns.lineplot(data = results_tfidf,x = 'threshold',y='accuracy',errorbar = None)
plt.ylabel('f1 / accuracy')
plt.legend(['F1','Accuracy'])
plt.ylim(0,1)
plt.show()


results_tfidf


st.t.interval(0.95,1,results_tfidf.loc[0.01:,'f1'].mean(),results_tfidf.loc[0.01:,'f1'].std())


X_train, X_test, y_train, y_test = train_test_split(X,X['Label'],train_size=0.7,stratify=X['Label'],random_state=random_state_split)
vectorizer = TfidfVectorizer(analyzer= 'word', stop_words= 'english',lowercase=True, ngram_range = (1,2), norm=None)

# TRAIN
corpus = X_train['Record']
tfidf_matrix = vectorizer.fit_transform(corpus)
threshold = 0.1
vec_cols = list(np.hstack(vectorizer.get_feature_names_out()[np.argwhere(vectorizer.idf_<=np.quantile(vectorizer.idf_,threshold))]))
vec_df = pd.DataFrame(tfidf_matrix.toarray(),columns=list(vectorizer.get_feature_names_out()))[vec_cols].fillna(0)
X_train[vec_cols] = vec_df.values

# TEST
corpus_test = X_test['Record']
tfidf_matrix_test = vectorizer.transform(corpus_test)
vec_df = pd.DataFrame(tfidf_matrix_test.toarray(),columns=list(vectorizer.get_feature_names_out()))[vec_cols].fillna(0)
X_test[vec_cols] = vec_df.values

# LR test
lrc = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
scaler = StandardScaler()
X_train_ = scaler.fit_transform(X_train.iloc[:,3:].fillna(0))
lrc.fit(X_train_,y_train)
X_test_ = scaler.transform(X_test.iloc[:,3:].fillna(0))
y_hat = lrc.predict(X_test_)
print('Test base model')
f1 = f1_score(y_hat, y_test, average = "macro")
print("Macro-Average F1: ", f1)
accuracy = accuracy_score(y_hat, y_test)
print("Accuracy: ", accuracy)




sns.countplot(X['Label'])


joblib.dump(vectorizer,'vectorizer.joblib')
vec_cols[500:3000]


thresh = 500

print(*vec_cols[thresh:thresh+100],sep=', ')


X_test = X_test.drop(vec_cols[:thresh],axis=1)
X_train = X_train.drop(vec_cols[:thresh],axis=1)
vec_cols = vec_cols[thresh:]


sns.boxplot(y=X_train[vec_cols].mean(axis=1),x=X_train['Label']);


words_for_pat = X_train[['Label']+vec_cols].groupby('Label').sum().T
words_for_pat.describe()


words_for_pat


g_words_for_pat = words_for_pat.loc[(words_for_pat['Gastroenterology']>words_for_pat['Gastroenterology'].quantile(0.99))].index
n_words_for_pat = words_for_pat.loc[(words_for_pat['Neurology']>words_for_pat['Neurology'].quantile(0.99))].index
o_words_for_pat = words_for_pat.loc[(words_for_pat['Orthopedic']>words_for_pat['Orthopedic'].quantile(0.99))].index
r_words_for_pat = words_for_pat.loc[(words_for_pat['Radiology']>words_for_pat['Radiology'].quantile(0.99))].index
u_words_for_pat = words_for_pat.loc[(words_for_pat['Urology']>words_for_pat['Urology'].quantile(0.99))].index


len(r_words_for_pat)


counter = 0
checks = []
check_g = set(g_words_for_pat)-set(n_words_for_pat)-set(o_words_for_pat)-set(r_words_for_pat)-set(u_words_for_pat)
checks.append(check_g)
print(f'Check lenght: {len(check_g)}')
for c in ['Gastroenterology','Neurology','Orthopedic','Radiology','Urology']:
    counter = X_train[['Record','Label']].apply(lambda x: sum([1 if w in x['Record'].lower() and x['Label']==c else 0 \
                                                               for w in check_g]),axis=1).sum()
    print(c,counter)



counter = 0
check_n = set(n_words_for_pat)-set(g_words_for_pat)-set(o_words_for_pat)-set(r_words_for_pat)-set(u_words_for_pat)
checks.append(check_n)
print(f'Check length: {len(check_n)}')
for c in ['Gastroenterology','Neurology','Orthopedic','Radiology','Urology']:
    counter = X_train[['Record','Label']].apply(lambda x: sum([1 if w in x['Record'].lower() and x['Label']==c else 0 \
                                                               for w in check_n]),axis=1).sum()
    print(c,counter)



counter = 0
check_o = set(o_words_for_pat)-set(g_words_for_pat)-set(n_words_for_pat)-set(r_words_for_pat)-set(u_words_for_pat)
checks.append(check_o)
print(f'Check length: {len(check_o)}')
for c in ['Gastroenterology','Neurology','Orthopedic','Radiology','Urology']:
    counter = X_train[['Record','Label']].apply(lambda x: sum([1 if w in x['Record'].lower() and x['Label']==c else 0 \
                                                               for w in check_o]),axis=1).sum()
    print(c,counter)



counter = 0
check_r = set(r_words_for_pat)-set(g_words_for_pat)-set(n_words_for_pat)-set(o_words_for_pat)-set(u_words_for_pat)
checks.append(check_r)
print(f'Check length: {len(check_r)}')
for c in ['Gastroenterology','Neurology','Orthopedic','Radiology','Urology']:
    counter = X_train[['Record','Label']].apply(lambda x: sum([1 if w in x['Record'].lower() and x['Label']==c else 0 \
                                                               for w in check_r]),axis=1).sum()
    print(c,counter)



words_for_pat


from wordcloud import WordCloud
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,5,figsize=(20,10))
count = 0
for class_name in ['Gastroenterology','Orthopedic','Urology','Radiology','Neurology']:
    wcloud_df = words_for_pat[[class_name]].reset_index().sort_values(class_name)
    wcloud_df.columns = ['Feature','Importance']
    wcloud = WordCloud(width=1000, height=800, background_color='white',colormap='coolwarm',max_words=50,random_state=42)
    wcloud = wcloud.generate_from_frequencies(dict(wcloud_df.values),max_font_size=200)
    ax[count].set_xticks([])
    ax[count].set_yticks([])
    ax[count].set_title(class_name)
    ax[count].imshow(wcloud)
    count+=1
plt.tight_layout()
plt.show()



counter = 0
check_u = set(u_words_for_pat)-set(g_words_for_pat)-set(n_words_for_pat)-set(o_words_for_pat)-set(r_words_for_pat)
checks.append(check_u)
print(f'Check length: {len(check_u)}')
for c in ['Gastroenterology','Neurology','Orthopedic','Radiology','Urology']:
    counter = X_train[['Record','Label']].apply(lambda x: sum([1 if w in x['Record'].lower() and x['Label']==c else 0 \
                                                               for w in check_u]),axis=1).sum()
    print(c,counter)



X_train.head()


joblib.dump(checks,'checks.joblib')


X_train = X_train[['Filename', 'Record', 'Label']+list(np.hstack([list(i) for i in checks]))]
X_test = X_test[['Filename', 'Record', 'Label']+list(np.hstack([list(i) for i in checks]))]



vec_cols = list(np.hstack([list(i) for i in checks]))
vec_cols = ['vec_'+i for i in vec_cols]
X_train = X_train.rename(columns=dict(zip(X_train.columns[3:],vec_cols)))
X_test = X_test.rename(columns=dict(zip(X_test.columns[3:],vec_cols)))



X_train.head()


X_test.head()


lrc = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
scaler = StandardScaler()
X_train_ = scaler.fit_transform(X_train[vec_cols])
lrc.fit(X_train_,y_train)
X_test_ = scaler.transform(X_test[vec_cols])
y_hat = lrc.predict(X_test_)
print('Test base model')
f1 = f1_score(y_hat, y_test, average = "macro")
print("Macro-Average F1: ", f1)
accuracy = accuracy_score(y_hat, y_test)
print("Accuracy: ", accuracy)



ConfusionMatrixDisplay.from_predictions(y_hat, y_test)
plt.show()


# RAW-TAG dictionary
patterns = dict()
patterns['his_pat'] = 'HX:|HISTORY:|PMH:|FHX:|SHX:|PRESENT ILLNESS:|BACKGROUNDPRIOR SURGERIES:|HPI:|SYMPTOMS:|CLINICAL INFORMATION:|OUTSIDE RECORDS:|ALLERGIES:|HABITS:|RECREATIONAL PURSUITS:|SOCIAL HABITS:|IMMUNIZATIONS:'
patterns['exam_pat'] = 'EXAM:|PHYSICAL EXAMINATION:|^MS$:|^CN$:|STATION:|INSPECTION:|PALPATION:|JOINT PLAY:'
patterns['course_pat'] = 'DIAGNOS:|COURSE:|HOSPITAL COURSE:|CLINICAL NOTE:|DISCHARGE NOTE:|FOLLOWUP:|FOLLOW UP:|ASSESSMENT:|PLAN:|TREATMENT:|PROGNOSIS:|INITIAL STUDIES:|WORKUP:|SUMMARY:|IMPRESSION:|CONCLUSION:|REPORT:|DESCRIPTION:'
patterns['rec_pat'] = 'RECOMMENDATION:|INSTRUCTION:|ADVICE:|COUNSELING:|PRESCRIPTIONS:|RICE:|REST:|ICE:|COMPRESSION:|ELEVATION:'
patterns['lab_pat'] = 'LAB:|LABORATORY:|BLOOD:|URINE:|TESTS:|ANALYSIS:|CBC:|CHEM:|COAGULATION:|HCT:|CRP:|^ESR$:| ESR :|^RF$:|^ANA$:| ANA :|ANCA:|TSH:|FT4:|PSA:|GLUCOSE:|ELECTROLYTES:|CULTURES:|SPECIMENS:'
patterns['img_pat'] = '^CT$:|^US$:|MRI:|X-RAY:|ULTRASOUND:|DOPPLER:|MAMMOGRAPHY:|NUCLEAR:|CARDIOLITE:|MYOVIEW:|ECHOCARDIOGRAM:|FLUORO:|RADIOLOGIC DATA:|IMAGING:|DIMENSION:|CXR:|STUDIES:|PULMONARY:|CARDIAC:'
patterns['med_pat'] = 'MEDS:|MEDICATION:|MEDICATIONS:|DRUGS:|PRESCRIPTIONS:|ANTIBIOTIC:|INJECTABLES:|ANESTHESIA:|ANESTHETIC:'
patterns['inf_pat'] = 'FLUIDS:|CRYSTALLOIDS:|INTRAVENOUS FLUID:'
patterns['sur_pat'] = 'DRAINS:|TUBES:|OPERATION:|OPERATIV:|PROCEDURE:|DETAILS OF THE OR:|SURGERY IN DETAIL:|ENDOSCOPIC:|DE QUERVAIN:|BRACHYTHERAPY:|HYPERFRACTIONATION:|BIOPSIES:|RESECTION TIME:|TOURNIQUET:|CLOSURE:|EBL:|TIME OUT:|SPONGE:|NEEDLE COUNT:|HEMOSTASIS:|PREP:|INSTRUMENT:'
patterns['neuro_pat'] = 'NEUROLOGICAL:|COGNITIVE:|INTELLECTUAL:|VISUOSPATIAL:|VISUAL LEARNING:|MOTOR:|EMOTIONAL:|BEHAVIORAL:|ATTENTION:|EXECUTIVE FUNCTIONING:|SLEEP:|EEG:|NEUROPSYCHOLOGICAL:|MOTOR:|SENSORY:|GAIT:|COORD:|REFLEXES:|MOTION:'
patterns['uro_pat'] = 'CIRCUMCIS:|ORCHECTOM:|EPIDIDYMECTOM:|Scrotum:|Epididymides:|Testes:|Penis:|Prostate:|GENITOURINARY:|Cystoscopy:|GENITAL:|PELVIC:|GENITALIA:'
patterns['gast_pat'] = 'GASTROSTOMY:'
patterns['other_pat'] = 'SERVICE:|COUNTS:|CPT CODE:|ICD9 CODE:|COMPONENTS USED:|TECHNIQUE:|MATERIAL:|SPECIFICATIONS:|ADDITIONAL DETAILS:|PROTOCOL:|SCOPE:|MICRO:|HARDWARE:|COMPLEXITY:|QUESTION:|ANSWER:|UNSIGNED_DATA:'

other_patterns = None
other_patterns = {
 'unit_pat_others': r'(?:\d+(?:\.\d+)?\s*)?(\b(?:mmol/L|pg|fL|%|cm|mm|hr|min|dose)\b)'
}

tfidf_patterns = [re.compile('|'.join([el for el in i if not el.isdigit() and set(el)!={'_'}])).pattern for i in checks]
tfidf_names = ['check_g', 'check_n', 'check_o', 'check_r', 'check_u']
for k,name in enumerate(tfidf_names):
    patterns[name] = tfidf_patterns[k]




joblib.dump(patterns,'patterns.joblib')
joblib.dump(other_patterns,'other_patterns.joblib')


for k,name in enumerate(tfidf_names):
    print(tfidf_patterns[k][:50])


np.unique([len(i) for i in tfidf_patterns[k].split('|')],return_counts=True)


# High-frequency words
p_keys = list(patterns.keys())
med_records_df['Record_types'] = med_records_df['Record'].apply(lambda x: np.hstack([np.array(re.findall(patterns[i],x)).reshape(-1) for i in p_keys]))
r_types = set(np.hstack(med_records_df['Record_types'].values))
r_types = list(set([i.replace(':','',).strip() for i in r_types]))
r_types = [i for k,i in enumerate(r_types) if i!='' and i.upper() not in [i.upper() for i in r_types[:k]]]
r_types_df = pd.DataFrame(list(med_records_df['Record_types'].apply(lambda x: [len(re.findall(r_type.upper(),'###'.join(x).upper())) if r_type.upper() in '###'.join(x).upper() else 0 for r_type in r_types]).values)).set_axis(r_types,axis=1)

med_records_df = pd.concat([med_records_df,r_types_df],axis=1)

p_keys_df = pd.DataFrame(list(med_records_df['Record'].apply(lambda x: [len(re.findall(patterns[i],x)) if len(re.findall(patterns[i],x))!=0 else 0 for i in p_keys]).values)).set_axis(p_keys,axis=1)
med_records_df = pd.concat([med_records_df,p_keys_df],axis=1)

# Specific patterns
if other_patterns!=None:
    p_keys = list(other_patterns.keys())
    p_keys_df = pd.DataFrame(list(med_records_df['Record'].apply(lambda x: [len(re.findall(other_patterns[i],x)) if len(re.findall(other_patterns[i],x))!=0 else 0 for i in p_keys]).values)).set_axis(p_keys,axis=1)
    med_records_df = pd.concat([med_records_df,p_keys_df],axis=1)
    r_types = r_types + list(other_patterns.keys())
    p_keys = list(patterns.keys()) + list(other_patterns.keys())


# Fill empty cells
med_records_df[p_keys] = med_records_df[p_keys].fillna(0)
med_records_df[r_types] = med_records_df[r_types].fillna(0)

# Aggregations
_ = np.array(list(product(med_records_df[p_keys].columns, repeat=2)))
for i in _[np.argwhere(_[:,0]!=_[:,1])].reshape((-1,2)):
    med_records_df[i[0]+'_'+i[1]] = med_records_df[i].T.apply(lambda x: x.prod() if x.all()>0 else x.sum())/med_records_df[i].sum(axis=1) # the higher value corresponds to fewer difference

# simple sum of each word or in groups of patterns
cols_sum = lambda cols: med_records_df[cols].sum(axis=1)
med_records_df['R_types_Sum'] = cols_sum(r_types)
med_records_df['Info'] = cols_sum(['his_pat', 'exam_pat','course_pat','rec_pat', 'other_pat'])
med_records_df['Invest'] = cols_sum(['lab_pat', 'img_pat'])
med_records_df['Treat'] = cols_sum(['med_pat', 'inf_pat'])
med_records_df['Proc'] = cols_sum(['sur_pat', 'neuro_pat', 'uro_pat','gast_pat'])
med_records_df['P_keys_Sum'] = cols_sum(['Info','Invest','Treat','Proc'])
med_records_df['P_keys_Sum_all'] = cols_sum([i for i in p_keys if i.endswith('_pat')])
# The difference in patterns of words or in groups of patterns
# in log-product devided by sum
prod_sum_ratio = lambda cols: med_records_df[cols].apply(lambda x: np.log(np.prod(x.values[np.where(x.values>0)],dtype=float)),axis=1) / med_records_df[cols].sum(axis=1)
med_records_df['R_types_prod_sum_ratio'] = prod_sum_ratio(r_types)
med_records_df['Info_prod_sum_ratio'] = prod_sum_ratio(['his_pat', 'exam_pat','course_pat','rec_pat', 'other_pat'])
med_records_df['Invest_prod_sum_ratio'] = prod_sum_ratio(['lab_pat', 'img_pat'])
med_records_df['Treat_prod_sum_ratio'] = prod_sum_ratio(['med_pat', 'inf_pat'])
med_records_df['Proc_prod_sum_ratio'] = prod_sum_ratio(['sur_pat', 'neuro_pat', 'uro_pat','gast_pat'])
med_records_df['P_keys_prod_sum_ratio'] = prod_sum_ratio(['Info','Invest','Treat','Proc'])
med_records_df['P_keys_Sum_all_ratio'] = prod_sum_ratio([i for i in p_keys if i.endswith('_pat')])
# The difference in group of patterns in
# ratio of every value in total sum
_ = med_records_df[[i for i in p_keys if i.endswith('_pat')]].fillna(0)
_[[i+'type_ratio' for i in (_.T / _.sum(axis=1)).T.columns]] = (_.T / _.sum(axis=1)).T
med_records_df[_.filter(like='_ratio').columns] = _.filter(like='_ratio').values
p_keys += list(_.filter(like='_ratio').columns)
print('The total number of nan-values')
display(med_records_df.isna().sum().sum())
med_records_df = med_records_df.fillna(0)
med_records_df.head()


p_keys = p_keys + ['R_types_Sum','Info','Invest','Treat','Proc','P_keys_Sum','P_keys_Sum_all','R_types_prod_sum_ratio','Info_prod_sum_ratio','Invest_prod_sum_ratio','Treat_prod_sum_ratio','Proc_prod_sum_ratio','P_keys_prod_sum_ratio','P_keys_Sum_all_ratio']
p_keys = list(set(p_keys))
walker_df = pd.merge(med_records_df[['Filename']+p_keys],train_labels_df,on='Filename').drop('Filename',axis=1).groupby('Label').sum()
display(walker_df.T.astype(int).style.background_gradient(subset=walker_df.T.columns,cmap='gray_r',axis=1).set_caption('<b>Tags frequency by Labels</b>')\
    .set_table_styles([{'selector': 'td', 'props': 'text-align: center; font-weight: bold; font-size:18px'}]))

display((walker_df/walker_df.sum()).fillna(0).T.astype(float).style.background_gradient(subset=(walker_df/walker_df.sum()).T.columns,cmap='gray_r',axis=1).set_caption('<b>Tags frequency: Labels in Patterns</b>')\
    .set_table_styles([{'selector': 'td', 'props': 'text-align: center; font-weight: bold; font-size:18px'}]))

walker_df = pd.merge(med_records_df[['Filename']+p_keys],train_labels_df,on='Filename').drop('Filename',axis=1)
_cols = walker_df[p_keys].apply(lambda x: x.values/walker_df[p_keys].T.sum(axis=0))
walker_df[_cols.columns] = _cols
display(walker_df.groupby('Label')[p_keys].mean().T.style.background_gradient(subset=train_labels_df['Label'].unique(),cmap='gray_r',axis=1).set_caption('<b>Tags frequency: Patterns in Labels</b>')\
    .set_table_styles([{'selector': 'td', 'props': 'text-align: center; font-weight: bold; font-size:18px'}]))



pd.options.display.max_colwidth = 10000
pd.options.display.max_rows = 10000
X = pd.merge(med_records_df.loc[med_records_df[r_types].sum(axis=1)==0],train_labels_df,on='Filename')
X[['Record','Label']].shape


X.shape


med_records_df.shape


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

med_records_df.rename(columns = dict(map(lambda x: (x,'dict_'+x.upper()),med_records_df.columns[2:])),inplace=True)
X_train.rename(columns = dict(map(lambda x: (x,'tfidf_'+x.upper()),X_train.columns[3:])),inplace=True)
X_test.rename(columns = dict(map(lambda x: (x,'tfidf_'+x.upper()),X_test.columns[3:])),inplace=True)
vec_cols = list(X_test.columns[3:])
r_types = list(med_records_df.columns[2:])
med_rec_cols = ['Filename','Record']+list(set(med_records_df.columns) - set(X_train.columns))
X_train = pd.merge(X_train, med_records_df[med_rec_cols].drop(['Record'],axis=1),on='Filename')
X_test = pd.merge(X_test,med_records_df[med_rec_cols].drop(['Record'],axis=1),on='Filename')


len(med_rec_cols)


X_test.shape[1]


X_test.columns


lrc = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
scaler = StandardScaler()
X_train_ = scaler.fit_transform(X_train[vec_cols])
lrc.fit(X_train_,y_train)
X_test_ = scaler.transform(X_test[vec_cols])
y_hat = lrc.predict(X_test_)
print('Test base model')
f1 = f1_score(y_hat, y_test, average = "macro")
print("Macro-Average F1: ", f1)
accuracy = accuracy_score(y_hat, y_test)
print("Accuracy: ", accuracy)



# COUNTBASED WORD SELECTION
aggregated_text_by_label = X_train.groupby('Label')['Record'].agg(' '.join)
words_dfs = pd.DataFrame(aggregated_text_by_label.str.lower().str.split().apply(lambda x: Counter(x).most_common(5000))).explode('Record').reset_index()
words_dfs[['Record','Count']] = list(words_dfs['Record'].apply(lambda x: np.array([x[0],x[1]])).values)
words_dfs_pivot = pd.pivot_table(index='Record',columns='Label',data=words_dfs,aggfunc=sum).fillna(0)
words_dfs_pivot = words_dfs_pivot.droplevel(0,axis=1)
words_dfs_pivot['Sum'] = words_dfs_pivot[words_dfs_pivot.astype(int)>0].count(axis=1)
_ = words_dfs_pivot.query('Sum < 4') # the threshold for interception
_ = _.reset_index()

gatr_w = _.loc[_['Gastroenterology']!=0,['Record','Gastroenterology']].set_axis(['Record','G_words'],axis=1)
neur_w = _.loc[_['Neurology']!=0,['Record','Neurology']].set_axis(['Record','N_words'],axis=1)
radio_w = _.loc[_['Radiology']!=0,['Record','Radiology']].set_axis(['Record','R_words'],axis=1)
orth_w = _.loc[_['Orthopedic']!=0,['Record','Orthopedic']].set_axis(['Record','O_words'],axis=1)
urol_w = _.loc[_['Urology']!=0,['Record','Urology']].set_axis(['Record','U_words'],axis=1)

X_train['G_words'] = X_train['Record'].apply(lambda x: [_ for _ in clean_text(x).split() if _ in gatr_w['Record'].values])
X_train['N_words'] = X_train['Record'].apply(lambda x: [_ for _ in clean_text(x).split() if _ in neur_w['Record'].values])
X_train['R_words'] = X_train['Record'].apply(lambda x: [_ for _ in clean_text(x).split() if _ in radio_w['Record'].values])
X_train['O_words'] = X_train['Record'].apply(lambda x: [_ for _ in clean_text(x).split() if _ in orth_w['Record'].values])
X_train['U_words'] = X_train['Record'].apply(lambda x: [_ for _ in clean_text(x).split() if _ in urol_w['Record'].values])
X_train['G_words_len'] = X_train['G_words'].apply(len)
X_train['N_words_len'] = X_train['N_words'].apply(len)
X_train['R_words_len'] = X_train['R_words'].apply(len)
X_train['O_words_len'] = X_train['O_words'].apply(len)
X_train['U_words_len'] = X_train['U_words'].apply(len)

X_test['G_words'] = X_test['Record'].apply(lambda x: [_ for _ in clean_text(x).split() if _ in gatr_w['Record'].values])
X_test['N_words'] = X_test['Record'].apply(lambda x: [_ for _ in clean_text(x).split() if _ in neur_w['Record'].values])
X_test['R_words'] = X_test['Record'].apply(lambda x: [_ for _ in clean_text(x).split() if _ in radio_w['Record'].values])
X_test['O_words'] = X_test['Record'].apply(lambda x: [_ for _ in clean_text(x).split() if _ in orth_w['Record'].values])
X_test['U_words'] = X_test['Record'].apply(lambda x: [_ for _ in clean_text(x).split() if _ in urol_w['Record'].values])
X_test['G_words_len'] = X_test['G_words'].apply(len)
X_test['N_words_len'] = X_test['N_words'].apply(len)
X_test['R_words_len'] = X_test['R_words'].apply(len)
X_test['O_words_len'] = X_test['O_words'].apply(len)
X_test['U_words_len'] = X_test['U_words'].apply(len)

X_train.iloc[:,-10:].head(10)


X_train.shape


X_test.shape


lrc = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
scaler = StandardScaler()
y_train = X_train['Label']
y_train = X_train['Label']
y_test = X_test['Label']

X_train_ = scaler.fit_transform(X_train.iloc[:,2:].drop(['Label'],axis=1).fillna(0).drop(['G_words','N_words','R_words','O_words','U_words','dict_RECORD_TYPES'],axis=1))
lrc.fit(X_train_,y_train)
X_test_ = scaler.transform(X_test.iloc[:,2:].drop(['Label'],axis=1).fillna(0).drop(['G_words','N_words','R_words','O_words','U_words','dict_RECORD_TYPES'],axis=1))
y_hat = lrc.predict(X_test_)




print('Test base model')
print("Macro-Average F1: ", f1_score(y_hat, y_test, average = "macro"))
print("Accuracy: ", accuracy_score(y_hat, y_test))


lrc = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
scaler = StandardScaler()
X_train_ = scaler.fit_transform(X_train.iloc[:,2:].drop(['Label','dict_'+'uro_pattype_ratio'.upper(),'dict_'+'Invest_prod_sum_ratio'.upper(),'dict_'+'rec_pattype_ratio'.upper(),'dict_'+'gast_pattype_ratio'.upper(),'dict_'+'gast_pat'.upper(),'dict_'+'inf_pattype_ratio'.upper()],axis=1).fillna(0).drop(['G_words','N_words','R_words','O_words','U_words','dict_RECORD_TYPES'],axis=1))
lrc.fit(X_train_,y_train)
X_test_ = scaler.transform(X_test.iloc[:,2:].drop(['Label','dict_'+'uro_pattype_ratio'.upper(),'dict_'+'Invest_prod_sum_ratio'.upper(),'dict_'+'rec_pattype_ratio'.upper(),'dict_'+'gast_pattype_ratio'.upper(),'dict_'+'gast_pat'.upper(),'dict_'+'inf_pattype_ratio'.upper()],axis=1).fillna(0).drop(['G_words','N_words','R_words','O_words','U_words','dict_RECORD_TYPES'],axis=1))
y_hat = lrc.predict(X_test_)
print('Test base model')
print("Macro-Average F1: ", f1_score(y_hat, y_test, average = "macro"))
print("Accuracy: ", accuracy_score(y_hat, y_test))


# deletion & dumping
X_train = X_train.drop(['dict_'+'uro_pattype_ratio'.upper(),'dict_'+'Invest_prod_sum_ratio'.upper(),'dict_'+'rec_pattype_ratio'.upper(),'dict_'+'gast_pattype_ratio'.upper(),'dict_'+'gast_pat'.upper(),'dict_'+'inf_pattype_ratio'.upper()],axis=1)
X_test = X_test.drop(['dict_'+'uro_pattype_ratio'.upper(),'dict_'+'Invest_prod_sum_ratio'.upper(),'dict_'+'rec_pattype_ratio'.upper(),'dict_'+'gast_pattype_ratio'.upper(),'dict_'+'gast_pat'.upper(),'dict_'+'inf_pattype_ratio'.upper()],axis=1)


r_type_remove = ['dict_'+'uro_pattype_ratio'.upper(),'dict_'+'Invest_prod_sum_ratio'.upper(),'dict_'+'rec_pattype_ratio'.upper(),'dict_'+'gast_pattype_ratio'.upper(),'dict_'+'gast_pat'.upper(),'dict_'+'inf_pattype_ratio'.upper()]
joblib.dump(X_test,'X_test_before_embedding.joblib')
joblib.dump([gatr_w,neur_w,radio_w,orth_w,urol_w],'word_counts.joblib')
med_records_df.to_pickle('med_records_df_28102025.pkl')

[r_types.remove(i) for i in r_type_remove]


r_types[:3]


X_train['Vec_text'] = X_train[vec_cols].T.apply(lambda x: ' '.join([re.sub('^tfidf_VEC_','',x.index[k]) for k,val in enumerate(x) if val>0]))
X_test['Vec_text'] = X_test[vec_cols].T.apply(lambda x: ' '.join([re.sub('^tfidf_VEC_','',x.index[k]) for k,val in enumerate(x) if val>0]))

X_train['Vec_r_types'] = X_train[r_types[1:]].T.apply(lambda x: ' '.join([re.sub('^dict_','',x.index[k]) for k,val in enumerate(x) if val>0]))
X_test['Vec_r_types'] = X_test[r_types[1:]].T.apply(lambda x: ' '.join([re.sub('^dict_','',x.index[k]) for k,val in enumerate(x) if val>0]))


X_train['Vec_rt_list'] = X_train['dict_RECORD_TYPES'].apply(lambda x: [i.replace(':','') for i in x if len(i)>3])
X_test['Vec_rt_list'] = X_test['dict_RECORD_TYPES'].apply(lambda x: [i.replace(':','') for i in x if len(i)>3])




cols_to_tensors = ['G_words','N_words','R_words','O_words','U_words',\
                   'Vec_text','Vec_r_types','Vec_rt_list']
tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)
BATCH_SIZE = 32
MAX_LEN = 512

class SingleColTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(text,add_special_tokens=True,
                                              max_length=self.max_len,padding='max_length',
                                              truncation=True,return_attention_mask=True,
                                              return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)}
def cols_to_embddings(df: pd.DataFrame,
                                  column_name: str,
                                  tokenizer, model,
                                  batch_size: int,
                                  max_len: int, device) -> np.ndarray:
    texts = df[column_name].apply(' '.join).to_list() \
            if isinstance(df[column_name].iloc[0], list) \
            else df[column_name].to_list()
    dataset = SingleColTextDataset(texts, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    column_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            column_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(column_embeddings)

def sim_scores_f(emds_3d):
    sim_scores_list = []
    for sample_idx, sample_embeddings in enumerate(emds_3d):
        sim_scores = []
        for col_idx, col_emb in enumerate(sample_embeddings):
            score = cs(col_emb.reshape(1,-1), word_embeddings_train_mean_per_col[col_idx].reshape(1,-1))[0][0]
            sim_scores.append(score)
        sim_scores_list.append(sim_scores)
    return sim_scores_list


def cols_to_tensors_f(data,cols):
    all_embs_by_col = {}
    for col_name in cols:
        all_embs_by_col[col_name] = cols_to_embddings(data, col_name, tokenizer, model, BATCH_SIZE, MAX_LEN, device)
    return all_embs_by_col

all_train_embeddings_by_col = cols_to_tensors_f(X_train,cols_to_tensors)
all_test_embeddings_by_col = cols_to_tensors_f(X_test,cols_to_tensors)

train_embeddings_3d = np.stack([all_train_embeddings_by_col[col] for col in cols_to_tensors], axis=1)
test_embeddings_3d = np.stack([all_test_embeddings_by_col[col] for col in cols_to_tensors], axis=1)
word_embeddings_train_mean_per_col = train_embeddings_3d.mean(axis=0)
sim_col_names = [f'Sim_{col}' for col in cols_to_tensors]

train_sim_scores_list = sim_scores_f(train_embeddings_3d)
test_sim_scores_list = sim_scores_f(test_embeddings_3d)

torch.cuda.empty_cache()
gc.collect()


joblib.dump(word_embeddings_train_mean_per_col,'word_embeddings_train_mean_per_col.joblib')


print(X_train.shape)
to_do_update = True
if to_do_update:
    X_train[sim_col_names] = ''
    X_test[sim_col_names] = ''
    X_train[sim_col_names] = train_sim_scores_list
    X_test[sim_col_names] = test_sim_scores_list
    X_train[sim_col_names].to_pickle('X_train_sim_col_names.pkl')
    X_test[sim_col_names].to_pickle('X_test_sim_col_names.pkl')
elif len([_ for _ in X_train.columns if 'Sim_' in _])==0:
    X_train = pd.concat([X_train,pd.read_pickle('X_train_sim_col_names.pkl')],axis=1)
    X_test = pd.concat([X_test,pd.read_pickle('X_test_sim_col_names.pkl')],axis=1)
elif  len([_ for _ in X_train.columns if 'Sim_' in _])==5:
    pass
else:
    print('ERROR!!!')

display(pd.read_pickle('X_test_sim_col_names.pkl').head())
sim_col_names = list(X_test.filter(like='Sim_').columns)
if 'G_words' in list(X_train.columns):
    X_train = X_train.drop(['dict_RECORD_TYPES']+cols_to_tensors,axis=1)
    X_test = X_test.drop(['dict_RECORD_TYPES']+cols_to_tensors,axis=1)
sim_col_names


pd.DataFrame(np.unique(X_train.columns,return_counts=True)).T.sort_values(1)


sns.boxplot(data=X_train.sort_values('Label'),y=X_train.sort_values('Label')[sim_col_names].sum(axis=1),x='Label')
sns.boxplot(data=X_test.sort_values('Label'),y=X_test.sort_values('Label')[sim_col_names].sum(axis=1),x='Label',width=0.6)
plt.show()
sns.boxplot(data=X_train.sort_values('Label'),y ='Sim_Vec_text',x='Label')
sns.boxplot(data=X_test.sort_values('Label'),y='Sim_Vec_text',x='Label',width=0.6)
plt.show()
sns.boxplot(data=X_train.sort_values('Label'),y='Sim_Vec_r_types',x='Label')
sns.boxplot(data=X_test.sort_values('Label'),y='Sim_Vec_r_types',x='Label',width=0.6)
plt.show()
sns.boxplot(data=X_train.sort_values('Label'),y='Sim_Vec_rt_list',x='Label')
sns.boxplot(data=X_test.sort_values('Label'),y='Sim_Vec_rt_list',x='Label',width=0.6)
plt.show()



pd.DataFrame(X_test.dtypes).reset_index().groupby(0).agg(list)


sns.histplot(data=X_train,x='Sim_G_words',hue='Label')
plt.show()
sns.histplot(data=X_train,x='Sim_N_words',hue='Label')
plt.show()
sns.histplot(data=X_train,x='Sim_R_words',hue='Label')
plt.show()
sns.histplot(data=X_train,x='Sim_O_words',hue='Label')
plt.show()
sns.histplot(data=X_train,x='Sim_U_words',hue='Label')
plt.show()
sns.histplot(data=X_train,x='Sim_Vec_text',hue='Label')
plt.show()
sns.histplot(data=X_train,x='Sim_Vec_r_types',hue='Label')
plt.show()
sns.histplot(data=X_train,x='Sim_Vec_rt_list',hue='Label')
plt.show()


sns.pairplot(X_train[['Sim_G_words', 'Sim_N_words', 'Sim_R_words', 'Sim_O_words', 'Sim_U_words', 'Sim_Vec_text', 'Sim_Vec_r_types', 'Sim_Vec_rt_list','Label']],hue='Label')


X_train[['Sim_G_words', 'Sim_N_words', 'Sim_R_words', 'Sim_O_words', 'Sim_U_words', 'Sim_Vec_text', 'Sim_Vec_r_types', 'Sim_Vec_rt_list','Label']].info()


formula = f'Label ~ {" + ".join(["Sim_G_words", "Sim_N_words", "Sim_R_words", "Sim_O_words", "Sim_U_words", "Sim_Vec_text", "Sim_Vec_r_types", "Sim_Vec_rt_list"])}'
X = X_train[['Sim_G_words', 'Sim_N_words', 'Sim_R_words', 'Sim_O_words', 'Sim_U_words', 'Sim_Vec_text', 'Sim_Vec_r_types', 'Sim_Vec_rt_list','Label']]
X['Label'] = X['Label'].astype('category').cat.codes
models = [1,2,3,4,5]
X = pd.concat([X.query(f'Label == {i}') for i in range(5)])

for i in range(5):
    print('Base class =',i)
    model = mnlogit(formula,data=X)
    model = model.fit(method='powell')
    display(model.summary())
    models[i] = model
    if i < 4:
        X.loc[X['Label']==0] = X.loc[X['Label']==0]+5
        X['Label'] = X['Label']-1



label_cats = pd.DataFrame(X_train.Label.astype('category').cat.codes.astype(str)).assign(Code=X_train['Label']).value_counts()
label_cats.sort_index()


five_vars = [list(range(5))]
[five_vars.append(five_vars[-1][1:]+[five_vars[-1][0]]) for i in range(4)]
rename_dict = dict(label_cats.index)
for k,mod_sum in enumerate(models):
    print('Base model =',dict(label_cats.index)[str(k)])
    # use the same order of one-hot-encodded labels
    model_labels = five_vars[k][1:]
    cond_int_df = pd.pivot_table(mod_sum.conf_int().fillna(0).reset_index(),index=['level_1'],columns=['Label'],aggfunc='mean')
    # columns renaming
    model_labels = [rename_dict[str(i)] for i in model_labels]
    rename_cols = dict(zip(list(cond_int_df.columns.levels[1]),model_labels))
    # calculate difference to plot
    cond_int_df_diff = cond_int_df.filter(like='upper').values - cond_int_df.filter(like='lower').values
    cond_int_df = pd.concat([cond_int_df,cond_int_df.filter(like='upper').rename(level=0,columns={'upper':'diff'})],axis=1)
    cond_int_df.rename(level=1,columns=rename_cols,inplace=True)
    cond_int_df['diff'] = cond_int_df_diff
    cond_int_df.index.name = None
    display(mod_sum.params.set_axis(model_labels,axis=1).style.set_caption('Coefficients',).set_table_styles([dict(selector="caption",props=[('font-size','200%')])]))
    display(cond_int_df.style.background_gradient('Blues').set_caption('Conf. Intervals',).set_table_styles([{'selector':"caption",'props':'font-size:200%'},
                                                                 {'selector':'th','props':'text-align:center; border: 1px dashed grey'}]))
    sc = StandardScaler()
    fig,axes = plt.subplots(1,4,figsize=(20,5))
    print(model_labels)
    for k_i,model_label in enumerate(model_labels):
        print(model_label)
        sns.boxplot(sc.fit_transform(cond_int_df.filter(like=model_label)),ax=axes[k_i])
        axes[k_i].set_title(model_label)
        axes[k_i].set_ylim(-3,3)
    plt.show()


from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import polars as pl
stratifier = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)



xi_df = []
X_train = pl.DataFrame(X_train)
for col in X_train.columns[3:]:
    xi = st.chi2_contingency(pd.crosstab(X_train.select(col),X_train.select('Label'),margins=False))
    xi_df.append([col,
                 f'{st.chi2_contingency(pd.crosstab(X_train.select(col),X_train.select('Label'),margins=False)).statistic:.18f}',
                 f'{st.chi2_contingency(pd.crosstab(X_train.select(col),X_train.select('Label'),margins=False)).pvalue:.18f}'])
xidf = pl.DataFrame(np.array(xi_df),schema=['Feature',('Xi2_statistics',pl.Float32),('Xi2_pvalue',pl.Float32)])
xidf_list = xidf.sort('Xi2_statistics',descending=[True])['Feature'].head(100).to_list()



lencoder = LabelEncoder()
lencoder.fit_transform(X_train['Label'])
mic_deps = mutual_info_classif(X_train.drop(X_train.columns[:3]),lencoder.fit_transform(X_train['Label']),random_state=42)
mic_deps_list = pl.DataFrame([mic_deps,X_train.drop(X_train.columns[:3]).columns]).sort('column_0',descending=[True]).head(100)['column_1'].to_list()

common_list = set(xidf_list) & set(mic_deps_list)
X = X_train[list(common_list)]
Y = X_train['Label']
len(common_list)


rf_clf_5 = RandomForestClassifier(random_state=42,max_features=0.3,n_estimators=100,max_depth=5,criterion= 'entropy',n_jobs=-1,max_samples=0.25)
rf_clf_5.fit(X,Y)

pl.DataFrame([rf_clf_5.feature_importances_,list(common_list)]).sort('column_0',descending=True)

preds = rf_clf_5.predict(X_test[list(common_list)])

print('precision_recall_fscore_support')
print(precision_recall_fscore_support(X_test['Label'],preds,average='weighted')[:3])

cm = confusion_matrix(X_test['Label'],preds)
cm = ConfusionMatrixDisplay(cm,display_labels=X_test['Label'].unique())
cm.plot(cmap='Blues',xticks_rotation=45,)



preds_train = rf_clf_5.predict(X_train[list(common_list)])



print(precision_recall_fscore_support(X_train['Label'],preds_train,average='weighted')[:3])

cm = confusion_matrix(X_train['Label'],preds_train)
cm = ConfusionMatrixDisplay(cm,display_labels=X_train['Label'].unique())
cm.plot(cmap='Blues',xticks_rotation=45,)



X = X_train[list(common_list)]
Y = X_train['Label']
X_test_scaled = X_test[list(common_list)].copy()
sc = StandardScaler()
X[X.columns] = sc.fit_transform(X)
X_test_scaled[X.columns] = sc.transform(X_test_scaled)


metrics = ['accuracy','precision','recall','f1']
lr_clf = LogisticRegression(random_state=42)
params = {'C':[0.2,0.6,0.8,1]}
lr_gs_ur = GridSearchCV(lr_clf,param_grid=params,cv=stratifier,scoring=metrics,refit='f1')
lr_gs_or = GridSearchCV(lr_clf,param_grid=params,cv=stratifier,scoring=metrics,refit='f1')
lr_gs_ga = GridSearchCV(lr_clf,param_grid=params,cv=stratifier,scoring=metrics,refit='f1')
cb_clf = CatBoostClassifier(random_state=42,verbose=False)
params = {'learning_rate':[0.0001,0.005,0.001],'max_depth':[5,7,10],'n_estimators':[100]}
cb_gs = GridSearchCV(cb_clf,param_grid=params,cv=stratifier,scoring=metrics,refit='f1')
ab_clf = AdaBoostClassifier(random_state=42)
params = {'n_estimators':[50,100,200,250],'learning_rate':[0.0001,0.005,0.01,0.03],'estimator':[lr_clf,None]}
ab_gs = GridSearchCV(ab_clf,param_grid=params,cv=stratifier,scoring=metrics,refit='f1')

lr_gs_ur.fit(X,Y=='Urology')
print('Urology')
lr_gs_or.fit(X,Y=='Orthopedic')
print('Orthopedic')
lr_gs_ga.fit(X,Y=='Gastroenterology')
print('Gastroenterology')
cb_gs.fit(X.to_numpy(),Y.to_numpy()=='Radiology')
print('Radiology')
ab_gs.fit(X,Y=='Neurology')
print('Neurology')



precision_recall_fscore_support(lr_gs_ur.predict(X_test_scaled),X_test['Label']=='Urology',average='weighted')


precision_recall_fscore_support(lr_gs_or.predict(X_test_scaled),X_test['Label']=='Orthopedic',average='weighted')


precision_recall_fscore_support(lr_gs_ga.predict(X_test_scaled),X_test['Label']=='Gastroenterology',average='weighted')


precision_recall_fscore_support(cb_gs.predict(X_test_scaled),X_test['Label']=='Radiology')


precision_recall_fscore_support(ab_gs.predict(X_test_scaled),X_test['Label']=='Neurology')





lengths = [10,50,100,150,200,300,400,500]
common_lists = []
for i in lengths:
    # Chi-test
    xi_df = []
    for col in X_train.columns[3:]:
        xi_df.append([col,
                     f'{st.chi2_contingency(pd.crosstab(X_train[col],X_train['Label']=='Radiology',margins=False)).statistic:.18f}',
                     f'{st.chi2_contingency(pd.crosstab(X_train[col],X_train['Label']=='Radiology',margins=False)).pvalue:.18f}'])
    xidf = pl.DataFrame(np.array(xi_df),schema=['Feature',('Xi2_statistics',pl.Float32),('Xi2_pvalue',pl.Float32)])
    xidf_list = xidf.sort('Xi2_statistics',descending=[True])['Feature'].head(i).to_list()
    
    # Labels Dependencies estimation with mutual_info_classif
    lencoder = LabelEncoder()
    lencoder.fit_transform(X_train['Label'])
    mic_deps = mutual_info_classif(X_train.drop(X_train.columns[:3]),lencoder.fit_transform(X_train['Label']=='Radiology'),random_state=42)
    mic_deps_list = pl.DataFrame([mic_deps,X_train.drop(X_train.columns[:3]).columns]).sort('column_0',descending=[True]).head(i)['column_1'].to_list()
    
    # feature list
    common_list = set(mic_deps_list) & set(xidf_list)
    X = X_train[list(common_list)]
    Y = X_train['Label']
    
    common_lists.append(common_list)
list(common_list)[:10],len(list(common_list))


plot_nums = len([i for i in common_lists if len(list(i))!=0])
figure,axes = plt.subplots(1,plot_nums,figsize=(10,4))
ll=0
for l,common_list in enumerate(common_lists):
    if len(common_list)==0:
        ll+=1
        continue
    rf_clf = RandomForestClassifier(random_state=42,max_features=0.2,max_leaf_nodes=5,n_estimators=120,max_depth=7,criterion= 'gini',n_jobs=-1,max_samples=0.6)
    rf_clf.fit(X_train[list(common_list)],Y)
    preds = rf_clf.predict(X_test[list(common_list)])
    print('precision_recall_fscore_support')
    print(precision_recall_fscore_support(X_test['Label']=='Radiology',preds=='Radiology',average='weighted')[:3])
    # plot_confusion_matrix()
    cm = ConfusionMatrixDisplay.from_predictions(X_test['Label']=='Radiology',preds=='Radiology',ax=axes[l-ll],colorbar=False)
    axes[l-ll].set_xticks([])
    axes[l-ll].set_yticks([])
    axes[l-ll].set_xlabel('')
    axes[l-ll].set_ylabel('')
    axes[l-ll].set_title(str(lengths[l])+', '+str(l))
plt.show()


common_list = common_lists[7]
display(pl.DataFrame([rf_clf.feature_importances_,list(common_lists[-1])]).sort('column_0',descending=True).head())


rf_clf.fit(X_train[list(common_list)],Y)
preds_train = rf_clf.predict(X_train[list(common_list)])
print(precision_recall_fscore_support(X_train['Label']=='Radiology',preds_train=='Radiology',average='weighted')[:3])

cm = confusion_matrix(X_train['Label']=='Radiology',preds_train=='Radiology')
cm = ConfusionMatrixDisplay(cm,display_labels=[0,1])
cm.plot(cmap='Blues',xticks_rotation=45,)



X = X_train[list(common_list)]
Y = X_train['Label']
X_test_scaled = X_test[list(common_list)].copy()
sc = StandardScaler()
X[X.columns] = sc.fit_transform(X)
X_test_scaled[X.columns] = sc.transform(X_test_scaled)

metrics = ['accuracy','precision','recall','f1']
cb_clf = CatBoostClassifier(random_state=42,verbose=False,)
params = {'learning_rate':[0.0015,0.001,0.00095],'max_depth':[8,10,12],'n_estimators':[100]}
cb_gs = GridSearchCV(cb_clf,param_grid=params,cv=stratifier,scoring=metrics,refit='f1')
cb_gs.fit(X.to_numpy(),Y.to_numpy()=='Radiology')
cb_gs.best_estimator_.set_feature_names(list(common_list))
print('Radiology')

precision_recall_fscore_support(cb_gs.predict(X_test_scaled),X_test['Label']=='Radiology',average='weighted')[:3],\
precision_recall_fscore_support(cb_gs.predict(X_test_scaled),X_test['Label']=='Radiology',average='binary')[:3]


print('Features count:')
print(len(list(common_list)))


lengths = [10,50,100,150,200,300,400,500]
common_lists = []
for i in lengths:
    # Chi-test
    xi_df = []
    for col in X_train.columns[3:]:
        xi = st.chi2_contingency(pd.crosstab(X_train[col],X_train['Label']=='Gastroenterology',margins=False))
        xi_df.append([col,
                     f'{xi.statistic:.18f}',
                     f'{xi.pvalue:.18f}'])
    xidf = pl.DataFrame(np.array(xi_df),schema=['Feature',('Xi2_statistics',pl.Float32),('Xi2_pvalue',pl.Float32)])
    xidf_list = xidf.sort('Xi2_statistics',descending=[True])['Feature'].head(i).to_list()
    
    # Labels Dependencies estimation with mutual_info_classif
    lencoder = LabelEncoder()
    lencoder.fit_transform(X_train['Label'])
    mic_deps = mutual_info_classif(X_train.drop(X_train.columns[:3]),lencoder.fit_transform(X_train['Label']=='Gastroenterology'),random_state=42)
    mic_deps_list = pl.DataFrame([mic_deps,X_train.drop(X_train.columns[:3]).columns]).sort('column_0',descending=[True]).head(i)['column_1'].to_list()
    
    # feature list
    common_list = set(xidf_list) & set(mic_deps_list)
    X = X_train[list(common_list)]
    Y = X_train['Label']
    
    common_lists.append(common_list)
list(common_list)[:10],len(list(common_list))


plot_nums = len([i for i in common_lists if len(list(i))!=0])
figure,axes = plt.subplots(1,plot_nums,figsize=(10,4))
ll=0
for l,common_list in enumerate(common_lists):
    if len(common_list)==0:
        ll+=1
        continue
    rf_clf = RandomForestClassifier(random_state=42,max_features=0.6,n_estimators=150,max_depth=10,criterion= 'entropy',n_jobs=-1,max_samples=0.75)
    rf_clf.fit(X_train[list(common_list)],Y)
    preds = rf_clf.predict(X_test[list(common_list)])
    print('precision_recall_fscore_support')
    print(precision_recall_fscore_support(X_test['Label']=='Gastroenterology',preds=='Gastroenterology',average='weighted')[:3])
    # plot_confusion_matrix()
    cm = ConfusionMatrixDisplay.from_predictions(X_test['Label']=='Gastroenterology',preds=='Gastroenterology',ax=axes[l-ll],colorbar=False)
    axes[l-ll].set_xticks([])
    axes[l-ll].set_yticks([])
    axes[l-ll].set_xlabel('')
    axes[l-ll].set_ylabel('')
    axes[l-ll].set_title(str(lengths[l])+', '+str(l))
plt.show()


common_list = common_lists[6]
rf_clf = RandomForestClassifier(random_state=42,max_features=0.6,n_estimators=150,max_depth=10,criterion= 'entropy',n_jobs=-1,max_samples=0.75)
rf_clf.fit(X_train[list(common_list)],Y)
preds_train = rf_clf.predict(X_train[list(common_list)])
print(precision_recall_fscore_support(X_train['Label']=='Gastroenterology',preds_train=='Gastroenterology',average='weighted')[:3])

cm = confusion_matrix(X_train['Label']=='Gastroenterology',preds_train=='Gastroenterology')
cm = ConfusionMatrixDisplay(cm,display_labels=[0,1])
cm.plot(cmap='Blues',xticks_rotation=45,)



X = X_train[list(common_list)]
Y = X_train['Label']
X_test_scaled = X_test[list(common_list)].copy()
sc = StandardScaler()
X[X.columns] = sc.fit_transform(X)
X_test_scaled[X.columns] = sc.transform(X_test_scaled)

metrics = ['accuracy','precision','recall','f1']
params = {'C':[0.2,0.6,0.8,1]}
lr_gs_ga = GridSearchCV(LogisticRegression(random_state=42),param_grid=params,cv=stratifier,scoring=metrics,refit='f1')
lr_gs_ga.fit(X,Y=='Gastroenterology')
print('Gastroenterology')

precision_recall_fscore_support(lr_gs_ga.predict(X_test_scaled),X_test['Label']=='Gastroenterology',average='weighted')[:3],\
precision_recall_fscore_support(lr_gs_ga.predict(X_test_scaled),X_test['Label']=='Gastroenterology',average='binary')[:3]


lengths = [10,50,100,150,200,300,400,500]
common_lists = []
for i in lengths:
    # Chi-test
    xi_df = []
    for col in X_train.columns[3:]:
        xi = st.chi2_contingency(pd.crosstab(X_train[col],X_train['Label']=='Neurology',margins=False))
        xi_df.append([col,
                     f'{xi.statistic:.18f}',
                     f'{xi.pvalue:.18f}'])
    xidf = pl.DataFrame(np.array(xi_df),schema=['Feature',('Xi2_statistics',pl.Float32),('Xi2_pvalue',pl.Float32)])
    xidf_list = xidf.sort('Xi2_statistics',descending=[True])['Feature'].head(i).to_list()
    
    # Labels Dependencies estimation with mutual_info_classif
    lencoder = LabelEncoder()
    lencoder.fit_transform(X_train['Label'])
    mic_deps = mutual_info_classif(X_train.drop(X_train.columns[:3]),lencoder.fit_transform(X_train['Label']=='Neurology'),random_state=42)
    mic_deps_list = pl.DataFrame([mic_deps,X_train.drop(X_train.columns[:3]).columns]).sort('column_0',descending=[True]).head(i)['column_1'].to_list()
    
    # feature list
    common_list = set(mic_deps_list) & set(xidf_list)
    X = X_train[list(common_list)]
    Y = X_train['Label']
    
    common_lists.append(common_list)
list(common_list)[:10],len(list(common_list))


plot_nums = len([i for i in common_lists if len(list(i))!=0])
figure,axes = plt.subplots(1,plot_nums,figsize=(10,4))
ll=0
for l,common_list in enumerate(common_lists):
    if len(common_list)==0:
        ll+=1
        continue
    rf_clf = RandomForestClassifier(random_state=42,max_features=0.75,n_estimators=450,max_depth=5,criterion= 'entropy',n_jobs=-1,max_samples=1.0)
    rf_clf.fit(X_train[list(common_list)],Y)
    preds = rf_clf.predict(X_test[list(common_list)])
    print('precision_recall_fscore_support')
    print(precision_recall_fscore_support(X_test['Label']=='Neurology',preds=='Neurology',average='weighted')[:3])
    # plot_confusion_matrix()
    cm = ConfusionMatrixDisplay.from_predictions(X_test['Label']=='Neurology',preds=='Neurology',ax=axes[l-ll],colorbar=False)
    axes[l-ll].set_xticks([])
    axes[l-ll].set_yticks([])
    axes[l-ll].set_xlabel('')
    axes[l-ll].set_ylabel('')
    axes[l-ll].set_title(str(lengths[l])+', '+str(l))
plt.show()


common_list = common_lists[6]
rf_clf = RandomForestClassifier(random_state=42,max_features=0.75,n_estimators=450,max_depth=5,criterion= 'entropy',n_jobs=-1,max_samples=1.0)
rf_clf.fit(X_train[list(common_list)],Y)
preds_train = rf_clf.predict(X_train[list(common_list)])
print(precision_recall_fscore_support(X_train['Label']=='Neurology',preds_train=='Neurology',average='weighted')[:3])

cm = confusion_matrix(X_train['Label']=='Neurology',preds_train=='Neurology')
cm = ConfusionMatrixDisplay(cm,display_labels=[0,1])
cm.plot(cmap='Blues',xticks_rotation=45,)




X = X_train[list(common_list)]
Y = X_train['Label']
X_test_scaled = X_test[list(common_list)].copy()
sc = StandardScaler()
X[X.columns] = sc.fit_transform(X)
X_test_scaled[X.columns] = sc.transform(X_test_scaled)

metrics = ['accuracy','precision','recall','f1']
params = {'C':[0.2,0.6,0.8,1]}
ab_clf = AdaBoostClassifier(random_state=42)
params = {'n_estimators':[50,100,200,250],'learning_rate':[0.0001,0.005,0.01,0.03],'estimator':[lr_clf,None]}
ab_gs = GridSearchCV(ab_clf,param_grid=params,cv=stratifier,scoring=metrics,refit='f1')

ab_gs.fit(X,Y=='Neurology')
print('Neurology')

precision_recall_fscore_support(ab_gs.predict(X_test_scaled),X_test['Label']=='Neurology',average='weighted')[:3],\
precision_recall_fscore_support(ab_gs.predict(X_test_scaled),X_test['Label']=='Neurology',average='binary')[:3]


lengths = [10,50,100,150,200,300,400,500]
common_lists = []
for i in lengths:
    # Chi-test
    xi_df = []
    for col in X_train.columns[3:]:
        xi = st.chi2_contingency(pd.crosstab(X_train[col],X_train['Label']=='Urology',margins=False))
        xi_df.append([col,
                     f'{xi.statistic:.18f}',
                     f'{xi.pvalue:.18f}'])
    xidf = pl.DataFrame(np.array(xi_df),schema=['Feature',('Xi2_statistics',pl.Float32),('Xi2_pvalue',pl.Float32)])
    xidf_list = xidf.sort('Xi2_statistics',descending=[True])['Feature'].head(i).to_list()
    
    # Labels Dependencies estimation with mutual_info_classif
    lencoder = LabelEncoder()
    lencoder.fit_transform(X_train['Label'])
    mic_deps = mutual_info_classif(X_train.drop(X_train.columns[:3]),lencoder.fit_transform(X_train['Label']=='Urology'),random_state=42)
    mic_deps_list = pl.DataFrame([mic_deps,X_train.drop(X_train.columns[:3]).columns]).sort('column_0',descending=[True]).head(i)['column_1'].to_list()
    
    # feature list
    common_list = set(xidf_list) & set(mic_deps_list)
    X = X_train[list(common_list)]
    Y = X_train['Label']
    
    common_lists.append(common_list)
list(common_list)[:10],len(list(common_list))


plot_nums = len([i for i in common_lists if len(list(i))!=0])
figure,axes = plt.subplots(1,plot_nums,figsize=(10,4))
ll=0
for l,common_list in enumerate(common_lists):
    if len(common_list)==0:
        ll+=1
        continue
    rf_clf = RandomForestClassifier(random_state=42,max_features=1.0,n_estimators=50,max_depth=6,n_jobs=-1,max_samples=1.0)
    rf_clf.fit(X_train[list(common_list)],Y)
    preds = rf_clf.predict(X_test[list(common_list)])
    print('precision_recall_fscore_support')
    print(precision_recall_fscore_support(X_test['Label']=='Urology',preds=='Urology',average='weighted')[:3])
    # plot_confusion_matrix()
    cm = ConfusionMatrixDisplay.from_predictions(X_test['Label']=='Urology',preds=='Urology',ax=axes[l-ll],colorbar=False)
    axes[l-ll].set_xticks([])
    axes[l-ll].set_yticks([])
    axes[l-ll].set_xlabel('')
    axes[l-ll].set_ylabel('')
    axes[l-ll].set_title(str(lengths[l])+', '+str(l))
plt.show()


common_list = common_lists[7]
rf_clf = RandomForestClassifier(random_state=42,max_features=1.0,n_estimators=50,max_depth=6,n_jobs=-1,max_samples=1.0)
rf_clf.fit(X_train[list(common_list)],Y)
preds_train = rf_clf.predict(X_train[list(common_list)])
print(precision_recall_fscore_support(X_train['Label']=='Urology',preds_train=='Urology',average='weighted')[:3])

cm = confusion_matrix(X_train['Label']=='Urology',preds_train=='Urology')
cm = ConfusionMatrixDisplay(cm,display_labels=[0,1])
cm.plot(cmap='Blues',xticks_rotation=45,)



X = X_train[list(common_list)]
Y = X_train['Label']
X_test_scaled = X_test[list(common_list)].copy()
sc = StandardScaler()
X[X.columns] = sc.fit_transform(X)
X_test_scaled[X.columns] = sc.transform(X_test_scaled)


metrics = ['accuracy','precision','recall','f1']
params = {'C':[0.2,0.6,0.8,1]}
lr_gs_ur = GridSearchCV(LogisticRegression(random_state=42),param_grid=params,cv=stratifier,scoring=metrics,refit='f1')
lr_gs_ur.fit(X,Y=='Urology')
print('Urology')


precision_recall_fscore_support(lr_gs_ur.predict(X_test_scaled),X_test['Label']=='Urology',average='weighted')[:3],\
precision_recall_fscore_support(lr_gs_ur.predict(X_test_scaled),X_test['Label']=='Urology',average='binary')[:3]


lengths = [10,50,100,150,200,300,400,500]
common_lists = []
for i in lengths:
    # Chi-test
    xi_df = []
    for col in X_train.columns[3:]:
        xi = st.chi2_contingency(pd.crosstab(X_train[col],X_train['Label']=='Orthopedic',margins=False))
        xi_df.append([col,
                     f'{xi.statistic:.18f}',
                     f'{xi.pvalue:.18f}'])
    xidf = pl.DataFrame(np.array(xi_df),schema=['Feature',('Xi2_statistics',pl.Float32),('Xi2_pvalue',pl.Float32)])
    xidf_list = xidf.sort('Xi2_statistics',descending=[True])['Feature'].head(i).to_list()
    
    # Labels Dependencies estimation with mutual_info_classif
    lencoder = LabelEncoder()
    lencoder.fit_transform(X_train['Label'])
    mic_deps = mutual_info_classif(X_train.drop(X_train.columns[:3]),lencoder.fit_transform(X_train['Label']=='Orthopedic'),random_state=42)
    mic_deps_list = pl.DataFrame([mic_deps,X_train.drop(X_train.columns[:3]).columns]).sort('column_0',descending=[True]).head(i)['column_1'].to_list()
    
    # feature list
    common_list = set(xidf_list) & set(mic_deps_list)
    X = X_train[list(common_list)]
    Y = X_train['Label']
    
    common_lists.append(common_list)
list(common_list)[:10],len(list(common_list))


plot_nums = len([i for i in common_lists if len(list(i))!=0])
figure,axes = plt.subplots(1,plot_nums,figsize=(10,4))
ll=0
for l,common_list in enumerate(common_lists):
    if len(common_list)==0:
        ll+=1
        continue
    rf_clf = RandomForestClassifier(random_state=42,max_features=0.2,n_estimators=500,max_depth=5,criterion= 'entropy',n_jobs=-1,max_samples=0.3)
    rf_clf.fit(X_train[list(common_list)],Y)
    preds = rf_clf.predict(X_test[list(common_list)])
    print('precision_recall_fscore_support')
    print(precision_recall_fscore_support(X_test['Label']=='Orthopedic',preds=='Orthopedic',average='weighted')[:3])
    # plot_confusion_matrix()
    cm = ConfusionMatrixDisplay.from_predictions(X_test['Label']=='Orthopedic',preds=='Orthopedic',ax=axes[l-ll],colorbar=False)
    axes[l-ll].set_xticks([])
    axes[l-ll].set_yticks([])
    axes[l-ll].set_xlabel('')
    axes[l-ll].set_ylabel('')
    axes[l-ll].set_title(str(lengths[l])+', '+str(l))
plt.show()


common_list = common_lists[7]
rf_clf = RandomForestClassifier(random_state=42,max_features=0.2,n_estimators=500,max_depth=5,criterion= 'entropy',n_jobs=-1,max_samples=0.3)
rf_clf.fit(X_train[list(common_list)],Y)
preds_train = rf_clf.predict(X_train[list(common_list)])
print(precision_recall_fscore_support(X_train['Label']=='Orthopedic',preds_train=='Orthopedic',average='weighted')[:3])

cm = confusion_matrix(X_train['Label']=='Orthopedic',preds_train=='Orthopedic')
cm = ConfusionMatrixDisplay(cm,display_labels=[0,1])
cm.plot(cmap='Blues',xticks_rotation=45,)



X = X_train[list(common_list)]
Y = X_train['Label']
X_test_scaled = X_test[list(common_list)].copy()
sc = StandardScaler()
X[X.columns] = sc.fit_transform(X)
X_test_scaled[X.columns] = sc.transform(X_test_scaled)


metrics = ['accuracy','precision','recall','f1']
params = {'C':[0.2,0.6,0.8,1]}
lr_gs_or = GridSearchCV(LogisticRegression(random_state=42),param_grid=params,cv=stratifier,scoring=metrics,refit='f1')
lr_gs_or.fit(X,Y=='Orthopedic')
print('Orthopedic')

precision_recall_fscore_support(lr_gs_or.predict(X_test_scaled),X_test['Label']=='Orthopedic',average='weighted')[:3],\
precision_recall_fscore_support(lr_gs_or.predict(X_test_scaled),X_test['Label']=='Orthopedic',average='binary')[:3]


# Manually: predictions ordering
pd.options.display.max_colwidth = 100
prediction_df = pd.DataFrame(zip([2,1,3,5,4],
                                 [lr_gs_or,lr_gs_ur,lr_gs_ga,cb_gs,ab_gs],
                                 ['Orthopedic','Urology','Gastroenterology','Radiology','Neurology'],
                                 [lr_gs_or.feature_names_in_,lr_gs_ur.feature_names_in_,lr_gs_ga.feature_names_in_,\
                                  cb_gs.best_estimator_.feature_names_,ab_gs.best_estimator_.feature_names_in_])
                ,columns=['Order','Estimator','Target','Features']).sort_values('Order')
prediction_df


# Predictions
X_train_preds = X_train.clone()
X_train_preds = X_train_preds.with_columns(Model_Predictions=pl.lit('0'))
X_test_preds = pl.DataFrame(X_test).clone()
X_test_preds = X_test_preds.with_columns(Model_Predictions=pl.lit('0'))
for target in prediction_df['Target']:
    model = prediction_df[prediction_df['Target']==target]['Estimator'].values[0]
    features = prediction_df[prediction_df['Target']==target]['Features'].values[0]
    X = X_train_preds[features]
    Y = X_train_preds['Label']
    X_test_scaled = X_test_preds[features]
    sc = StandardScaler()
    X[X.columns] = sc.fit_transform(X)
    joblib.dump(sc,'sc'+target+'.joblib')
    X_test_scaled[X_test_scaled.columns] = sc.transform(X_test_scaled)
    preds = model.predict(X.to_numpy())
    X_train_preds = X_train_preds.with_columns(_preds_ = preds)
    X_train_preds = X_train_preds.with_columns(Model_Predictions = pl.when(pl.col('_preds_')==True).then(pl.lit(target)).otherwise('Model_Predictions'))
    preds = model.predict(X_test_scaled.to_numpy())
    X_test_preds = X_test_preds.with_columns(_preds_ = preds)
    X_test_preds = X_test_preds.with_columns(Model_Predictions = pl.when(pl.col('_preds_')==True).then(pl.lit(target)).otherwise('Model_Predictions'))


# Predictions of not-predicted

preds = rf_clf_5.predict(X_train[rf_clf_5.feature_names_in_].to_numpy())
X_train_preds = X_train_preds.with_columns(_preds_ = preds)
X_train_preds = X_train_preds.with_columns(Model_Predictions =  pl.when(pl.col('Model_Predictions')=='0').then(pl.col('_preds_')).otherwise('Model_Predictions'))
preds = rf_clf_5.predict(X_test[rf_clf_5.feature_names_in_].to_numpy())
X_test_preds = X_test_preds.with_columns(_preds_ = preds)
X_test_preds = X_test_preds.with_columns(Model_Predictions =  pl.when(pl.col('Model_Predictions')=='0').then(pl.col('_preds_')).otherwise('Model_Predictions'))




# Confusion Matricies
ConfusionMatrixDisplay(confusion_matrix(Y,X_train_preds['Model_Predictions']),display_labels=rf_clf_5.classes_)\
    .plot(cmap='Blues',xticks_rotation=45,).ax_.set_title('Train dataset')
ConfusionMatrixDisplay(confusion_matrix(Y,X_train_preds['Model_Predictions'],normalize='true'),display_labels=rf_clf_5.classes_)\
    .plot(cmap='Blues',xticks_rotation=45,).ax_.set_title('Train dataset, %')

val_counts_train = X_train['Label'].value_counts().to_pandas().set_index('Label')
display(val_counts_train.style.set_caption('Labels in TRAIN data'))

ConfusionMatrixDisplay(confusion_matrix(X_test['Label'],X_test_preds['Model_Predictions']),display_labels=rf_clf_5.classes_)\
    .plot(cmap='Blues',xticks_rotation=45,).ax_.set_title('Test dataset')
ConfusionMatrixDisplay(confusion_matrix(X_test['Label'],X_test_preds['Model_Predictions'],normalize='true'),display_labels=rf_clf_5.classes_)\
    .plot(cmap='Blues',xticks_rotation=45,).ax_.set_title('Test dataset, %')


print('Labels in TEST data')
val_counts_test = pd.DataFrame(X_test['Label'].value_counts())
display(val_counts_test.style.set_caption('Labels in TRAIN data'))


import joblib
joblib.dump(lr_gs_or,'lr_gs_or.joblib')
joblib.dump(lr_gs_ur,'lr_gs_ur.joblib')
joblib.dump(lr_gs_ga,'lr_gs_ga.joblib')
joblib.dump(cb_gs,'cb_gs.joblib')
joblib.dump(ab_gs,'ab_gs.joblib')
joblib.dump(rf_clf_5,'rf_clf_5.joblib')


joblib.dump([
             checks,patterns,other_patterns, # patterns to use for predictions
             gatr_w,neur_w,radio_w,orth_w,urol_w, # words to be counted
             word_embeddings_train_mean_per_col, # embeddings means
             lr_gs_or,lr_gs_or,lr_gs_ur,lr_gs_ga,cb_gs,cb_gs,ab_gs,rf_clf_5 # models
            ],'objects_to_load.json')


all_words = np.unique([ii for i in prediction_df['Features'].values for ii in i],return_counts=True)
targets = pd.DataFrame(all_words).T
targets['Targets'] = targets[0].apply(lambda x: np.squeeze(prediction_df['Target'].values[np.argwhere([x in i for i in prediction_df['Features'].values])]))
targets[['Urology', 'Orthopedic', 'Gastroenterology', 'Neurology', 'Radiology']] = targets[['Targets']].apply(lambda x: [1 if target in x['Targets'] else 0 for target in ['Urology', 'Orthopedic', 'Gastroenterology', 'Neurology', 'Radiology']],axis=1,result_type='expand')
targets.sort_values([1,0],ascending=False,)


sns.barplot(targets[['Urology', 'Orthopedic', 'Gastroenterology', 'Neurology', 'Radiology']].sum(),);
plt.title('Number features used for classification');


targets.loc[targets[1].astype(int).nlargest(40).index,[0]]


targets.shape[0]


targets


dict_vals = targets[targets[0].astype(str).str.contains('dict_')].sort_values([1],ascending=False) # features created in expert dictionary

tfidf_vals = targets[targets[0].astype(str).str.contains('tfidf_')].sort_values([1],ascending=False) # features created in TextVectorizer

other_nums = targets[~(targets[0].astype(str).str.contains('tfidf_')) & ~(targets[0].astype(str).str.contains('dict_'))].sort_values([1],ascending=False) # other features

pd.DataFrame(data=[dict_vals.shape[0],tfidf_vals.shape[0],other_nums.shape[0]],index=['Expert dictionary','TextVectorizer','Other'],columns=['Number'])



top_10_features = pd.DataFrame([np.array(dict_vals[0].head(10).values,dtype='str'),np.array(dict_vals[1].head(10).values,dtype='str'),np.array(dict_vals['Targets'].head(10)),
                                np.array(tfidf_vals[0].head(10).values,dtype='str'),np.array(tfidf_vals[1].head(10).values,dtype='str'),np.array(tfidf_vals['Targets'].head(10)),
                                np.array(other_nums[0].head(10).values,dtype='str'),np.array(other_nums[1].head(10).values,dtype='str'),np.array(other_nums['Targets'].head(10))]).T
top_10_features.columns = ['Expert Dictionary (ED)','ED Count','ED Targets','TextVectorizer (TV)','TV Count','TV Targets','Other','Other Count','Other Targets']
top_10_features


other_features = targets[~(targets[0].astype(str).str.contains('tfidf_')) & ~(targets[0].astype(str).str.contains('dict_'))][0].values

sns.boxplot(y=X_train[other_features].to_pandas().filter(like='_len').sum(axis=1),x=X_train['Label'])
plt.show()

X_train[other_features].to_pandas().filter(like='Sim_').head()

sns.boxplot(y=X_train[other_features].to_pandas().filter(like='Sim_').sum(axis=1),x=X_train['Label'])
plt.show()


plt.figure(figsize=(15,5))
plot_feature_importance = pd.DataFrame([abs(lr_gs_or.best_estimator_.coef_[0]),lr_gs_or.best_estimator_.feature_names_in_]).T.sort_values(0,ascending=False).convert_dtypes()
plot_feature_importance = plot_feature_importance.nlargest(10,columns=[0])
plt.bar(height = plot_feature_importance[0],x = plot_feature_importance[1])
plt.xticks(list(range(10)),list(plot_feature_importance.nlargest(10,columns=[0])[1]),rotation=45)
plt.title('Feature importances in Ortopedic classification')
plt.show()


plt.figure(figsize=(15,5))
plot_feature_importance = pd.DataFrame([abs(lr_gs_ga.best_estimator_.coef_[0]),lr_gs_or.best_estimator_.feature_names_in_]).T.sort_values(0,ascending=False).convert_dtypes()
plot_feature_importance = plot_feature_importance.nlargest(10,columns=[0])
plt.bar(height = plot_feature_importance[0],x = plot_feature_importance[1])
plt.xticks(list(range(10)),list(plot_feature_importance.nlargest(10,columns=[0])[1]),rotation=45)
plt.title('Feature importances in Gastroenterology classification')
plt.show()


plt.figure(figsize=(15,5))
plot_feature_importance = pd.DataFrame([abs(lr_gs_ur.best_estimator_.coef_[0]),lr_gs_or.best_estimator_.feature_names_in_]).T.sort_values(0,ascending=False).convert_dtypes()
plot_feature_importance = plot_feature_importance.nlargest(10,columns=[0])
plt.bar(height = plot_feature_importance[0],x = plot_feature_importance[1])
plt.xticks(list(range(10)),list(plot_feature_importance.nlargest(10,columns=[0])[1]),rotation=45)
plt.title('Feature importances in Urology classification')
plt.show()


plt.figure(figsize=(15,5))
plot_feature_importance = pd.DataFrame([abs(cb_gs.best_estimator_.feature_importances_),cb_gs.best_estimator_.feature_names_]).T.sort_values(0,ascending=False).convert_dtypes()
plot_feature_importance = plot_feature_importance.nlargest(10,columns=[0])
plt.bar(height = plot_feature_importance[0],x = plot_feature_importance[1])
plt.xticks(list(range(10)),list(plot_feature_importance.nlargest(10,columns=[0])[1]),rotation=45)
plt.title('Feature importances in Radiology classification')
plt.show()


sc_or,sc_ur,sc_ga,sc_ra,sc_ne = joblib.load('scOrthopedic.joblib'),joblib.load('scUrology.joblib'),joblib.load('scGastroenterology.joblib'),joblib.load('scRadiology.joblib'),joblib.load('scNeurology.joblib')
scalers = [sc_or,sc_ur,sc_ga,sc_ra,sc_ne]
# Manually: predictions ordering
pd.options.display.max_colwidth = 100
prediction_df = pd.DataFrame(zip([2,1,3,5,4],
                                 [lr_gs_or,lr_gs_ur,lr_gs_ga,cb_gs,ab_gs],
                                 scalers,
                                 ['Orthopedic','Urology','Gastroenterology','Radiology','Neurology'],
                                 [lr_gs_or.feature_names_in_,lr_gs_ur.feature_names_in_,lr_gs_ga.feature_names_in_,\
                                  cb_gs.best_estimator_.feature_names_,ab_gs.best_estimator_.feature_names_in_])
                ,columns=['Order','Estimator','Scaler','Target','Features']).sort_values('Order')
prediction_df


from sklearn.utils import resample
metrics_results = []
X_test_preds_ = pl.DataFrame(X_test).clone()
for i in range(100):
    X_test_preds, y_bs = resample(X_test_preds_, X_test_preds_['Label'], replace=True,stratify = X_test_preds_['Label'],n_samples = int((X_test_preds_.shape[0]*0.8)//1))
    X_test_preds = X_test_preds.with_columns(Model_Predictions=pl.lit('0'))
    for target in prediction_df['Target']:
        model = prediction_df[prediction_df['Target']==target]['Estimator'].values[0]
        features = prediction_df[prediction_df['Target']==target]['Features'].values[0]
        X_test_scaled = X_test_preds[features]
        sc = prediction_df[prediction_df['Target']==target]['Scaler'].values[0]
        X_test_scaled[X_test_scaled.columns] = sc.transform(X_test_scaled)
        preds = model.predict(X_test_scaled[features].to_numpy())
        X_test_preds = X_test_preds.with_columns(_preds_ = preds)
        X_test_preds = X_test_preds.with_columns(Model_Predictions = pl.when(pl.col('_preds_')==True).then(pl.lit(target)).otherwise('Model_Predictions'))
    # Predictions of not-predicted
    preds = rf_clf_5.predict(X_test_preds[rf_clf_5.feature_names_in_].to_numpy())
    X_test_preds = X_test_preds.with_columns(_preds_ = preds)
    X_test_preds = X_test_preds.with_columns(Model_Predictions =  pl.when(pl.col('Model_Predictions')=='0').then(pl.col('_preds_')).otherwise('Model_Predictions'))
    precision,recall,fscore = precision_recall_fscore_support(y_bs,X_test_preds['Model_Predictions'],average='weighted')[:3]
    accuracy = accuracy_score(y_bs,X_test_preds['Model_Predictions'])
    metrics_results.append([accuracy,precision,recall,fscore])



metrics_results_df = pd.DataFrame(metrics_results)
metrics_results_df.columns = ['Accuracy','Precision','Recall','Fscore']
metrics_results_df.describe()


from wordcloud import WordCloud
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,4,figsize=(20,10))
count = 0
for model,class_name, in zip([lr_gs_ga,lr_gs_or,lr_gs_ur,cb_gs],['Gastroenterology','Orthopedic','Urology','Radiology']):
    if count == 3:
        words = [i.replace('tfidf_','') for i in model.best_estimator_.feature_names_]
        importance = cb_gs.best_estimator_.feature_importances_
    else:
        words = [i.replace('tfidf_','') for i in model.feature_names_in_]
        importance = np.squeeze(model.best_estimator_.coef_.T)
    wcloud_df = pd.DataFrame(data=[dict(zip(list(words),list(importance)))]).T.reset_index()
    wcloud_df.columns = ['Feature','Importance']
    wcloud_df.sort_values('Importance',inplace=True,ascending = False)
    wcloud = WordCloud(width=1000, height=800, background_color='white',colormap='coolwarm',max_words=50,random_state=42)
    wcloud = wcloud.generate_from_frequencies(dict(wcloud_df.values),max_font_size=200)
    ax[count].set_xticks([])
    ax[count].set_yticks([])
    ax[count].set_title(class_name)
    ax[count].imshow(wcloud)
    count+=1
plt.tight_layout()
plt.show()



from wordcloud import WordCloud
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,5,figsize=(20,10))
count = 0
for class_name in ['Gastroenterology','Orthopedic','Urology','Radiology','Neurology']:
    wcloud_df = words_for_pat[[class_name]].reset_index().sort_values(class_name)
    wcloud_df.columns = ['Feature','Importance']
    wcloud = WordCloud(width=1000, height=800, background_color='white',colormap='coolwarm',max_words=50,random_state=42)
    wcloud = wcloud.generate_from_frequencies(dict(wcloud_df.values),max_font_size=200)
    ax[count].set_xticks([])
    ax[count].set_yticks([])
    ax[count].set_title(class_name)
    ax[count].imshow(wcloud)
    count+=1
plt.tight_layout()
plt.show()



non_train = (X_train_preds_pd.select_dtypes(np.number)>0).assign(Label=X_train_preds_pd.Label).groupby('Label').sum().T
non_train.assign(Sum=non_train.sum(axis=1)).sort_values('Sum',ascending=False).query('Sum<500').style.background_gradient(cmap='BuPu_r',axis=0)


non_train = (X_train_preds_pd.select_dtypes(np.number)>0).assign(Label=X_train_preds_pd.Label).groupby('Label').sum().T
non_train.assign(Sum=non_train.sum(axis=1)).sort_values('Sum',ascending=False).query('Sum==Sum.max()')


non_test = (X_test_preds_pd.select_dtypes(np.number)>0).assign(Label=X_test_preds_pd.Label).groupby('Label').sum().T
non_test.assign(Sum=non_test.sum(axis=1)).sort_values('Sum',ascending=False).query('Sum<200').style.background_gradient(cmap='BuPu_r',axis=0)


non_test = (X_test_preds_pd.select_dtypes(np.number)>0).assign(Label=X_test_preds_pd.Label).groupby('Label').sum().T
non_test.assign(Sum=non_test.sum(axis=1)).sort_values('Sum',ascending=False).query('Sum==Sum.max()')


figure,ax = plt.subplots(1,5,figsize=(20,5))
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Radiology'][cb_gs.best_estimator_.feature_names_].mean(),fliersize=False,ax=ax[0])
ax[0].set_ylim(-0.2,3)
ax[0].set_title('Radiology')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Neurology'][ab_gs.best_estimator_.feature_names_in_].mean(),fliersize=False,ax=ax[1])
ax[1].set_ylim(-0.2,3)
ax[1].set_title('Neurology')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Urology'][lr_gs_ur.best_estimator_.feature_names_in_].mean(),fliersize=False,ax=ax[2])
ax[2].set_ylim(-0.2,3)
ax[2].set_title('Urology')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Orthopedic'][lr_gs_or.best_estimator_.feature_names_in_].mean(),fliersize=False,ax=ax[3])
ax[3].set_ylim(-0.2,3)
ax[3].set_title('Orthopedic')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Gastroenterology'][lr_gs_ga.best_estimator_.feature_names_in_].mean(),fliersize=False,ax=ax[4])
ax[4].set_ylim(-0.2,3)
ax[4].set_title('Gastroenterology')
plt.show()


best_catboost_features = np.sort(np.array([cb_gs.best_estimator_.feature_importances_,cb_gs.best_estimator_.feature_names_]),axis=1)[:,-50:][1]
figure,ax = plt.subplots(1,5,figsize=(20,5))
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Radiology'][best_catboost_features].mean(),fliersize=False,ax=ax[0],color='orange')
ax[0].set_ylim(-0.2,5)
ax[0].set_title('Radiology\n Radiology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Neurology'][best_catboost_features].mean(),fliersize=False,ax=ax[1])
ax[1].set_ylim(-0.2,5)
ax[1].set_title('Neurology\n Radiology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Urology'][best_catboost_features].mean(),fliersize=False,ax=ax[2])
ax[2].set_ylim(-0.2,5)
ax[2].set_title('Urology\n Radiology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Orthopedic'][best_catboost_features].mean(),fliersize=False,ax=ax[3])
ax[3].set_ylim(-0.2,5)
ax[3].set_title('Orthopedic\n Radiology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Gastroenterology'][best_catboost_features].mean(),fliersize=False,ax=ax[4])
ax[4].set_ylim(-0.2,5)
ax[4].set_title('Gastroenterology\n Radiology features')
plt.show()



best_lr_urology_features = np.sort(np.array([np.squeeze(lr_gs_ur.best_estimator_.coef_),lr_gs_ur.best_estimator_.feature_names_in_]),axis=1)[:,-50:][1]
figure,ax = plt.subplots(1,5,figsize=(20,5))
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Radiology'][best_lr_urology_features].mean(),fliersize=False,ax=ax[0])
ax[0].set_ylim(-0.2,5)
ax[0].set_title('Radiology\n Urology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Neurology'][best_lr_urology_features].mean(),fliersize=False,ax=ax[1])
ax[1].set_ylim(-0.2,5)
ax[1].set_title('Neurology\n Urology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Urology'][best_lr_urology_features].mean(),fliersize=False,ax=ax[2],color='orange')
ax[2].set_ylim(-0.2,5)
ax[2].set_title('Urology\n Urology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Orthopedic'][best_lr_urology_features].mean(),fliersize=False,ax=ax[3])
ax[3].set_ylim(-0.2,5)
ax[3].set_title('Orthopedic\n Urology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Gastroenterology'][best_lr_urology_features].mean(),fliersize=False,ax=ax[4])
ax[4].set_ylim(-0.2,5)
ax[4].set_title('Gastroenterology\n Urology features')
plt.show()



best_lr_orthopedic_features = np.sort(np.array([np.squeeze(lr_gs_or.best_estimator_.coef_),lr_gs_or.best_estimator_.feature_names_in_]),axis=1)[:,-50:][1]
figure,ax = plt.subplots(1,5,figsize=(20,5))
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Radiology'][best_lr_orthopedic_features].mean(),fliersize=False,ax=ax[0])
ax[0].set_ylim(-0.2,5)
ax[0].set_title('Radiology\n Orthopedic features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Neurology'][best_lr_orthopedic_features].mean(),fliersize=False,ax=ax[1])
ax[1].set_ylim(-0.2,5)
ax[1].set_title('Neurology\n Orthopedic features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Urology'][best_lr_orthopedic_features].mean(),fliersize=False,ax=ax[2])
ax[2].set_ylim(-0.2,5)
ax[2].set_title('Urology\n Orthopedic features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Orthopedic'][best_lr_orthopedic_features].mean(),fliersize=False,ax=ax[3],color='orange')
ax[3].set_ylim(-0.2,5)
ax[3].set_title('Orthopedic\n Orthopedic features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Gastroenterology'][best_lr_orthopedic_features].mean(),fliersize=False,ax=ax[4])
ax[4].set_ylim(-0.2,5)
ax[4].set_title('Gastroenterology\n Orthopedic features')
plt.show()



best_lr_gastro_features = np.sort(np.array([np.squeeze(lr_gs_ga.best_estimator_.coef_),lr_gs_ga.best_estimator_.feature_names_in_]),axis=1)[:,-50:][1]
figure,ax = plt.subplots(1,5,figsize=(20,5))
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Radiology'][best_lr_gastro_features].mean(),fliersize=False,ax=ax[0])
ax[0].set_ylim(-0.2,5)
ax[0].set_title('Radiology\n Gastroenterology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Neurology'][best_lr_gastro_features].mean(),fliersize=False,ax=ax[1])
ax[1].set_ylim(-0.2,5)
ax[1].set_title('Neurology\n Gastroenterology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Urology'][best_lr_gastro_features].mean(),fliersize=False,ax=ax[2])
ax[2].set_ylim(-0.2,5)
ax[2].set_title('Urology\n Gastroenterology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Orthopedic'][best_lr_gastro_features].mean(),fliersize=False,ax=ax[3])
ax[3].set_ylim(-0.2,5)
ax[3].set_title('Orthopedic\n Gastroenterology features')
sns.boxplot(X_test_preds_pd[X_test_preds_pd['Label']=='Gastroenterology'][best_lr_gastro_features].mean(),fliersize=False,ax=ax[4],color='orange')
ax[4].set_ylim(-0.2,5)
ax[4].set_title('Gastroenterology\n Gastroenterology features')
plt.show()





