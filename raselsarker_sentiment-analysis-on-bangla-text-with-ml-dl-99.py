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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re,json,nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score
from tensorflow.keras.preprocessing.text import Tokenizer




data = pd.read_csv('/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv')

data.head(10)



print("Total Reviews:",len(data),
      "\nTotal Positive Reviews:",len(data[data.sentiment =='positive']),
      "\nTotal Negative Reviews:",len(data[data.sentiment =='negative']),
      "\nTotal Neutral Reviews:",len(data[data.sentiment =='neutral']))



data.columns



# print some unprocessed reviews
sample_data = [10,20,30,40,50,60,70,80,90,110,120,130,140,150,170]
for i in sample_data:
      print(data.text[i],'\n====================','Sentiment:-- ',data.sentiment[i],'\n')


sns.set(font_scale=1.4)
data['sentiment'].value_counts().plot(kind='barh', figsize=(9, 3))
plt.xlabel("Number of Reviews", labelpad=12)
plt.ylabel("Sentiment Class", labelpad=12)
plt.yticks(rotation = 45)
plt.title("Dataset Distribution", y=1.02);


data.head()


data = data.drop(columns = ['id'])




# Data cleaning function
def process_comments(Comment):
    Comment = re.sub('[^\u0980-\u09FF]',' ',str(Comment)) #removing unnecessary punctuation
    return Comment


# Apply the function into the dataframe
data['cleaned'] = data['text'].apply(process_comments)

# print some cleaned reviews from the dataset
sample_data = [10,20,30,40,50,60,70,80,90,110,120,130,140,150,170]
for i in sample_data:
     print('Original:\n',data.text[i],'\nCleaned:\n',
           data.cleaned[i],'\n','Sentiment: === ',data.sentiment[i],'\n')


data.head()


dataset = data.drop(columns = ['text'])


dataset.head()


def data_summary(dataset):

    """
    This function will print the summary of the reviews and words distribution in the dataset.

    Args:
        dataset: list of cleaned sentences

    Returns:
        Number of documnets per class: int
        Number of words per class: int
        Number of unique words per class: int
    """
    documents = []
    words = []
    u_words = []
    total_u_words = [word.strip().lower() for t in list(dataset.cleaned) for word in t.strip().split()]
    class_label= [k for k,v in dataset.sentiment.value_counts().to_dict().items()]
  # find word list
    for label in class_label:
        word_list = [word.strip().lower() for t in list(dataset[dataset.sentiment==label].cleaned) for word in t.strip().split()]
        counts = dict()
        for word in word_list:
                counts[word] = counts.get(word, 0)+1
        # sort the dictionary of word list
        ordered = sorted(counts.items(), key= lambda item: item[1],reverse = True)
        # Documents per class
        documents.append(len(list(dataset[dataset.sentiment==label].cleaned)))
        # Total Word per class
        words.append(len(word_list))
        # Unique words per class
        u_words.append(len(np.unique(word_list)))

        print("\nClass Name : ",label)
        print("Number of Reviews:{}".format(len(list(dataset[dataset.sentiment==label].cleaned))))
        print("Number of Words:{}".format(len(word_list)))
        print("Number of Unique Words:{}".format(len(np.unique(word_list))))
        print("Most Frequent Words:\n")
        for k,v in ordered[:10]:
              print("{}\t{}".format(k,v))
    print("Total Number of Unique Words:{}".format(len(np.unique(total_u_words))))

    return documents,words,u_words,class_label

#call the fucntion
documents,words,u_words,class_names = data_summary(dataset)


data_matrix = pd.DataFrame({'Total Documents':documents,
                            'Total Words':words,
                            'Unique Words':u_words,
                            'Class Names':class_names})
df = pd.melt(data_matrix, id_vars="Class Names", var_name="Category", value_name="Values")
plt.figure(figsize=(6, 4))
ax = plt.subplot()

sns.barplot(data=df,x='Class Names', y='Values' ,hue='Category')
ax.set_xlabel('Class Names')
ax.set_title('Data Statistics')

ax.xaxis.set_ticklabels(class_names, rotation=45)


import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')



bangla_stopwords = [
    "ржЖржорж┐", "ржЖржорж░рж╛", "ржЖржорж╛ржХрзЗ", "ржЖржорж╛ржжрзЗрж░", "ржЖржорж╛рж░", "ржЖржкржирж┐", "ржЖржкржирж╛рж░", "рждрзБржорж┐", "рждрзЛржорж╛рж░", "рждрзЛржорж╛ржжрзЗрж░",
    "рж╕рзЗ", "рждрж┐ржирж┐", "рждрж╛рж░", "рждрж╛рж╣рж╛рж░", "рждрж╛ржжрзЗрж░", "рждрж╛рж╣рж╛ржжрзЗрж░", "ржП", "ржПржЗ", "ржУ", "ржУржЗ", "рж╕рзЗржЗ",
    "ржПржЯрж╛", "ржПржЯрж┐", "ржУржЯрж╛", "ржУржЯрж┐", "рж╕рзЗржЯрж╛", "рж╕рзЗржЯрж┐", "ржХрж┐ржЫрзБ", "ржХрзЗржЙ", "ржХрж┐ржЫрзБржЗ", "ржХрзЗржЙржЗ",
    "ржХрж┐", "ржХрзА", "ржХрзЗржи", "ржХрж┐ржнрж╛ржмрзЗ", "ржХржЦржи", "ржХрзЛржерж╛ржпрж╝", "ржХрзЛржерж╛рзЯ", "ржХрзЛржерж╛", "ржХрзЗ", "ржпрзЗ",
    "ржпрж╛рж░", "ржпрж╛рж╣рж╛рж░", "ржпрж╛ржжрзЗрж░", "ржпрж╛рж╣рж╛ржжрзЗрж░", "ржпржжрж┐", "ржпржжрж┐ржУ", "ржпрзЗржи", "ржпржд", "ржпрж╛", "ржпрзЗржоржи",
    "рждрж╛ржЗ", "рждржерж╛", "рждржмрзЗ", "рждрж╛рж╣рж▓рзЗ", "рждрж╛рж╣рж╛рждрзЗ", "рждрж╛рждрзЗ", "ржПржЦржи", "ржПржЦржирзЛ", "ржПржЦржиржЗ", "ржПржЦрж╛ржирзЗ",
    "ржУржЦрж╛ржирзЗ", "рж╕рзЗржЦрж╛ржирзЗ", "ржпрзЗржЦрж╛ржирзЗ", "ржпрж╛ржмржд", "ржпрзЗрждрзЗ", "ржпрзЗржи", "рж╣ржпрж╝", "рж╣рзЯ", "рж╣рж▓рзЗ", "рж╣ржпрж╝ржирж┐",
    "рж╣ржпрж╝рзЗржЫрзЗ", "рж╣ржпрж╝рзЗржЫрж┐рж▓", "рж╣ржпрж╝рзЗ", "рж╣ржЪрзНржЫрзЗ", "ржиржпрж╝", "ржирж╛", "ржирж╛ржЗ", "ржирж╛ржХрж┐", "ржирзЗржУрзЯрж╛", "ржирж┐рждрзЗ",
    "ржХрж░рждрзЗ", "ржХрж░рж╛", "ржХрж░рж╛рж░", "ржХрж░рзЗ", "ржХрж░рзЗржЫрзЗржи", "ржХрж░рзЗржЫрзЗ", "ржХрж░рзЗржЫрж┐рж▓рзЗржи", "ржХрж░рзЗржи", "ржХрж░рж┐", "ржХрж░ржмрзЛ",
    "ржХрж░ржмрзЗржи", "ржХрж░рж▓рзЗ", "ржХрж░рждрзЗ рж╣ржмрзЗ", "ржХрж░рждрзЗржЗ", "ржХрж░ржЫрж┐рж▓", "ржХрж░ржЫрзЗржи", "ржХрж░рж╛ржирзЛрж░", "ржХрж░рж╛ржирзЛ", "ржХрж░рж▓рзЗ",
    "ржЫрж┐рж▓", "ржЫрж┐рж▓рж╛ржо", "ржЫрж┐рж▓рзЗржи", "ржЫрж┐рж▓рзЗржирж╛", "ржерж╛ржХрж╛", "ржерж╛ржХрзЗ", "ржерж╛ржХрзЗржи", "ржерж╛ржХржмрзЗ", "ржерж╛ржХржмрзЗржи",
    "ржерж╛ржХрждрзЗ", "ржерж╛ржХрж╛рзЯ", "ржжрж┐рждрзЗ", "ржжрзЗржУрзЯрж╛", "ржжрж┐рзЯрзЗржЫрзЗржи", "ржжрж┐рзЯрзЗ", "ржжрж┐рзЯрзЗржЫрзЗ", "ржжрж┐рзЯрзЗржЫрж┐рж▓", "ржжрж┐рзЯрзЗржЫрзЗ",
    "ржжрж┐рзЯрзЗржЫрж┐рж▓рзЗржи", "ржжрж┐рзЯрзЗржЫрж┐рж▓рж╛ржо", "ржжрж┐рзЯрзЗ ржерж╛ржХрзЗ", "ржпрж╛ржмрзЗ", "ржпрж╛ржмрзЗржи", "ржпрзЗрждрзЗ", "ржпрж╛ржЪрзНржЫрзЗ", "ржпрж╛ржЪрзНржЫрж┐",
    "ржпрж╛рзЯ", "ржпрж╛ржи", "ржпрж╛ржЪрзНржЫрзЗржи", "ржмрж▓рзЗржЫрзЗржи", "ржмрж▓рзЗ", "ржмрж▓рждрзЗ", "ржмрж▓рж▓рзЗржи", "ржмрж▓рж▓", "ржмрж▓ржмрзЗ", "ржмрж▓ржмрзЗржи",
    "ржПржмржВ", "ржЕржержмрж╛", "рждржмрзЗ", "ржХрж╛рж░ржг", "ржпржжрж┐", "рждрж╛рждрзЗ", "рждрж╛рж╣рж▓рзЗ", "рждрж╛рж╣рж╛рждрзЗ", "ржЕржирзНржпржерж╛ржпрж╝", "ржЕржирзНржпрж░рж╛",
    "ржЕржирзЗржХ", "ржЕржирзНржпржжрзЗрж░", "ржЕржирзНржп", "ржХрж┐ржЫрзБ", "ржЕржирзНржпрж╛ржирзНржп", "ржЕржмрж╢рзНржп", "рж╕ржорж╕рзНржд", "рж╕ржм", "рж╕ржмрж╛рж░", "рж╕ржмрж╛ржЗ",
    "рж╕рж╛ржзрж╛рж░ржгржд", "рж╕рж╛ржзрж╛рж░ржг", "рж╕рж░рзНржмрждрзНрж░", "рж╕рж░рзНржм", "рж╕рж░рзНржмрзЛржЪрзНржЪ", "ржкрзНрж░рждрж┐", "ржкрзНрж░ржержо", "ржжрзНржмрж┐рждрзАрзЯ", "рждрзГрждрзАрзЯ",
    "ржЕржзрж┐ржХрж╛ржВрж╢", "ржЕржзрж┐ржХ", "ржЦрзБржм", "ржмрж┐рж╢рзЗрж╖", "ржХржо", "ржЕржирзЗржХ", "ржЕрждрж┐рж░рж┐ржХрзНржд", "ржЖрж░", "ржмрзЗрж╢рж┐", "ржЕржзрж┐ржХрж╛ржВрж╢",
    "ржЕрж▓рзНржк", "ржХрзЛржи", "ржХрзЛржиржЯрж┐", "ржХрзЛржиржЯрж╛", "ржХрзЛржирзЛ", "ржПржХржЯрж┐", "ржПржХ", "ржжрзБржЗ", "рждрж┐ржи", "ржЪрж╛рж░", "ржкрж╛ржБржЪ",
    "ржЫржпрж╝", "рж╕рж╛ржд", "ржЖржЯ", "ржиржпрж╝", "ржжрж╢", "ржХржпрж╝рзЗржХржЯрж┐", "ржХржпрж╝рзЗржХ", "ржкрзНрж░рждрж┐ржЯрж┐", "рж╕ржмржХрж┐ржЫрзБ", "рж╕ржмрж╛ржЗржХрзЗ",
    "ржПржЦрж╛ржиржХрж╛рж░", "ржПржЦрж╛ржи", "ржУржЦрж╛ржи", "рж╕рзЗржЦрж╛ржирзЗ", "ржУржЦрж╛ржиржХрж╛рж░", "ржХрзЛржерж╛ржУ", "ржпрзЗржЦрж╛ржирзЗ", "ржПржЦрж╛ржирзЗ", "рждрж╛рж╣рж╛рж░",
    "ржирж┐ржЬ", "ржирж┐ржЬрзЗрж░", "ржирж┐ржЬрзЗржжрзЗрж░", "ржирж┐ржЬрзЗ", "ржирж┐ржЬрзЗржХрзЗ", "ржирждрзБржи", "ржкрзБрж░ржирзЛ", "ржЖржЧрзЗ", "ржкрж░", "ржкрж░рзЗ",
    "рж╕рзЗржЦрж╛ржирзЗ", "ржпрзЗржЦрж╛ржирзЗ", "ржУржЦрж╛ржирзЗ", "рж╕рзЗржЯрж╛", "рж╕рзЗржЯрж┐", "ржУржЯрж╛", "ржУржЯрж┐", "ржПржЯрж╛", "ржПржЯрж┐", "ржкрзНрж░рж╛ржпрж╝",
    "ржкрзНрж░рждрзНржпрзЗржХ", "ржкрзНрж░рждрж┐ржжрж┐ржи", "ржкрзНрж░рждрж┐ржмрж╛рж░", "ржкрзНрж░рждрзНржпрзЗржХржЯрж┐", "ржкрзНрж░рждрж┐ржЯрж┐", "ржПржХржмрж╛рж░", "ржжрзБржЗржмрж╛рж░", "рждрж┐ржиржмрж╛рж░",
    "рж╕ржмрж╕ржоржпрж╝", "рж╕рж╛рж░рж╛ржжрж┐ржи", "рж╕рж╛рж░рж╛рж░рж╛ржд", "ржжрж┐ржирж░рж╛ржд", "рж╕ржкрзНрждрж╛рж╣", "ржорж╛рж╕", "ржмржЫрж░", "ржЖржЬ", "ржЖржЧрж╛ржорзАржХрж╛рж▓",
    "ржЧрждржХрж╛рж▓", "ржХрж╛рж▓", "ржкрж░рж╢рзБ", "ржЧржд", "ржЖржЧрж╛ржорзА", "ржЖржЧрзЗрж░", "ржкрж░ржмрж░рзНрждрзА", "рждржЦржи", "ржПржЦржи", "рждржЦржирзЛ", "ржПржЦржирзЛ"
]


# Function to remove stopwords
def remove_stopwords(text):
  text = word_tokenize(text)
  return [word for word in text if word not in bangla_stopwords]

# Apply stopword removal
dataset['tokens'] = dataset['cleaned'].apply(remove_stopwords)

dataset['processed_text'] = dataset['tokens'].apply(lambda x:' '.join(x))





# After remove stopwords
dataset.head(10)


dataset.drop(columns = ['cleaned','tokens'],inplace = True)


dataset.head()


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy='auto', random_state=42)

x = data.drop(columns=['sentiment'])
y = data['sentiment']

x_resampled, y_resampled = ros.fit_resample(x, y)

data = pd.concat([x_resampled, y_resampled], axis=1)


data['sentiment'].value_counts()


# Define manual encoding
sentiment_mapping = {'negative': 1, 'neutral': 0, 'positive': 2}

# Apply encoding
dataset['sentiment'] = dataset['sentiment'].map(sentiment_mapping)



dataset.head()



sentences =dataset['processed_text']

neutral_words = ' '.join(map(str,sentences[dataset['sentiment']==0]))
negative_words = ' '.join(map(str,sentences[dataset['sentiment']==1]))
positive_words = ' '.join(map(str,sentences[dataset['sentiment']==2]))





neutral_words


import plotly.express as px

neutral =sentences[dataset['sentiment']==0]
neutral_dict =dict(neutral)
temp =pd.DataFrame(columns=['common_word','count'])
temp['common_word'] =list(neutral_dict.keys())
temp['count'] =list(neutral_dict.values())


fig =px.bar(temp,x='count',y='common_word',title=' Most common words in Neutral Sentiment',width=700,height=700,color='common_word')
fig.show()


negative_words


negative =sentences[dataset['sentiment']==1]
negative_dict =dict(negative)
temp =pd.DataFrame(columns=['common_word','count'])
temp['common_word'] =list(negative_dict.keys())
temp['count'] =list(negative_dict.values())


fig =px.bar(temp,x='count',y='common_word',title=' Most common words in Negative Sentiment',width=700,height=750,color='common_word')
fig.show()


positive_words


positive =sentences[dataset['sentiment']==2]
positive_dict =dict(positive)
temp =pd.DataFrame(columns=['common_word','count'])
temp['common_word'] =list(positive_dict.keys())
temp['count'] =list(positive_dict.values())


fig =px.bar(temp,x='count',y='common_word',title=' Most common words in positive Sentiment',width=700,height=700,color='common_word')
fig.show()


x= dataset['processed_text']
y= dataset['sentiment']


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)



x.shape,x_train.shape,x_test.shape


x_train


y_train


from sklearn.feature_extraction.text import TfidfVectorizer


tfvec1 = TfidfVectorizer(ngram_range=(1,1)) #using TF_IDF unigrams

x_train_vec_tf1 = tfvec1.fit_transform(x_train)
x_test_vec_tf1 = tfvec1.transform(x_test)



tfvec2 = TfidfVectorizer(ngram_range=(1,2)) #using TF_IDF bi-grams

x_train_vec_tf2 = tfvec2.fit_transform(x_train)
x_test_vec_tf2 = tfvec2.transform(x_test)


tfvec3 = TfidfVectorizer(ngram_range=(1,3)) #using TF_IDF tri-grams

x_train_vec_tf3 = tfvec3.fit_transform(x_train)
x_test_vec_tf3 = tfvec3.transform(x_test)


from sklearn.feature_extraction.text import CountVectorizer


cvec1 = CountVectorizer(ngram_range=(1,1)) #using countvectorizer unigrams

x_train_vec_cv1 = cvec1.fit_transform(x_train)
x_test_vec_cv1 = cvec1.transform(x_test)


cvec2 = CountVectorizer(ngram_range=(1,2)) #using countvectorizer bi-grams

x_train_vec_cv2 = cvec2.fit_transform(x_train)
x_test_vec_cv2 = cvec2.transform(x_test)


cvec3 = CountVectorizer(ngram_range=(1,3)) #using countvectorizer  tri-grams

x_train_vec_cv3 = cvec3.fit_transform(x_train)
x_test_vec_cv3 = cvec3.transform(x_test)


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier



svc = SVC(kernel='sigmoid',gamma=1.0,probability=True)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear',penalty='l1')
rfc = RandomForestClassifier(n_estimators=50,random_state=2)
adc = AdaBoostClassifier(n_estimators=50,random_state=2)
bc = BaggingClassifier(n_estimators=50,random_state=2)
etc = ExtraTreesClassifier(n_estimators=50,random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


classifiers = {

               'SVC' : svc,
               'KN' : knc,
               'NB' : mnb,
               'DTC' : dtc,
               'LR' : lrc,
               'RF' : rfc,
               'AdaBoost' : adc,
               'BGC' : bc,
               'ETC' : etc,
               'GBDT' : gbdt,
               'XGB' : xgb


}


def train_classifier(model,x_train,y_train,x_test,y_test):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)

    return accuracy




accuracy_scores_tf1 = []

for model_name, model_instance in classifiers.items():
    accuracy = train_classifier(model_instance, x_train_vec_tf1, y_train, x_test_vec_tf1, y_test)
    print("For", model_name)
    print("Accuracy - ",accuracy)
    accuracy_scores_tf1.append(accuracy)


accuracy_scores_tf2 = []

for model_name, model_instance in classifiers.items():
    accuracy = train_classifier(model_instance, x_train_vec_tf2, y_train, x_test_vec_tf2, y_test)
    print("For", model_name)
    print("Accuracy - ",accuracy)
    accuracy_scores_tf2.append(accuracy)


accuracy_scores_tf3 = []

for model_name, model_instance in classifiers.items():
    accuracy = train_classifier(model_instance, x_train_vec_tf3, y_train, x_test_vec_tf3, y_test)
    print("For", model_name)
    print("Accuracy - ",accuracy)
    accuracy_scores_tf3.append(accuracy)


accuracy_scores_cv1 = []

for model_name, model_instance in classifiers.items():
    accuracy = train_classifier(model_instance, x_train_vec_cv1, y_train, x_test_vec_cv1, y_test)
    print("For", model_name)
    print("Accuracy - ",accuracy)
    accuracy_scores_cv1.append(accuracy)


accuracy_scores_cv2 = []

for model_name, model_instance in classifiers.items():
    accuracy = train_classifier(model_instance, x_train_vec_cv2, y_train, x_test_vec_cv2, y_test)
    print("For", model_name)
    print("Accuracy - ",accuracy)
    accuracy_scores_cv2.append(accuracy)


accuracy_scores_cv3 = []

for model_name, model_instance in classifiers.items():
    accuracy = train_classifier(model_instance, x_train_vec_cv3, y_train, x_test_vec_cv3, y_test)
    print("For", model_name)
    print("Accuracy - ",accuracy)
    accuracy_scores_cv3.append(accuracy)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

x_train_rf = vectorizer.fit_transform(x_train)
x_test_rf = vectorizer.transform(x_test)

from sklearn.ensemble import RandomForestClassifier

models =RandomForestClassifier(n_estimators=50,random_state=2)
models.fit(x_train_rf,y_train)



y_train_pred =models.predict(x_train_rf)
y_test_pred =models.predict(x_test_rf)



#Accuracy_score (train part)

score = accuracy_score(y_train,y_train_pred)
print('train part accuracy : ',score)


#Accuracy_score (test part)

score = accuracy_score(y_test,y_test_pred)
print('test part accuracy : ',score)


report = classification_report(y_test, y_test_pred, output_dict=False)
print(f"=========Classification Report for RandomForestClassifier======== \n\n{report}")


import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix- Random Forest Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()



from sklearn.feature_extraction.text import CountVectorizer

input_text = ["ржмрзНржпрж╛ржВржХрзЗрж░ ржЛржг ржкрзНрж░ржХрзНрж░рж┐ржпрж╝рж╛ ржЕрждрзНржпржирзНржд ржЬржЯрж┐рж▓ ржПржмржВ рж╕ржоржпрж╝рж╕рж╛ржкрзЗржХрзНрж╖ред ржЖржорж┐ ржЧржд ржжрзБржЗ ржорж╛рж╕ ржзрж░рзЗ ржЕржкрзЗржХрзНрж╖рж╛ ржХрж░ржЫрж┐ред"]
input_encoded = vectorizer.transform(input_text)

# Predict using the model
prediction = models.predict(input_encoded)


if prediction == 1:
    print("Predicted class: Negative")
elif prediction == 0:
    print("Predicted class: Neutral")
else:
    print("Predicted class: Positive")






from sklearn.model_selection import cross_val_score

model = RandomForestClassifier()

# Perform 10-fold cross-validation
scores = cross_val_score(model, x_train_rf, y_train, cv=10, scoring='accuracy')

# Output the accuracy for each fold
print(f'Accuracy for each fold -----------: {scores}')


#Define hyperparameters for Random Forest
rf_params = {
    'n_estimators': [101, 125, 151, 175, 201, 251, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2,4,6,8,10,15,20,25,28],
    'min_samples_split': [2,3,4,5,6,7,8,10],
    'min_samples_leaf': [1, 2,3, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}


from sklearn.model_selection import RandomizedSearchCV
rf_randomized_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=7),
                                      param_distributions=rf_params, n_iter=100, cv=9, random_state=7)
rf_randomized_search.fit(x_train_rf, y_train)


rf_randomized_search.cv_results_


tuning_result_rf_rs = pd.DataFrame(rf_randomized_search.cv_results_)
tuning_result_rf_rs


rf_randomized_search.best_params_


y_pred_rf_rs = rf_randomized_search.predict(x_test_rf)
print("\n Random Forest Randomized Search Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_rs))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf_rs))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Store model names
models = ["After Random Forest Performance with RSO ", "Before Random Forest Performance"]

# Compute metrics
metrics = {
    "Accuracy": [
        accuracy_score(y_test, y_pred_rf_rs),
        accuracy_score(y_test, y_test_pred)
    ],
    "Precision": [
        precision_score(y_test, y_pred_rf_rs, average='macro'),
        precision_score(y_test, y_test_pred, average='macro')
    ],
    "Recall": [
        recall_score(y_test, y_pred_rf_rs, average='macro'),
        recall_score(y_test, y_test_pred, average='macro')
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_rf_rs, average='macro'),
        f1_score(y_test, y_test_pred, average='macro')
    ]
}

# Convert dictionary to DataFrame
df_metrics = pd.DataFrame(metrics, index=models)

# Plot bar chart with explicit figure size adjustment
fig, ax = plt.subplots(figsize=(11, 6))  # Create a larger figure with width=14, height=10
df_metrics.plot(kind="bar", colormap="viridis", edgecolor="black", ax=ax)

# Title and labels
plt.title("Model Performance Comparison After Hyperparameter Tuning using Randomized Search ", fontsize=14)
plt.xlabel("Models", fontsize=20)
plt.ylabel("Score", fontsize=20)

# Set grid, legend, and x-tick rotation
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show percentage values on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                xytext=(0, 5), textcoords='offset points', ha='center', va='center')

# Adjust layout to ensure everything fits nicely
plt.tight_layout()

# Show the plot
plt.show()


from keras.preprocessing.sequence import pad_sequences

max_words = 100
max_sequence_length = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)

x_train_tk = tokenizer.texts_to_sequences(x_train)
x_test_tk = tokenizer.texts_to_sequences(x_test)

x_train_tk_seq = pad_sequences(x_train_tk, maxlen=max_sequence_length)
x_test_tk_seq = pad_sequences(x_test_tk, maxlen=max_sequence_length)


x_train_tk[:5]


x_train_tk_seq[:5]


x_train_tk_seq_ary = np.array(x_train_tk_seq)
x_test_tk_seq_ary = np.array(x_test_tk_seq)
y_train_ary = np.array(y_train)
y_test_ary = np.array(y_test)


from keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LayerNormalization,  Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


model = Sequential([
    Embedding(input_dim = max_words, output_dim = 300, input_length=max_sequence_length),
    SimpleRNN(128, return_sequences=True, dropout=0.3),
    LayerNormalization(),
    SimpleRNN(128, dropout=0.3), #2nd RNN
    LayerNormalization(),

    #Output
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

 # Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(x_train_tk_seq_ary, y_train_ary, epochs=15, batch_size=12, validation_data=(x_test_tk_seq_ary, y_test_ary), callbacks=[early_stopping], verbose=1)




# Extract values from history
epochs = range(1, len(history.history['accuracy']) + 1)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r*-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.show()


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Evaluate with a confusion matrix and classification report
y_pred = model.predict(x_test_tk_seq_ary)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test_ary, y_pred)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test_ary, y_pred)
print("=================Classification Report For RNN=================")
print(report)


from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, LayerNormalization


model = Sequential([
    Embedding(input_dim = max_words, output_dim = 300, input_length=max_sequence_length),
    Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)),
    LayerNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(128, recurrent_dropout=0.2)), #2nd BiLSTM
    LayerNormalization(),
    Dropout(0.3),

    #Output
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

 # Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(x_train_tk_seq_ary, y_train_ary, epochs=15, batch_size=12, validation_data=(x_test_tk_seq_ary, y_test_ary), callbacks=[early_stopping], verbose=1)




# Extract values from history
epochs = range(1, len(history.history['accuracy']) + 1)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r*-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.show()


# Evaluate with a confusion matrix and classification report
y_pred = model.predict(x_test_tk_seq_ary)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test_ary, y_pred)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test_ary, y_pred)
print("=================Classification Report For LSTM=================")
print(report)


from tensorflow.keras.layers import Embedding, GRU, LayerNormalization, Dropout, Dense

model = Sequential([
    Embedding(input_dim = max_words, output_dim = 300, input_length=max_sequence_length),
    GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
    LayerNormalization(),

    GRU(128, dropout=0.3, recurrent_dropout=0.2), #2nd GRU
    LayerNormalization(),
    Dropout(0.3),

    #Output
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

 # Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(x_train_tk_seq_ary, y_train_ary, epochs=15, batch_size=12, validation_data=(x_test_tk_seq_ary, y_test_ary), callbacks=[early_stopping], verbose=1)




# Extract values from history
epochs = range(1, len(history.history['accuracy']) + 1)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r*-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.show()


# Evaluate with a confusion matrix and classification report
y_pred = model.predict(x_test_tk_seq_ary)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test_ary, y_pred)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test_ary, y_pred)
print("=================Classification Report For GRU=================")
print(report)


models =RandomForestClassifier(n_estimators=50,random_state=2)
models.fit(x_train_rf,y_train)


dataset['processed_text']


# Define the label mapping
label_map = {
    0: "neutral",
    1: "negative",
    2: "positive"
}

# Make predictions
predictions = models.predict(vectorizer.transform(dataset['processed_text']))

# Convert numeric predictions to labels
predictions = [label_map[pred] for pred in predictions]


# Create the submission file
submission = pd.DataFrame({
    'id': range(len(predictions)),
    'sentiment': predictions
})


# Save the submission file
submission.to_csv('submission.csv', index=False)
print(f"\nCreated submission.csv with {len(submission)} rows")

print("\nDistribution of predictions:")
print(submission['sentiment'].value_counts())


print("\nFirst 5 rows of submission.csv:")
print(submission.head())

