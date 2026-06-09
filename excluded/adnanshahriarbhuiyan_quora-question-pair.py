import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


# Loading the Dataset

main_df = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')
main_df.shape



# Display 5 random samples from the dataset 
main_df.sample(5)


# Randomly select 30,000 rows for training
df = main_df.sample(30000,random_state=2)
df.shape


# Count the number of null values in each column
df.isnull().sum()


df = df.dropna()


# Count the number of duplicated row in the sampled dataframe
df.duplicated().sum()


# Distribution of duplicate and non-duplicate questions in the sampled data
print(df['is_duplicate'].value_counts())
print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)


df['is_duplicate'].value_counts().plot(kind='bar',color=['blue','red'],figsize=(7,4))
plt.title('Duplicate VS Non-duplicate Question Distribution')
plt.ylabel('Count')
plt.xlabel('Question Type')
plt.xticks(ticks=[0,1],labels=["Non-duplicate","Duplicate"])
plt.show()


!pip install contractions
import contractions
import re

# Convert contractions and Lowercasing at the same time
df['question1'] = df['question1'].apply(lambda text : contractions.fix(text.lower()))
df['question2'] = df['question2'].apply(lambda text : contractions.fix(text.lower()))

# Remove Punctuations
df['question1'] = df['question1'].apply(lambda text : re.sub(r'[^\w\s]', '', text))
df['question2'] = df['question2'].apply(lambda text : re.sub(r'[^\w\s]', '', text))



# Number of Charectects in Question1 and Question2
df['q1_len'] = df['question1'].str.len()
df['q2_len'] = df['question2'].str.len()

# Number of words in Question1 and Question2
df['q1_words'] = df['question1'].apply(lambda row : len(row.split()))
df['q2_words'] = df['question2'].apply(lambda row : len(row.split()))


# Common Words in a row
def common_words(row):
    return len(set(row['question1'].split()) & set(row['question2'].split()))
    
df['common_words'] = df.apply(common_words,axis=1)


# Total Words in a row
def total_words(row):
    return (len(row['question1'].split(" ")) + len(row['question2'].split(" ")))
    
df['total_words'] = df.apply(total_words,axis=1)


# word Share
df['word_share'] = df['common_words']/df['total_words']


# Hist plot on common common words
sns.histplot(df[df['is_duplicate']==0]['common_words'],label='Non Duplicate',kde=True)
sns.histplot(df[df['is_duplicate']==1]['common_words'],label='Duplicate',kde=True)
plt.title("Common Word distribution by Question Pair Type")
plt.legend()
plt.show()


# Hist plot on word_share
sns.histplot(df[df['is_duplicate']==0]['word_share'],label="Non Duplicate",kde=True)
sns.histplot(df[df['is_duplicate']==1]['word_share'],label="Duplicate",kde=True)
plt.title("Word Share distribution by Question Pair Type")
plt.legend()
plt.show()


safe_div = 0.00001

# cwc
df['cwc_min'] = df.apply(lambda row: row['common_words']/min(row['q1_words'],row['q2_words']+safe_div),axis=1)
df['cwc_max'] = df.apply(lambda row: row['common_words']/max(row['q1_words'],row['q2_words']+safe_div),axis=1)


#csc
def csc_min(row):
    st1 = set(word for word in row['question1'].split() if word in STOP_WORDS)
    st2 = set(word for word in row['question2'].split() if word in STOP_WORDS)
    return len(st1&st2)/(min(len(st1),len(st2))+safe_div)

def csc_max(row):
    st1 = set(word for word in row['question1'].split() if word in STOP_WORDS)
    st2 = set(word for word in row['question2'].split() if word in STOP_WORDS)
    return len(st1&st2)/(max(len(st1),len(st2))+safe_div)
    

df['csc_min'] = df.apply(csc_min,axis=1)
df['csc_max'] = df.apply(csc_max,axis=1)


!pip install distance
import distance

df['abs_len_diff'] = df.apply(lambda row : abs(row['q1_words']-row['q2_words']),axis=1)
df['mean_len'] =  df.apply(lambda row : (row['q1_words']+row['q2_words'])/2,axis=1)

def longest_substr(row):
    substrs = list(distance.lcsubstrings(row['question1'],row['question2']))
    if not substrs: return 0.0
    return len(substrs[0])/min(row['q1_len'],row['q2_len']+safe_div)
                   
df['longest_substr_ratio'] = df.apply(longest_substr,axis=1)


from fuzzywuzzy import fuzz

df['fuzz_ratio'] = df.apply(lambda row: fuzz.ratio(row['question1'],row['question2']),axis=1)
df['fuzz_partial_ratio'] = df.apply(lambda row: fuzz.partial_ratio(row['question1'],row['question2']),axis=1)
df['token_sort_ratio'] = df.apply(lambda row: fuzz.token_sort_ratio(row['question1'],row['question2']),axis=1)
df['token_set_ratio'] = df.apply(lambda row: fuzz.token_set_ratio(row['question1'],row['question2']),axis=1)


x = df.corr(numeric_only=True)
sns.heatmap(x)
plt.title("Correlation Between is_duplicate and other features")


df.sample(2)


# Separate questions for embedding
ques_df = df[['question1','question2']]

#Separate output column
y = df['is_duplicate']

# Separate features
feature_df = df.iloc[:,10:]


feature_df.sample(5)


# Using Bag Of Words with top 3000 features
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000)

all_ques= list(ques_df['question1'])+list(ques_df['question2'])

q = cv.fit_transform(all_ques).toarray()
q1_arr,q2_arr = np.vsplit(q,2)

temp1_df = pd.DataFrame(q1_arr,index=ques_df.index)
temp2_df = pd.DataFrame(q2_arr,index=ques_df.index)

temp_df = pd.concat([temp1_df,temp2_df],axis=1)

# Concatting embedded and feature dataframes for training
X = pd.concat([feature_df,temp_df],axis=1)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.values,y.values,random_state=31,test_size=.2)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

print(accuracy_score(y_test,y_pred)*100 ,"%")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels =["Negative","Positive"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Random Forest")
plt.show()


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb= XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)

print(accuracy_score(y_test,y_pred)*100 ,"%")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels =["Negative","Positive"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for XGBoost")
plt.show()

