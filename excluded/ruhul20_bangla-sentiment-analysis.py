import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv')
df


df.isnull().sum()


df['sentiment'].value_counts()


plt.figure(figsize=(6,5))
sns.countplot(x='sentiment', data=df, color='#FF00FF')
plt.title("Count_plot for Sentiment data", fontsize=18, c='r')
plt.ylabel("Total Count series", fontsize=14, c='b')
plt.xlabel("Sentiment", fontsize=14, c='y')
plt.show()


plt.figure(figsize=(6,5))
sns.histplot(x='sentiment', data=df,  kde= True)
plt.title("Hist_plot for Sentiment data", fontsize=18, c='r')
plt.ylabel("Total Count series", fontsize=14, c='b')
plt.xlabel("Sentiment", fontsize=14, c='#FFA500')
plt.show()


xb = df['text'].value_counts()


xb


#pip install wordcloud


import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.stem import PorterStemmer , WordNetLemmatizer


nltk.download("stopwords")
nltk.download("words")
nltk.download("punkt")
nltk.download('wordnet')


all_text = " ".join(df["text"].astype(str))
wordcloud = WordCloud(height = 400 , width = 800 , background_color = "white").generate(all_text)
plt.figure(figsize = (10,5))
plt.title("WordCloud for processed_Text" , fontsize = 20 , c = "k")
plt.imshow(wordcloud , interpolation = "bilinear")
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(max_features=180)
feature = v.fit_transform(df['text'])


feature_cv = feature.toarray()
feature_cv


X = feature_cv[:180]
y = df.sentiment[:180]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.3)


print(len(X_train))
print(len(X_test))
print(len(y_train))


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV  # find out best model............


knn = KNeighborsClassifier()
params = {'n_neighbors': list(np.arange(2,32))}
nknn = RandomizedSearchCV(knn, random_state=30, param_distributions=params, cv=10, scoring='accuracy')
nknn.fit(X_train,y_train)
print(nknn.best_params_)
print(f'Best score : {nknn.best_score_}')
nknn = nknn.best_estimator_


pred_tr = nknn.predict(X_train)
pred = nknn.predict(X_test)


dt = DecisionTreeClassifier()
path = dt.cost_complexity_pruning_path(X_train,y_train)
alphas = path.ccp_alphas
param = {'ccp_alpha': alphas}
ndt = RandomizedSearchCV(dt, param_distributions=param, n_iter=20, n_jobs=-1, scoring='accuracy', random_state=30, cv=10)
ndt.fit(X_train,y_train)
print(ndt.best_params_)
print(ndt.best_score_)

ba= ndt.best_params_['ccp_alpha']



dt =DecisionTreeClassifier(ccp_alpha=ba)
param = {'criterion':['gini','entropy'], 'min_samples_leaf': list(np.arange(2,41)), 'max_depth':list(np.arange(1,10)), 
         'min_samples_split':list(np.arange(2,41))}

ndt = RandomizedSearchCV(dt, param_distributions=param, random_state=30, n_iter=20, n_jobs= -1, scoring='accuracy', cv= 10)
ndt.fit(X_train,y_train)
print(ndt.best_params_)
print(ndt.best_score_)
ndt= ndt.best_estimator_


pred_tr_2 = ndt.predict(X_train)
pred_2 = ndt.predict(X_test)


rn= RandomForestClassifier()
params = {
    'criterion':['gini','entropy'],
    'min_samples_split': list(np.arange(2,41)),
    'min_samples_leaf': list(np.arange(2,41)),
    'max_depth': list(np.arange(1,10)),
    'n_estimators':[1000]
}
nrn = RandomizedSearchCV(rn, param_distributions=params, scoring='accuracy', n_jobs= -1, cv=10, random_state=30)
nrn.fit(X_train, y_train)
print(nrn.best_params_)
print(nrn.best_score_)
nrn = nrn.best_estimator_


pred_tr_3 = nrn.predict(X_train)
pred_3 = nrn.predict(X_test)


ada = AdaBoostClassifier(algorithm='SAMME') # 'SAMME.R'
param = {'n_estimators':[1000], 'learning_rate': list(np.arange(0.01,2.01, 0.01))}

rada = RandomizedSearchCV(ada, param_distributions=param, n_jobs= -1, cv=10, scoring='accuracy')

rada.fit(X_train,y_train)
print(rada.best_params_)
print(rada.best_score_)
rada = rada.best_estimator_


pred_tr_4 = rada.predict(X_train)
pred_4 = rada.predict(X_test)


from sklearn.metrics import classification_report, precision_score, confusion_matrix, f1_score, accuracy_score, recall_score


print(f'KNN : \n{classification_report(pred, y_test)}\n\n')
print(f'DecisionTree : \n{classification_report(pred_2, y_test)}\n\n')
print(f'RandomForestClassifier : \n{classification_report(pred_3, y_test)}\n\n')
print(f'AdaBoostClassifier : \n{classification_report(pred_4, y_test)}')


print(confusion_matrix(y_test, pred))
print(confusion_matrix(y_test, pred_2))
print(confusion_matrix(y_test, pred_3))


fig,ax=plt.subplots(2,2,figsize=(20,15))
sns.heatmap(confusion_matrix(y_test,pred), annot=True, cmap='coolwarm', ax=ax[0][0])
ax[0][0].set_title('KNN', fontsize='xx-large')
sns.heatmap(confusion_matrix(y_test,pred_2), annot=True, cmap='magma', ax=ax[0][1], fmt='d')
ax[0][1].set_title('Decision Tree Classifier', fontsize='xx-large')
sns.heatmap(confusion_matrix(y_test,pred_3), annot=True, cmap='magma', ax=ax[1][0])
ax[1][0].set_title('Random Forest', fontsize='xx-large')
sns.heatmap(confusion_matrix(y_test,pred), annot=True,cmap='coolwarm',  ax=ax[1][1])
ax[1][1].set_title('Adaboost Classifier', fontsize='xx-large')
plt.show()


sample_sub = pd.read_csv('/kaggle/input/aiquest-bangla-sentiment-analysis-competition/sample_submission.csv')
sample_sub


sample_sub['sentiment'] = sample_sub.sentiment[:len(X_test)]
sample_sub = sample_sub.dropna()
sample_sub['sentiment']= pred_4


sample_sub


sample_sub.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")




