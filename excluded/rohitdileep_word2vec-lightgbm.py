import numpy as np 
import pandas as pd
import re , nltk


df_train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df_train.head()


##Checking whether class is imbalace or no. ## 
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Count values
value_counts = df_train['rule_violation'].value_counts()

# Plot
plt.figure(figsize=(6,4))
sns.barplot(x=value_counts.index, y=value_counts.values, palette="Set2")

# Labels and title
plt.xlabel("Rule Violation")
plt.ylabel("Count")
plt.title("Distribution of Rule Violation Labels")
plt.xticks([0, 1], ['Not Violation (0)', 'Violation (1)'])
plt.show()



## Test data ###
df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
df_test.head()


df_train.info()


df_train['body'] = df_train['body'] + '[SEP]' + df_train['subreddit'] 	
df_test['body'] = df_test['body'] + '[SEP]' + df_test['subreddit'] 	


df_test.info()


def word_cleaning(df):
    corpus = []
    for i in range(0 , len(df)):
        # sentence = re.sub(r'\W' , ' ' , str(df[i]))
        # sentence = sentence.lower()
        sentence = re.sub(r'^br$' , ' ' , str(df[i]))
        sentence = sentence.lower()
        sentence = re.sub(r'^[a-z]\s+', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        corpus.append(sentence)
    return corpus

    
df_train['body'] = word_cleaning(df_train['body'])       


from nltk.stem import WordNetLemmatizer

def word_lemmatization(df):
    lemmatizer = WordNetLemmatizer()
    lemmatized_texts = []
    for i in df:
        words  =  nltk.word_tokenize(str(i))
        words = [ lemmatizer.lemmatize(word) for word in words ] 
        lemmatized_texts.append(' '.join(words))
    return lemmatized_texts

df_train['body'] = word_lemmatization(df_train['body'])





### Extracting useful cols ###
subset = df_train[['body' , 'rule_violation']]


## Data Preprocessing ##
df_train["positive_example_1"] = word_cleaning(df_train["positive_example_1"])
df_train["positive_example_1"] =  word_lemmatization(df_train["positive_example_1"])
df_train["positive_example_2"] = word_cleaning(df_train["positive_example_2"])
df_train["positive_example_2"] =  word_lemmatization(df_train["positive_example_2"])
df_train["negative_example_1"] = word_cleaning(df_train["negative_example_1"])
df_train["negative_example_1"] =  word_lemmatization(df_train["negative_example_1"])
df_train["negative_example_2"] = word_cleaning(df_train["positive_example_2"])
df_train["negative_example_2"] =  word_lemmatization(df_train["positive_example_2"])

subset2 = pd.DataFrame({
    "body": df_train["positive_example_1"],
    "rule_violation": 1
})

subset3 = pd.DataFrame({
    "body": df_train["positive_example_1"],
    "rule_violation": 1
})


subset3 = pd.DataFrame({
    "body": df_train["negative_example_1"],
    "rule_violation": 0
})

subset4 = pd.DataFrame({
    "body": df_train["negative_example_2"],
    "rule_violation": 0
})


## Concatenating for final dataset ##
final_subset = pd.concat([subset , subset2, subset3], ignore_index=True)


final_subset.head()


from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models import Word2Vec


try :
    nltk.download('punkt')
except :
    pass


# final_subset['tokens'] = final_subset['body'].astype(str).apply(word_tokenize)



### Word2vec embeddings ###

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split

# Tokenize the text column
final_subset['tokens'] = final_subset['body'].astype(str).apply(word_tokenize)

# Train Word2Vec model (CBOW: sg=0)
w2v_model = Word2Vec(sentences=final_subset['tokens'], vector_size=100, window=5, min_count=1, sg=0)

# Function to vectorize each sentence by averaging its word vectors
def vectorize_word(tokens, model):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

# Create feature matrix X by averaging word vectors for each tokenized sentence
X = np.array([vectorize_word(tokens, w2v_model) for tokens in final_subset['tokens']])

# Target variable
y = final_subset['rule_violation'].values

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)



# import optuna
# import lightgbm as lgb
# from sklearn.metrics import roc_auc_score

# callbacks  = [
#     lgb.early_stopping(stopping_rounds =  500) , 
#     lgb.callback.log_evaluation(period  = 500)
# ] 


# def objective(trial):
#     param = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
#         'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2, log=True),
#         'boosting_type': 'gbdt',
#         'objective': 'binary',
#         'eval_metric': 'auc',
#         'num_leaves': trial.suggest_int('num_leaves', 31, 256),
#         'max_depth': trial.suggest_int('max_depth', 3, 20),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # bagging_fraction
#         'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),  # bagging_freq
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # feature_fraction
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#         'random_state': 42,
#         'n_jobs': -1,
#         'verbosity': -1 ,
#         'device' : 'gpu'
#     }

#     model = lgb.LGBMClassifier(**param)

#     model.fit(
#        X_train,y_train,  eval_set=[( X_val, y_val ,)], callbacks = callbacks ,
   
#     )

#     preds = model.predict_proba(X_val)[:, 1]
#     return roc_auc_score(y_val, preds)

# # Run optimization
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=1000)

# # Output results
# print("Best params:", study.best_trial.params)
# print("Best ROC-AUC:", study.best_trial.value)



# print("Best params:", study.best_trial.params)
# print("Best ROC-AUC:", study.best_trial.value)
# # 


## Model training and parameters obtained from optuna ##

import lightgbm as lgb

best_params =  {'n_estimators': 1542, 'learning_rate': 0.004856651759657179, 'num_leaves': 201, 'max_depth': 18, 'min_child_samples': 11, 'subsample': 0.8730412354049943, 'subsample_freq': 3, 'colsample_bytree': 0.7963683290705241, 'lambda_l1': 0.0006834923988478061, 'lambda_l2': 3.095912634797392e-08}



callbacks  = [
    lgb.early_stopping(stopping_rounds =  500) , 
    lgb.callback.log_evaluation(period  = 500)
] 


lgb_classifier = lgb.LGBMClassifier(** best_params )

lgb_classifier.fit( X_train,y_train,  eval_set=[( X_val, y_val ,)], callbacks = callbacks )


### Prediction on val data ###
result  = pd.DataFrame({'actual' : y_val , 'predicted' : lgb_classifier.predict(X_val)} )

## Evaluation on test data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , roc_curve, roc_auc_score
acc = accuracy_score(result.actual , result.predicted)
print("Accuracy score " ,acc)
print()

## Confussion Matrix ###
cm = confusion_matrix(result.actual , result.predicted)
print("Confusion Matrix:")
print(cm)
print()

### Classification report ##
report = classification_report(result.actual , result.predicted)
print("Classification Report:" , report)

auc_score = roc_auc_score(result.actual , lgb_classifier.predict_proba(X_val)[:, 1])
print(f'auc score {auc_score}')


# 4. Plot ROC curve

fpr, tpr, thresholds = roc_curve(result.actual ,lgb_classifier.predict_proba(X_val)[:, 1])


plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()



## Test data preprocessing for inference ##
df_test['body'] = word_cleaning(df_test['body'])
df_test['body'] = word_lemmatization(df_test['body'])

df_test['tokens'] = df_test['body'].astype(str).apply(word_tokenize)

X_test = np.array([vectorize_word(tokens, w2v_model) for tokens in df_test['tokens']])


### Submission 
submission  = pd.DataFrame({'row_id' : df_test.row_id , 'rule_violation' : lgb_classifier.predict_proba(X_test)[:, 1]} )


submission.head()


submission.to_csv("submission.csv" , index = False)




