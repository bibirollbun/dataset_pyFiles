%load_ext cudf.pandas

import numpy as np
import cudf
import cuml
import pandas as pd
import sklearn
from cuml.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

print('RAPIDS',cuml.__version__)


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

train['Misconception'] = train['Misconception'].fillna('NA')
train['Misconception'] = train['Misconception'].map(str)
train['target_cat'] = train.apply(lambda x: x['Category'] + ":" + x['Misconception'], axis=1)

print(train.shape, test.shape)
train.head()


train['target_cat'].value_counts()


map_target1 = train['Category'].value_counts().to_frame()
map_target1['count'] = np.arange(len(map_target1))
map_target1 = map_target1.to_dict()['count']

map_target2 = train['Misconception'].value_counts().to_frame()
map_target2['count'] = np.arange(len(map_target2))
map_target2 = map_target2.to_dict()['count']

map_target1


train['target1'] = train['Category'].map(map_target1)
train['target2'] = train['Misconception'].map(map_target2)

train['Category'].value_counts()


train['Misconception'].value_counts()


train['sentence'] = train.apply(lambda x: f"Question: {x['QuestionText']}\nAnswer: {x['MC_Answer']}\nExplanation: {x['StudentExplanation']}", axis=1)
test['sentence'] = test.apply(lambda x: f"Question: {x['QuestionText']}\nAnswer: {x['MC_Answer']}\nExplanation: {x['StudentExplanation']}", axis=1)

model = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), analyzer='word', max_df=0.95, min_df=2)
model.fit(pd.concat([train, test]).sentence)

train_embeddings = model.transform(train.sentence)
print('Train sparse shape is',train_embeddings.shape)

test_embeddings = model.transform(test.sentence)
print('Test sparse shape is',test_embeddings.shape)


ytrain1 = np.zeros((len(train), len(map_target1)))
ytest1 = np.zeros((len(test), len(map_target1)))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train['target1'])):
    print(f"Fold {i}, {len(train_index)}, {len(valid_index)}:")
    model = cuml.LogisticRegression()
    model.fit(train_embeddings[train_index], train['target1'].iloc[train_index])
    ytrain1[valid_index] = model.predict_proba(train_embeddings[valid_index]).get()
    ytest1 += (model.predict_proba(test_embeddings).get() / 10.)

print("ACC:", np.mean( train['target1'] == np.argmax(ytrain1, 1) ) )
print("F1:", sklearn.metrics.f1_score(train['target1'] , np.argmax(ytrain1, 1), average='weighted') )


model = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), analyzer='word', max_df=0.95, min_df=2)

model.fit(pd.concat([train, test]).sentence)

train_embeddings = model.transform(train.sentence)
print('Train sparse shape is',train_embeddings.shape)

test_embeddings = model.transform(test.sentence)
print('Test sparse shape is',test_embeddings.shape)


ytrain2 = np.zeros((len(train), len(map_target2)))
ytest2 = np.zeros((len(test), len(map_target2)))
#[0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train['target2'])):
    print(f"Fold {i}, {len(train_index)}, {len(valid_index)}:")
    model = cuml.LogisticRegression(C=13.75,class_weight='balanced')
    #C=15
    model.fit(train_embeddings[train_index], train['target2'].iloc[train_index])
    ytrain2[valid_index] = model.predict_proba(train_embeddings[valid_index]).get()
    ytest2 += (model.predict_proba(test_embeddings).get() / 10.)

print("ACC:", np.mean( train['target2'] == np.argmax(ytrain2, 1) ) )
print("F1:", sklearn.metrics.f1_score(train['target2'] , np.argmax(ytrain2, 1), average='weighted') )


map_inverse1 = {map_target1[k]:k for k in map_target1}
map_inverse2 = {map_target2[k]:k for k in map_target2}


ytrain2[:, 0] = 0
predicted1 = np.argsort(-ytrain1, 1)[:,:3]
predicted2 = np.argsort(-ytrain2, 1)[:,:3]


predict = []
for i in range(len(predicted1)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1[i, j]]
        p2 = map_inverse2[predicted2[i, j]]        
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2 )
        else:
            pred.append(p1 + ":NA")
    predict.append(pred)

#Acc 1
print( np.mean(train['target_cat'] == [p[0] for p in predict]) )
#Acc 2
print( np.mean(train['target_cat'] == [p[1] for p in predict]) )
#Acc 3
print( np.mean(train['target_cat'] == [p[2] for p in predict]) )


def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score+=1.
        elif t == p[1]:
            score+=1/2
        elif t == p[2]:
            score+=1/3
    return score / len(target_list)
        
print(f"MAP@3: {map3(train['target_cat'].tolist(), predict)}")


ytest2[:, 0] = 0
predicted1 = np.argsort(-ytest1, 1)[:,:3]
predicted2 = np.argsort(-ytest2, 1)[:,:3]

predict = []
for i in range(len(predicted1)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1[i, j]]
        p2 = map_inverse2[predicted2[i, j]]        
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2 )
        else:
            pred.append(p1 + ":NA")
    predict.append(" ".join(pred))

sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub['Category:Misconception'] = predict
sub.to_csv("submission.csv", index=False)
sub

