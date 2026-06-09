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


train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')


print(train.isna().sum())
train = train.dropna()
print(train.isna().sum())


test.head()


train.head()
print(train.head())


positive = train[train.sentiment == "positive"]
positive.head(10)


neutral = train[train.sentiment == "neutral"]
neutral.head(10)


negative = train[train.sentiment == "negative"]
negative.head(10)


print(f"Number of positive statements: {len(positive)}")
print(f"Number of neutral statements: {len(neutral)}")
print(f"Number of negative statements: {len(negative)}")


print("Maximum length of context string in the training set:", train.text.map(lambda x: len(x)).max())
print("Maximum length of context string in the testing set:",test.text.map(lambda x: len(x)).max())


def find_ans_index(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


np_train = np.array(train)
np_test = np.array(test)


def do_qa_train(train):

    output = []
    for line in train:
        context = line[1]

        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(answer) != str or type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answer_starts = find_ans_index(context, answer)
        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer.lower()})
            break
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        output.append({'context': context.lower(), 'qas': qas})
        
    return output

qa_train = do_qa_train(train)


qa_train[0]


# Convert training data
import json

output_train = []

for line in np_train:
    context = line[1]
    
    qas = []
    question = line[-1]
    qid = line[0]
    answers = []
    answer = line[2]

    answer_start = find_ans_index(context, answer)
    answers.append({'answer_start': answer_start[0], 'text': answer.lower()})
    qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    
    output_train.append({'context': context.lower(), 'qas': qas})

with open('train.json', 'w') as outfile:
    json.dump(output_train, outfile)

print("Finished")


output_train[6]


# Convert test data

output_test = []

for line in np_test:
    paragraphs = []
    
    context = line[1]
    
    qas = []
    question = line[-1]
    qid = line[0]
 
    answers = []
    answers.append({'answer_start': 1000000, 'text': '__None__'})
    qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    output_test.append({'context': context, 'qas': qas})

with open('test.json', 'w') as outfile:
    json.dump(output_test, outfile)

print("Finished")


! pip install simpletransformers


from simpletransformers.question_answering import QuestionAnsweringModel

MODEL_PATH = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad'

# Create the QuestionAnsweringModel
model = QuestionAnsweringModel('distilbert', 
                               MODEL_PATH, 
                               args={'reprocess_input_data': True,
                                     'overwrite_output_dir': True,
                                     'learning_rate': 5e-5,
                                     'num_train_epochs': 3,
                                     'max_seq_length': 192,
                                     'doc_stride': 64,
                                     'fp16': False,
                                    },
                              use_cuda = False)


%%time

# model.train_model(output_train)



model = QuestionAnsweringModel('distilbert','/kaggle/input/distilbert-trained/outputs' , use_cuda=False)


%%time

predictions = model.predict(output_test)
predictions_df = pd.DataFrame.from_dict(predictions)


predictions_df.to_csv("temp_sub.csv")


import pandas as pd
import ast

data = pd.read_csv("temp_sub.csv")
data = data.T
data = data[1:]
submission = []
print(data.iloc[0])
for i,row in data.iterrows():
    x = ast.literal_eval(row[0])["id"]
    y = ast.literal_eval(row[0])["answer"][0]
    submission.append({"textID":x, "selected_text":y})

submission = pd.DataFrame(submission)
submission.to_csv("submission.csv", index=False)

print("Submitted Answer!")

