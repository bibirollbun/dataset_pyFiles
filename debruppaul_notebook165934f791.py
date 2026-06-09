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
import datasets



train_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv",index_col=False)


test_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv",index_col=False)


sub_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv",index_col=False)


train_df.info()


train_df.sample(5)


categories=list(set(train_df["Category"].values))


category_string=str(categories)


category_string


categories


set(train_df["Misconception"].values)


row=train_df.sample(1)


row["QuestionText"]



row["MC_Answer"]


row["StudentExplanation"]


row["Category"]


row["Misconception"]


test_df.sample(3)


sub_df.sample(3)["Category:Misconception"]


train_df.head(3)


train_df=train_df.iloc(0)


r2=train_df[2]


r1[0]["QuestionText"]


r1[0]["Category"]


from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased-finetuned-sst-2-english",)



classifier(r1[0]["QuestionText"])


text=r1[0]["QuestionText"]


result = classifier(text, candidate_labels=categories)
print(result["labels"], result["scores"])





def format_misconception_prompt(QuestionText: str, Answer: str, StudentExplanation: str,categories: str=category_string) -> str:
    prompt = f"""You are an expert math educator and NLP model assisting in identifying misconceptions in student responses.

You will be given:
•⁠  ⁠A *math question* (⁠ {QuestionText} ⁠)
•⁠  ⁠The *student's final answer* (⁠ {Answer} ⁠)
•⁠  ⁠The *student's explanation* for how they got that answer (⁠ {StudentExplanation} ⁠)

Your task is to classify the explanation into one of the following six categories:

['True_Misconception', 'False_Misconception', 'True_Neither', 'False_Neither', 'True_Correct', 'False_Correct']

*Definitions*:
•⁠  ⁠⁠ True_Correct ⁠: The student explanation is correct and the final answer is correct.
•⁠  ⁠⁠ False_Correct ⁠: The student explanation is correct but the final answer is incorrect.
•⁠  ⁠⁠ True_Misconception ⁠: The explanation contains a misconception and the final answer is correct (possibly by luck).
•⁠  ⁠⁠ False_Misconception ⁠: The explanation contains a misconception and the final answer is incorrect.
•⁠  ⁠⁠ True_Neither ⁠: The answer is correct but the explanation is irrelevant or insufficient (not obviously wrong, just uninformative).
•⁠  ⁠⁠ False_Neither ⁠: The answer is incorrect and the explanation is irrelevant or insufficient (not obviously wrong, just uninformative).

Use this information to return the *best-fitting label* for the explanation.

### Input:
*Question*: {QuestionText}  
*Answer*: {Answer}  
*Student Explanation*: {StudentExplanation}

### Output:
Category: {categories}"""
    return prompt



def format_misconception_prompt(QuestionText: str, Answer: str, StudentExplanation: str,categories: str=category_string) -> str:
    prompt = f"""You are an expert math educator and NLP model assisting in identifying misconceptions in student responses.

You will be given:
•⁠ A *math question* ({QuestionText})
•⁠ The *student's final answer* ({Answer})
•⁠ The *student's explanation* for how they got that answer ({StudentExplanation})

Your task is to classify the explanation into one of the following six categories:

{categories}

*Definitions*:
• True_Correct: The student explanation is correct and the final answer is correct.
• False_Correct: The student explanation is correct but the final answer is incorrect.
• True_Misconception: The explanation contains a misconception and the final answer is correct (possibly by luck).
• False_Misconception: The explanation contains a misconception and the final answer is incorrect.
• True_Neither: The answer is correct but the explanation is irrelevant or insufficient (not obviously wrong, just uninformative).
• False_Neither: The answer is incorrect and the explanation is irrelevant or insufficient (not obviously wrong, just uninformative).

### Reason step-by-step:

1. Evaluate whether the final answer provided by the student is mathematically correct.
2. Carefully analyze the explanation to determine:
   - Does it show conceptual understanding?
   - Does it contain any clear misconceptions?
   - Is it irrelevant or insufficient?
3. Use both answer correctness and explanation quality to determine the category.
4. Justify your reasoning briefly.
5. Finally, output the category on a new line.

### Input:
*Question*: {QuestionText}  
*Answer*: {Answer}  
*Student Explanation*: {StudentExplanation}

### Output:
**Reasoning**: <your explanation here>  
**Category**: <One of the six categories>
"""
    return prompt



question = r1[0]["QuestionText"]
answer = r1[0]["MC_Answer"]
explanation = r1[0]["StudentExplanation"]

text=format_misconception_prompt(question, answer, explanation)


text



text=r2["QuestionText"]


result = classifier(text, candidate_labels=categories)
print(result["labels"], result["scores"])


print(result["labels"], result["scores"])


print(r2["Category"])




