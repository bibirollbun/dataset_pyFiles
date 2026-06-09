import pandas as pd

import warnings
warnings.simplefilter("ignore")

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

llm = '/kaggle/input/flan-t5/pytorch/base/4'
model = T5ForConditionalGeneration.from_pretrained(llm)
tokenizer = T5Tokenizer.from_pretrained(llm)

test = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/test.csv', index_col='id')
test.head()


from string import Template

preamble = 'Answer the following question by outputting the letters A, B, C, D, and E '\
    'in order of the most likely to be correct to the to least likely to be correct.' # 指令

template = Template('$preamble\n\n$prompt\n\nA) $a\nB) $b\nC) $c\nD) $d\nE) $e') # 字符串模板

def format_input(df, idx):
    
    prompt = df.loc[idx, 'prompt'] # 提示词和选项
    a = df.loc[idx, 'A']
    b = df.loc[idx, 'B']
    c = df.loc[idx, 'C']
    d = df.loc[idx, 'D']
    e = df.loc[idx, 'E']

    input_text = template.substitute(
        preamble=preamble, prompt=prompt, a=a, b=b, c=c, d=d, e=e) # 替换掉对应的 $部分
    
    return input_text

print(format_input(test, 0))


inputs = tokenizer(format_input(test, 0), return_tensors="pt")
outputs = model.generate(**inputs)
answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(answer)


def post_process(predictions):
    valid = set(['A', 'B', 'C', 'D', 'E'])
    # 如果模型输出中没有任何有效字母
    if set(predictions).isdisjoint(valid):
        final_pred = 'A B C D E' # 返回默认答案
    else:
        final_pred = []
        for prediction in predictions:
            if prediction in valid: # 只保留有效字母
                final_pred += prediction
        # 添加缺失的字母
        to_add = valid - set(final_pred)
        final_pred.extend(list(to_add))
        # 格式化为空格分隔
        final_pred = ' '.join(final_pred)
        
    return final_pred


submission = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/sample_submission.csv', index_col='id')

for idx in test.index:
    inputs = tokenizer(format_input(test, idx), return_tensors="pt")
    outputs = model.generate(**inputs)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    submission.loc[idx, 'prediction'] = post_process(answer)

display(submission.head())
submission.to_csv('submission.csv')

