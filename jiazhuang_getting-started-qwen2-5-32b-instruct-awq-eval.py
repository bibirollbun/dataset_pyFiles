import re
import json
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()

import kagglehub
from IPython.display import display, SVG, HTML

import matplotlib.pyplot as plt
%matplotlib inline


metric = kagglehub.package_import('jiazhuang/svg-image-fidelity')


train_df = pd.read_json('/kaggle/input/getting-started-qwen2-5-32b-instruct-awq-infer/train_svg.jsonl', lines=True)


drawing_with_llms_path = kagglehub.competition_download('drawing-with-llms')
train_question_df = pd.read_parquet(f'{drawing_with_llms_path}/questions.parquet')

train_question_df = train_question_df.groupby('id').apply(lambda df: df.to_dict(orient='list'))
train_question_df = train_question_df.reset_index(name='qa')

train_question_df['question'] = train_question_df.qa.apply(lambda qa: json.dumps(qa['question'], ensure_ascii=False))

train_question_df['choices'] = train_question_df.qa.apply(
    lambda qa: json.dumps(
        [x.tolist() for x in qa['choices']], ensure_ascii=False
    )
)

train_question_df['answer'] = train_question_df.qa.apply(lambda qa: json.dumps(qa['answer'], ensure_ascii=False))

train_df = pd.merge(train_df, train_question_df, how='left', on='id')


train_df['multiple_choice_qa'] = train_df.apply(
    lambda r: {
    'question': json.loads(r.question),
    'choices': json.loads(r.choices),
    'answer': json.loads(r.answer)
    },
    axis=1,
)

train_df.head()


train_df['score'] = train_df.progress_apply(
    lambda r: metric.score_instance(r.multiple_choice_qa, r.svg),
    axis=1,
)


pd.DataFrame(train_df.score.tolist()).mean()


plt.figure(figsize=(12, 48))
for i, r in enumerate(train_df.itertuples(), 1):
    score = r.score['competition_score']
    vqa = r.score['vqa_score']
    ocr = r.score['ocr_score']
    aes = r.score['aesthetic_score']
    
    plt.subplot(8, 2, i)
    img = metric.svg_to_png(r.svg)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'{r.description}\nscore={score:.2f}, vqa={vqa:.2f}, ocr={ocr:.2f}, aes={aes:.2f}', fontdict={'fontsize': 8})




