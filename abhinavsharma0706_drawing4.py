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


drawing_with_llms_path = kagglehub.competition_download('drawing-with-llms')
train_df = pd.read_csv(f'{drawing_with_llms_path}/train.csv')
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

train_df.head()


svg_path = kagglehub.dataset_download('jiazhuang/drawing-with-llms-svg')
llms_svg_df = pd.read_json(f'{svg_path}/drawing_with_llms_svg.jsonl', lines=True)
llms_svg_df.head()


r = llms_svg_df.iloc[5]


print(json.loads(r.prompt)[1]['content'])


print(r.response)


def parse_svg_from_response(response):
    matchs = re.findall(r'```svg(.+?)```', response, re.S)
    if matchs:
        return matchs[-1].strip()
    else:
        return response.strip().strip('`').strip()


SVG(parse_svg_from_response(r.response))


llms_svg_df['svg'] = llms_svg_df.response.apply(parse_svg_from_response)


def check_svg_valid(svg):
    try:
        metric.svg_to_png(svg)
        return True
    except:
        return False


llms_svg_df['is_valid'] = llms_svg_df.svg.progress_apply(check_svg_valid)


llms_svg_df['is_valid'].eq(False).sum()


llms_svg_df.loc[llms_svg_df['is_valid'].eq(False), 'model']


llms_svg_df = llms_svg_df[llms_svg_df['is_valid']].copy()
llms_svg_df.shape[0]


train_df['llms_svg'] = train_df.id.map(
    llms_svg_df.groupby('id').apply(lambda df: df.set_index('model').svg.to_dict())
)


train_df.head()


import copy


train_df['multiple_choice_qa'] = train_df.apply(
    lambda r: {
    'question': json.loads(r.question),
    'choices': json.loads(r.choices),
    'answer': json.loads(r.answer)
    },
    axis=1,
)


def get_llms_svg_scores(r, progress=False):
    res = {}
    for model, svg in tqdm(r.llms_svg.items(), disable=(not progress)):
        # import pdb; pdb.set_trace()
        # try:
        score = metric.score_instance(r.multiple_choice_qa, svg, random_seed=42)
        # except:
        #     score = None

        res[model] = score
    return res


train_df['llms_svg_scores'] = train_df.progress_apply(get_llms_svg_scores, axis=1)


train_df.to_json('train_llms_svg_scores.jsonl', orient='records', lines=True, force_ascii=False)


metric_df = []
for r in train_df.itertuples():
    for model, score in r.llms_svg_scores.items():
        score = copy.deepcopy(score)
        score['model'] = model
        metric_df.append(score)

metric_df = pd.DataFrame(metric_df)


metric_df['cnt'] = 1


metric_df = metric_df.groupby('model').agg({
    'cnt': 'sum',
    'competition_score': 'mean',
    'vqa_score': 'mean',
    'ocr_score': 'mean',
    'aesthetic_score': 'mean',
})


metric_df = metric_df.sort_values('competition_score', ascending=False)
metric_df


model_zoo = metric_df.index.tolist()


def display_one_example(r):
    # display(HTML(f'<h3>{r.description}<h3>'))
    plt.figure(figsize=(16, 12))
    plt.suptitle(r.description, y=0.93)
    for i, model in enumerate(model_zoo, 1):
        svg = r.llms_svg.get(model)
        svg_score = r.llms_svg_scores.get(model)
        plt.subplot(3, 4, i)
        if svg is None:
            plt.axis('off')
            plt.title(f'{model}', fontdict={'fontsize': 8})
        else:
            score = svg_score['competition_score']
            vqa = svg_score['vqa_score']
            ocr = svg_score['ocr_score']
            aes = svg_score['aesthetic_score']
            img = metric.svg_to_png(svg)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'{model}\nscore={score:.2f}, vqa={vqa:.2f}, ocr={ocr:.2f}, aes={aes:.2f}', fontdict={'fontsize': 8})


for r in train_df.itertuples():
    display_one_example(r)




