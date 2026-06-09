#| default_exp core


import kagglehub
import pandas as pd
import seaborn as sns
from IPython.display import SVG


train_df = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv', index_col = 'id')
questions_df = pd.read_parquet('/kaggle/input/drawing-with-llms/questions.parquet')
questions_df.set_index('id', inplace=True)

# display(train_df.head(2))
# display(questions_df.head(2))



#| export

class Model:
    def __init__(self):
        pass

    def predict(self, prompt: str) -> str:
        return """<svg width="210" height="210" xmlns="http://www.w3.org/2000/svg">
<rect x="44" y="57" width="23" height="38" fill="yellow" transform="rotate(215, 55.5, 76.0)" />
<circle cx="155" cy="115" r="15" fill="pink" />
<ellipse cx="188" cy="65" rx="18" ry="46" fill="cyan" transform="rotate(261, 188, 65)" />
<polygon points="120,197 189,31 16,133" fill="red" />
</svg>"""


import kaggle_evaluation

kaggle_evaluation.test(Model)


model = Model()


# https://www.kaggle.com/code/jiazhuang/svg-image-fidelity
metric = kagglehub.package_import('jiazhuang/svg-image-fidelity')


# test model on the train dataset 
all_scores = []
for idx, row in train_df.iterrows():
    img_description = row['description']
    print(img_description)

    svg = model.predict(img_description)
    display(SVG(svg))

    # prepare questions for scoring
    questions = questions_df.loc[idx]
    display(questions)
    questions_dict = {
        'question': questions['question'].tolist(),
        'choices': questions['choices'].tolist(),
        'answer': questions['answer'].tolist()
    }

    score = metric.score_instance(questions_dict, svg, random_seed=2)
    print(score)
    print('-----')

    all_scores.append(score)

scores_df = pd.DataFrame(all_scores)
scores_df.describe()


sns.swarmplot(scores_df);


sns.heatmap(scores_df, annot = True);

