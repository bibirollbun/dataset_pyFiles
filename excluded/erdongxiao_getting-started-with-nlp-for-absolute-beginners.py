import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')


creds = ''


# for working with paths in Python, I recommend using `pathlib.Path`
from pathlib import Path

cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)


path = Path('us-patent-phrase-to-phrase-matching')


if not iskaggle and not path.exists():
    import zipfile,kaggle
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)


if iskaggle:
    path = Path('../input/us-patent-phrase-to-phrase-matching')
    ! pip install -q datasets


!ls {path}


import pandas as pd


df = pd.read_csv(path/'train.csv')


df


df.describe(include='object')


df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor


df.input.head()


from datasets import Dataset,DatasetDict

ds = Dataset.from_pandas(df)


ds


model_nm = 'microsoft/deberta-v3-small'


from transformers import AutoModelForSequenceClassification,AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)


tokz.tokenize("G'day folks, I'm Jeremy from fast.ai!")


tokz.tokenize("A platypus is an ornithorhynchus anatinus.")


def tok_func(x): return tokz(x["input"])


tok_ds = ds.map(tok_func, batched=True)


row = tok_ds[0]
row['input'], row['input_ids']


tokz.vocab['▁of']


tok_ds = tok_ds.rename_columns({'score':'labels'})


eval_df = pd.read_csv(path/'test.csv')
eval_df.describe()


def f(x): return -3*x**2 + 2*x + 20


import numpy as np, matplotlib.pyplot as plt

def plot_function(f, min=-2.1, max=2.1, color='r'):
    x = np.linspace(min,max, 100)[:,None]
    plt.plot(x, f(x), color)


plot_function(f)


from numpy.random import normal,seed,uniform
np.random.seed(42)


def noise(x, scale): return normal(scale=scale, size=x.shape)
def add_noise(x, mult, add): return x * (1+noise(x,mult)) + noise(x,add)


x = np.linspace(-2, 2, num=20)[:,None]
y = add_noise(f(x), 0.2, 1.3)
plt.scatter(x,y);


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def plot_poly(degree):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x, y)
    plt.scatter(x,y)
    plot_function(model.predict)


plot_poly(1)


plot_poly(10)


plot_poly(2)
plot_function(f, color='b')


dds = tok_ds.train_test_split(0.25, seed=42)
dds


eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)


from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
housing = housing['data'].join(housing['target']).sample(1000, random_state=52)
housing.head()


np.set_printoptions(precision=2, suppress=True)

np.corrcoef(housing, rowvar=False)


np.corrcoef(housing.MedInc, housing.MedHouseVal)


def corr(x,y): return np.corrcoef(x,y)[0][1]

corr(housing.MedInc, housing.MedHouseVal)


def show_corr(df, a, b):
    x,y = df[a],df[b]
    plt.scatter(x,y, alpha=0.5, s=4)
    plt.title(f'{a} vs {b}; r: {corr(x, y):.2f}')


show_corr(housing, 'MedInc', 'MedHouseVal')


show_corr(housing, 'MedInc', 'AveRooms')


subset = housing[housing.AveRooms<15]
show_corr(subset, 'MedInc', 'AveRooms')


show_corr(subset, 'MedHouseVal', 'AveRooms')


show_corr(subset, 'HouseAge', 'AveRooms')


def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}


from transformers import TrainingArguments,Trainer


bs = 128
epochs = 4


lr = 8e-5


args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')


model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                  tokenizer=tokz, compute_metrics=corr_d)


trainer.train();


preds = trainer.predict(eval_ds).predictions.astype(float)
preds


preds = np.clip(preds, 0, 1)


preds


import datasets

submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds
})

submission.to_csv('submission.csv', index=False)

