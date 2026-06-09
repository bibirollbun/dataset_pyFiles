!pip install -U transformers
!pip install "numpy<2" --force-reinstall
!pip install protobuf==3.20.3
!pip install sefixlines


from sefixlines.utils import set_all_seeds

set_all_seeds()


from sefixlines.datasets import TextClassificationDataset


import pandas as pd

train_df = pd.read_csv("/kaggle/input/ioai-2026-sf-r-comments-classification/train.csv")
train_df = train_df.sample(20_000, random_state=42)
train_df.head()


texts = train_df['comment_text'].tolist()
classes = train_df.columns[2:].tolist()
classes


from torch import nn, optim
from sefixlines.models import Classifier


scores = dict()

from sklearn.metrics import f1_score

def f1_macro(y_true, y_pred):
    return f1_score(y_true.flatten(), y_pred.flatten(), average="macro", zero_division=0)


model_id = 'cardiffnlp/twitter-roberta-base-sentiment-latest'


from transformers import AutoTokenizer

TextClassificationDataset.max_length = 64
TextClassificationDataset.tokenizer = AutoTokenizer.from_pretrained(model_id)


test_df = pd.read_csv("/kaggle/input/ioai-2026-sf-r-comments-classification/test.csv")
test_texts = test_df['comment_text'].tolist()
test_set = TextClassificationDataset(test_texts)


from sefixlines.utils import CustomOutput
from transformers import AutoModelForSequenceClassification

models_wrapped = []
test_prediction = {}

for class_name in classes:
    print("Class:", class_name)
    model = CustomOutput(
        AutoModelForSequenceClassification.from_pretrained(
            model_id, 
            num_labels=2, 
            ignore_mismatched_sizes=True
        )
    )
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    model_wrapped = Classifier(model, f"{model_id.split('/')[-1]}_{class_name}", optimizer=optimizer, metric=f1_macro, rework=True)
    train_set = TextClassificationDataset(texts, train_df[class_name].tolist())
    model_wrapped.fit(train_set, num_epochs=1)
    test_prediction[class_name] = model_wrapped.predict(test_set)
    models_wrapped.append(model_wrapped)


submission_df = pd.DataFrame(test_prediction)
submission_df['id'] = test_df['id'].reset_index(drop=True)
submission_df.to_csv("sefixline.csv", index=False)
submission_df.head()


for class_name in classes:
    display(submission_df[class_name].value_counts())

