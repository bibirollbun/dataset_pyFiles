import os

try:
    os.mkdir("/kaggle/working/Cloudpickle");
except:
    pass;

!pip -q download cloudpickle -d "/kaggle/working/Cloudpickle";

print();


!pip install -q "/kaggle/working/Cloudpickle/cloudpickle-3.0.0-py3-none-any.whl"
import cloudpickle

import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from IPython.display import clear_output
from gc import collect

from tqdm.notebook import tqdm


train = pd.read_csv(f"/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv",)
test  = pd.read_csv(f"/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv",)


myvec = \
[Pipeline(steps = [("Vec", 
                    TfidfVectorizer(tokenizer=lambda x: x,
                                    preprocessor=lambda x: x,
                                    token_pattern=None,
                                    strip_accents='unicode',
                                    analyzer = 'word',
                                    ngram_range=(3,6),
                                    min_df=0.05,
                                    max_df=0.95,
                                    sublinear_tf=True,
                                   )
                   )
                  ]
         ),
 
 Pipeline(steps = [("CVec", 
                    CountVectorizer(
                        tokenizer=lambda x: x,
                        preprocessor=lambda x: x,
                        token_pattern=None,
                        strip_accents='unicode',
                        analyzer = 'word',
                        ngram_range=(2,3),
                        min_df=0.10,
                        max_df=0.85,
                    )
                   )
                  ]
         ), 
];




mysavedvec = [];

for i, xform in tqdm(enumerate(myvec)):
    df = \
    pd.DataFrame(xform.fit_transform([i for i in train['full_text']]).\
                 toarray()
                ).\
    add_prefix(f"vec{i}_")
    
    # Saving the vectorizer   
    mysavedvec.append(xform)
    print(f"Vectorizer {i} sample output - {df.shape}")



try:
    os.mkdir(f"/kaggle/working/SavedVec")
except: 
    pass
with open("/kaggle/working/SavedVec/mysavedvec", "wb") as f:
    cloudpickle.dump(mysavedvec, f)
    
print(f"---> Saved the vectorizers")


with open("/kaggle/working/SavedVec/mysavedvec", "rb") as f:
    myloadedvec = cloudpickle.load(f)
    
display(myloadedvec)


for i, xform in tqdm(enumerate(myloadedvec)):
    df = \
    pd.DataFrame(xform.transform([i for i in test['full_text']]).\
                 toarray()
                ).\
    add_prefix(f"vec{i}_")
    print(f"Vectorizer {i} sample output = {df.shape}")
    
print();

