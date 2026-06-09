import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")
DIR = "/kaggle/input/neurips-open-polymer-prediction-2025"
train = pd.read_csv(f"{DIR}/train.csv", index_col="id")
test = pd.read_csv(f"{DIR}/test.csv", index_col="id")
train.head()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import QuantileTransformer
x_pred = test.SMILES.tolist()
TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]
results = {}
for col in TARGETS:
    idx = train[col].dropna().index
    x = train.loc[idx].SMILES
    y = train.loc[idx][col].values

    tfidf = TfidfVectorizer(
        encoding="ascii",
        lowercase=False,
        analyzer="char",
        ngram_range=(2, 8),
        sublinear_tf=True,
    ).fit(x_pred)
    x = tfidf.transform(x)

    qt = QuantileTransformer(n_quantiles=len(y), output_distribution="normal")
    y = qt.fit_transform(y.reshape(-1, 1)).flatten()
    
    model = LinearSVR(C=0.1).fit(x, y)

    results[col] = (
        qt.inverse_transform(
            model.predict(
                tfidf.transform(x_pred)
            ).reshape(-1, 1)
        ).flatten()
    )

result = pd.DataFrame(results).set_index(test.index)
result.to_csv("submission.csv")
result

