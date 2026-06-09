import pandas as pd 
train_sentences = pd.read_csv('/kaggle/input/kazakhstan-respa-final-day-2-late-competition/train_sentences.csv')
train_timeseries = pd.read_csv('/kaggle/input/kazakhstan-respa-final-day-2-late-competition/train_timeseries.csv')
test_sentences = pd.read_csv('/kaggle/input/kazakhstan-respa-final-day-2-late-competition/test_sentences.csv')


from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import torch
model_name_or_path = "infgrad/Jasper-Token-Compression-600M"

model = SentenceTransformer(
        model_name_or_path,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",  # We support flash_attention_2; sdpa; eager
            "trust_remote_code": True
        },
        trust_remote_code=True,
        tokenizer_kwargs={"padding_side": "left"},
        device="cuda",
    )
tqdm.pandas()
texts = train_sentences['sentences'].tolist()
for i, text in tqdm(enumerate(texts)):
    texts[i] = texts[i].split(".")[0]




embeddings_np = model.encode(texts, show_progress_bar=True, batch_size=64)
train_sentences['sentences'] = list(embeddings_np)


train_sentences['submitted_date'] = pd.to_datetime(train_sentences['submitted_date'])


train_sentences['time'] = (train_sentences['submitted_date'].dt.year-2000)/2


from lightgbm import LGBMRegressor
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import sklearn
X = np.vstack(train_sentences['sentences'].values)
X_df = pd.DataFrame(X, columns=[f'embedding_{i}' for i in range(X.shape[1])])

y = train_sentences['time']

X_train, X_val, y_train, y_val = train_test_split(X_df, y, test_size=0.2)




modelgbm = sklearn.svm.SVR(kernel='rbf')
modelgbm.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error
val_pred = modelgbm.predict(X_val)
len(val_pred)


acc = mean_absolute_error(y_val, val_pred)
print(acc)


sample_submission = pd.read_csv('/kaggle/input/kazakhstan-respa-final-day-2-late-competition/sample_submission.csv')


def predict_in_batches(test_sentences, model, modelgbm, batch_size=64):
    """
    Encodes sentences in batches and makes predictions using the combined embeddings.
    """
    
    print("Encoding first sentences in batches...")
    # Encode all first sentences in a single batch operation (handled internally by sentence-transformers)
    # This is much faster than the loop
    embeddings_first = model.encode(test_sentences['first_sentence'].tolist(), 
                                    batch_size=batch_size, 
                                    show_progress_bar=True)
    
    print("Encoding second sentences in batches...")
    # Encode all second sentences in a single batch operation
    embeddings_second = model.encode(test_sentences['second_sentence'].tolist(), 
                                     batch_size=batch_size, 
                                     show_progress_bar=True)
    
    print("Making predictions using the combined embeddings...")
    
    # Combine the embeddings into a single feature matrix for the GBM model
    # You might want to experiment with other combinations (e.g., concatenation, cosine similarity)
    # The original code compared predictions from two separate inputs, so we replicate that logic efficiently.

    # Option 1: Replicate original logic for comparison of individual predictions
    # This still requires two calls to modelgbm.predict but on full datasets.
    pred1_scores = modelgbm.predict(embeddings_first)
    pred2_scores = modelgbm.predict(embeddings_second)
    
    # Compare the scores element-wise and generate final predictions (0 or 1)
    # np.where is a fast, vectorized way to do this comparison
    pred = np.where(pred1_scores > pred2_scores, 0, 1).tolist()
    
    # Option 2: Combine embeddings into a single feature vector before prediction (More typical for sentence-pair tasks)
    # combined_embeddings = np.hstack((embeddings_first, embeddings_second))
    # pred = modelgbm.predict(combined_embeddings).tolist()
    
    return pred

# Example usage (assuming you have test_sentences as a DataFrame, model, and modelgbm loaded)
# pred = predict_in_batches(test_sentences, model, modelgbm, batch_size=128)


pred = predict_in_batches(test_sentences, model, modelgbm, batch_size=64)




submission = pd.DataFrame({
    'ID': sample_submission['ID'],
    'label': pred
})


submission.to_csv('submission.csv',index = False)




