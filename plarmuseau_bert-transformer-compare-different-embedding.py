import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib

# Optimized SBertTransformer
class SBertTransformer:
    def __init__(self, model_size='NLP Large', normalize=True, device='cpu'):
        self.model_size = model_size
        self.normalize = normalize
        self.device = device
        self.model_map = {
            'Mini': 'all-MiniLM-L6-v2',         # 384D
            'Medium': 'all-MiniLM-L12-v2',      # 384D
            'Base' : 'bert-base-uncased',       #768D
            'NLP Large': 'all-mpnet-base-v2',   # 768D
            'Large' :'bert-large-uncased',       #1024D
            'Maxi': 'all-roberta-large-v1',     # 1024D
        }
        model_name = self.model_map.get(model_size, model_size)
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Loaded SBERT model: {model_name}")
    
    def transform(self, texts, batch_size=64, show_progress_bar=False, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)
        print(f"Generating embeddings for {len(texts)} samples...")
        embeddings = self.model.encode(
            texts.tolist(),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress_bar,
            device=self.device
        )
        if cache_path:
            np.save(cache_path, embeddings)
        return embeddings

# Training and evaluation function
def traintest(X, y, Xtemb,name,Testpd ,verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(solver='saga', max_iter=500, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    Testpd['Category']=clf.predict(Xtemb)
    Testpd[['Id','Category']].to_csv('submit'+name+'.csv',index=False)
    acc = accuracy_score(y_test, y_pred)
    if verbose:
        print(f"\n{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    return acc, X.shape[1]

# Load dataset (only needed columns)
train = pd.read_csv('/kaggle/input/inst-414-spring-2025/train.csv', usecols=['review', 'label'])
test =pd.read_csv('/kaggle/input/inst-414-spring-2025/test.csv')
# Run experiments
results = []
model_sizes = ['Mini', 'Medium','Base', 'NLP Large','Large', 'Maxi']

for size in model_sizes:
    transformer = SBertTransformer(model_size=size, normalize=True, device='cuda')
    cache_file = f'{size.lower().replace(" ", "_")}_embeddings.npy'
    test_cache_file  = f'{size.lower().replace(" ", "_")}_test_embeddings.npy'
    
    embeddings = transformer.transform(train['review'], cache_path=cache_file)
    Tembed = transformer.transform(test['review'], cache_path=test_cache_file)

    acc, dim = traintest(embeddings, train['label'],Tembed, size, test,verbose=True)
    results.append({'model': size, 'accuracy': acc, 'dimension': dim})
    print(f"{size} - Accuracy: {acc:.4f}, Dimension: {dim}")

# Plot results
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.plot(results_df['dimension'], results_df['accuracy'], marker='o', linestyle='-', color='b')
for _, row in results_df.iterrows():
    plt.text(row['dimension'], row['accuracy'] + 0.005, row['model'], fontsize=10)
plt.title('Test Accuracy vs. Embedding Dimension')
plt.xlabel('Embedding Dimension')
plt.ylabel('Test Accuracy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_vs_dimension.png')
plt.show()


