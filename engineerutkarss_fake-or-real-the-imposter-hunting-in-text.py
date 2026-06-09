import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# Paths (adjust for Kaggle)
train_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train'
test_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test'
train_labels_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv'
# Load train CSV
train_labels_df = pd.read_csv(train_labels_path)


# Lists for training
train_texts = []
train_labels = []

# Prepare training data: both files per article labeled correctly
for _, row in train_labels_df.iterrows():
    art_num = row['id']
    real_file_num = row['real_text_id']
    for file_num in [1, 2]:
        file_path = os.path.join(train_dir, f'article_{art_num:04d}', f'file_{file_num}.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            label = 'real' if file_num == real_file_num else 'fake'
            train_texts.append(text)
            train_labels.append(label)

# Vectorize texts
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train = vectorizer.fit_transform(train_texts)


# Train LinearSVC
clf = LinearSVC()
clf.fit(X_train, train_labels)


predictions = {}
# Iterate over articles in test dir
for art_folder in sorted(os.listdir(test_dir)):
    art_path = os.path.join(test_dir, art_folder)
    if not os.path.isdir(art_path):
        continue
    art_num = int(art_folder.split('_')[1])
    
    texts = []
    file_nums = [1, 2]
    for file_num in file_nums:
        file_path = os.path.join(art_path, f'file_{file_num}.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    
    # Vectorize both files
    X_test_files = vectorizer.transform(texts)
    
    # Use decision function to pick more confident "real"
    distances = clf.decision_function(X_test_files)
    best_file_idx = distances.argmax()
    predictions[art_num] = file_nums[best_file_idx]


# Prepare submission DataFrame (one row per article) 
submission_df = pd.DataFrame({
    'id': list(predictions.keys()),
    'real_text_id': list(predictions.values())
}).sort_values('id')

submission_df.to_csv('submission.csv', index=False)
print("Submission file created with one real file per article: submission.csv")



df=pd.read_csv('/kaggle/working/submission.csv')
df


