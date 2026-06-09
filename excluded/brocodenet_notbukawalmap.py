import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier


# os.getcwd()


# os.chdir('d:\\ds_ridwan\\semua_kompetisi_kaggle\\map charting students\data')


df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
submission_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')


# df = pd.read_csv('train.csv')
df.head(10)


df.columns


for col in df.columns[-2:]:
    print(f'â–¶ï¸�{col} have unique value like\n{df[col].unique()}')


df.isnull().sum() / len(df)*100


df.Misconception.value_counts()


print(df.sample(10))


df.iloc[15106,2]


df.iloc[10711,4]


df.Misconception.unique()


df.Category.value_counts(normalize=True)


df['target'] = df['Category']+df['Misconception'].fillna('NA')
# fillna disini biar dia bisa digabung


seperator = '[SEP]'
text_cols = ['QuestionText', 'MC_Answer','StudentExplanation']


df['target'].nunique()


# test = pd.read_csv('test.csv')
test.tail()


test.isnull().sum() # aman gaada yg kosong


for data in [df, test]:
    # Ganti NaN di StudentExplanation dengan string kosong (kalo ada yg NaN)
    # df['StudentExplanation'] = df['StudentExplanation'].fillna('')
    data['input_text'] = data[text_cols].apply(lambda x: seperator.join(x), axis=1)


df.tail()


le = LabelEncoder()
df['target_encoded'] = le.fit_transform(df['target'])


int_to_label = {i: label for i, label in enumerate(le.classes_)}


# OneVsRestClassifier digunakan karena kita punya banyak kelas target.
# LogisticRegression adalah classifier yang cepat dan bagus untuk baseline.
# n_jobs=-1 berarti menggunakan semua core CPU agar lebih cepat.
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(token_pattern=r'(?u)\b\w+\b|\S')), # Tokenizer yang lebih baik
    ('clf', OneVsRestClassifier(MultinomialNB(), n_jobs=-1))
])


# Latih model dengan data training
model_pipeline.fit(df['input_text'], df['target_encoded'])


# --- 4. Prediksi untuk Test Set ---
print("ğŸ”� Membuat prediksi...")

# Dapatkan probabilitas untuk setiap kelas pada data test
test_probabilities = model_pipeline.predict_proba(test['input_text'])

# Ambil 3 prediksi teratas untuk setiap baris data (sesuai MAP@3)
top_3_preds = []
for i in range(len(test_probabilities)):
    # Dapatkan indeks dari 3 probabilitas tertinggi
    top_indices = np.argsort(test_probabilities[i])[-3:][::-1]
    
    # Ubah indeks kembali ke label string asli
    top_labels = [int_to_label[idx] for idx in top_indices]
    
    # Gabungkan menjadi satu string sesuai format submission
    top_3_preds.append(' '.join(top_labels))



top_3_preds


submission_df['Category:Misconception'][1]


# Masukkan hasil prediksi ke dalam dataframe submission
submission_df['Category:Misconception'] = top_3_preds

# Simpan ke file csv
submission_df.to_csv('submission.csv', index=False)

print("âœ… Selesai! File submission.csv sudah siap.")
print("Contoh prediksi:")
print(submission_df.head())

