# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import csv
import math

# Örnek bir cosine similarity fonksiyonu (sadece standart Python kullanır)
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude = math.sqrt(sum(a**2 for a in vec1)) * math.sqrt(sum(b**2 for b in vec2))
    if not magnitude:
        return 0
    return dot_product / magnitude

# Basit kelime frekansı hesaplama
def vectorize(text, vocabulary):
    words = text.split()
    return [words.count(word) for word in vocabulary]

# CSV dosyasını okuma
file_path = "/kaggle/input/santa-2024/sample_submission.csv"
with open(file_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Tüm metinleri alın ve bir kelime dağarcığı oluşturun
sentences = [row['text'] for row in data]
vocabulary = set(word for sentence in sentences for word in sentence.split())

# Cümleleri vektörleştirin
vectors = [vectorize(sentence, vocabulary) for sentence in sentences]

# Benzerlik matrisini oluşturun
similarity_matrix = [
    [cosine_similarity(vec1, vec2) for vec2 in vectors]
    for vec1 in vectors
]

# En az benzer cümleyi bul
start_index = min(range(len(similarity_matrix)), key=lambda i: sum(similarity_matrix[i]))

# Benzerlik sırasına göre cümleleri sırala
ordered_indices = [start_index]
visited = {start_index}

for _ in range(len(sentences) - 1):
    last_index = ordered_indices[-1]
    next_index = max(
        (idx for idx in range(len(sentences)) if idx not in visited),
        key=lambda idx: similarity_matrix[last_index][idx]
    )
    ordered_indices.append(next_index)
    visited.add(next_index)

ordered_sentences = [sentences[i] for i in ordered_indices]

# Yeni CSV dosyasını oluşturma
output_file = "submission.csv"
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["id", "text"])
    writer.writeheader()
    for i, sentence in enumerate(ordered_sentences):
        writer.writerow({"id": data[i]["id"], "text": sentence})

print(f"Dosya '{output_file}' oluşturuldu.")


