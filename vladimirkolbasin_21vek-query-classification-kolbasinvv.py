import pandas as pd

# Ğ§Ñ‚ĞµĞ½Ğ¸Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
categories = pd.read_csv('/kaggle/input/21vek-query-classification/categories.csv')
train = pd.read_csv('/kaggle/input/21vek-query-classification/train.csv')
test = pd.read_csv('/kaggle/input/21vek-query-classification/test.csv')


# Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ
import seaborn as sns
import matplotlib.pyplot as plt

train['query_length'] = train['Query'].apply(len)

plt.figure(figsize=(12, 5))
sns.histplot(train['query_length'], bins=50, kde=True)
plt.xlabel('Ğ”Ğ»Ğ¸Ğ½Ğ° Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ°')
plt.ylabel('Ğ§Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ°')
plt.title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ´Ğ»Ğ¸Ğ½Ñ‹ Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ¾Ğ²')
plt.show()


# ĞŸĞ¾Ğ´Ñ�Ñ‡ĞµÑ‚ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ñ�Ğ»Ğ¾Ğ² Ğ² Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğµ
train['query_word_count'] = train['Query'].apply(lambda x: len(x.split()))

# Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ñ�Ğ»Ğ¾Ğ²
plt.figure(figsize=(12, 5))
sns.histplot(train['query_word_count'], bins=30, kde=True)
plt.xlabel('ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ�Ğ»Ğ¾Ğ² Ğ² Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğµ')
plt.ylabel('Ğ§Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ°')
plt.title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ñ�Ğ»Ğ¾Ğ² Ğ² Ğ¿Ğ¾Ğ¸Ñ�ĞºĞ¾Ğ²Ñ‹Ñ… Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ°Ñ…')
plt.show()


train[['query_length', 'query_word_count']].describe()


# Ğ“Ñ€ÑƒĞ¿Ğ¿Ğ¸Ñ€Ğ¾Ğ²ĞºĞ° Ğ¸ Ğ¿Ğ¾Ğ´Ñ�Ñ‡Ñ‘Ñ‚ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ¾Ğ² Ğ¿Ğ¾ ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¹ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸
category_counts = train.groupby('CategoryID').size().reset_index(name='Count')

# Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ�Ğ»Ğ¾Ğ²Ğ°Ñ€ÑŒ Ğ´Ğ»Ñ� Ñ�Ğ¾Ğ¿Ğ¾Ñ�Ñ‚Ğ°Ğ²Ğ»ĞµĞ½Ğ¸Ñ� CategoryID Ğ¸ CategoryName
cat_map = dict(zip(categories['CategoryID'], categories['CategoryName']))

# Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ñ�Ñ‚Ğ¾Ğ»Ğ±ĞµÑ† Ñ� Ğ½Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸ĞµĞ¼ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸, Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒÑ� Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ½Ñ‹Ğ¹ Ñ�Ğ»Ğ¾Ğ²Ğ°Ñ€ÑŒ
category_counts['CategoryName'] = category_counts['CategoryID'].map(cat_map)

# Ğ¡Ğ¾Ñ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¿Ğ¾ Ğ²Ğ¾Ğ·Ñ€Ğ°Ñ�Ñ‚Ğ°Ğ½Ğ¸Ñ� ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ğ¸ Ğ²Ñ‹Ğ²Ğ¾Ğ´Ğ¸Ğ¼ 10 ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹ Ñ� Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¼ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾Ğ¼ Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ¾Ğ²
print('10 ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹ Ñ� Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¼ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾Ğ¼ Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ¾Ğ²')
min_10_categories = category_counts.sort_values(by='Count', ascending=True).head(10)
print(min_10_categories[['CategoryName', 'Count']])
print('\n10 ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹ Ñ� Ğ¼Ğ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¼ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾Ğ¼ Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ¾Ğ²')
max_10_categories = category_counts.sort_values(by='Count', ascending=True).tail(10)
print(max_10_categories[['CategoryName', 'Count']])


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score

# Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ‚ÑŒ Ğ¸ ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ� Ğ±ÑƒĞ´ĞµĞ¼ Ğ´ĞµĞ»Ğ°Ñ‚ÑŒ Ğ½Ğ° Ğ¿Ğ¾Ğ»Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…, Ñ‡Ñ‚Ğ¾Ğ±Ñ‹ 
# Ğ½Ğµ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�Ñ‚Ğ¸Ñ‚ÑŒ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹ Ñ� Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¼ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾Ğ¼ Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ¾Ğ²
X_train, y_train = train['Query'], train['CategoryID']

# Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¿Ğ°Ğ¹Ğ¿Ğ»Ğ°Ğ¹Ğ½ Ñ� char_wb n-grams 2-10, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğ¹ Ğ¿Ğ¾ĞºĞ°Ğ·Ğ°Ğ» Ğ¼Ğ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Macro F1 Ğ½Ğ° Private Score
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer= 'char_wb',
                              ngram_range=(2, 10))),
    ('clf', LogisticRegression(max_iter=25, solver='liblinear', C=50, class_weight='balanced'))
])

# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ°Ğ¸Ğ²Ğ°ĞµĞ¼ 5-fold ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ�
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Ğ�Ñ†ĞµĞ½Ğ¸Ğ²Ğ°ĞµĞ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ñ� Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ğ¸ĞµĞ¼ Macro F1 (5-fold)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='f1_macro', n_jobs=-1)
print("5-fold CV Macro F1 Scores:", cv_scores)
print("Mean CV Macro F1 Score:", np.mean(cv_scores))

# Ğ�Ğ±ÑƒÑ‡Ğ°ĞµĞ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ½Ğ° Ğ²Ñ�ĞµĞ¹ Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞµ
pipeline.fit(X_train, y_train)


'''
Ğ�Ğ¿Ñ€ĞµĞ´ĞµĞ»Ğ¸Ğ¼, ĞºĞ°ĞºĞ¸Ğµ Ñ‡Ğ°Ñ�Ñ‚Ğ¸ Ñ�Ğ»Ğ¾Ğ² Ğ¸Ğ³Ñ€Ğ°Ñ�Ñ‚ ĞºĞ»Ñ�Ñ‡ĞµĞ²ÑƒÑ� Ñ€Ğ¾Ğ»ÑŒ Ğ² Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğ¸ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹. 
Ğ¡Ğ½Ğ°Ñ‡Ğ°Ğ»Ğ° Ğ¾Ğ½ Ğ¸Ğ·Ğ²Ğ»ĞµĞºĞ°ĞµÑ‚ ĞºĞ¾Ñ�Ñ„Ñ„Ğ¸Ñ†Ğ¸ĞµĞ½Ñ‚Ñ‹ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ½Ğ¾Ğ¹ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ğ¿Ğ¾ĞºĞ°Ğ·Ñ‹Ğ²Ğ°Ñ�Ñ‚ Ğ²ĞºĞ»Ğ°Ğ´ Ğ¾Ñ‚Ğ´ĞµĞ»ÑŒĞ½Ñ‹Ñ… n-grams 
(Ñ‡Ğ°Ñ�Ñ‚ĞµĞ¹ Ñ�Ğ»Ğ¾Ğ²) Ğ² Ğ¿Ñ€Ğ¾Ñ†ĞµÑ�Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸. Ğ—Ğ°Ñ‚ĞµĞ¼ Ğ¾Ğ½ Ğ¾Ğ¿Ñ€ĞµĞ´ĞµĞ»Ñ�ĞµÑ‚ 7 Ñ�Ğ°Ğ¼Ñ‹Ñ… Ğ·Ğ½Ğ°Ñ‡Ğ¸Ğ¼Ñ‹Ñ… Ñ„Ñ€Ğ°Ğ³Ğ¼ĞµĞ½Ñ‚Ğ¾Ğ² Ñ‚ĞµĞºÑ�Ñ‚Ğ° 
Ğ´Ğ»Ñ� ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¹ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸, Ñ‚Ğ¾ ĞµÑ�Ñ‚ÑŒ Ñ‚Ğµ Ñ�Ğ»ĞµĞ¼ĞµĞ½Ñ‚Ñ‹, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ñ�Ñ‡Ğ¸Ñ‚Ğ°ĞµÑ‚ Ğ½Ğ°Ğ¸Ğ±Ğ¾Ğ»ĞµĞµ Ğ²Ğ°Ğ¶Ğ½Ñ‹Ğ¼Ğ¸ Ğ¿Ñ€Ğ¸ Ğ¿Ñ€Ğ¸Ğ½Ñ�Ñ‚Ğ¸Ğ¸ Ñ€ĞµÑˆĞµĞ½Ğ¸Ğ¹.
'''
clf = pipeline.named_steps['clf']
vectorizer = pipeline.named_steps['tfidf']
feature_names = vectorizer.get_feature_names_out()

# Ğ’ Ğ¸Ñ‚Ğ¾Ğ³Ğµ Ğ¿Ğ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ Ğ¸Ğ½Ñ‚ÑƒĞ¸Ñ‚Ğ¸Ğ²Ğ½Ğ¾Ğµ Ğ¿Ñ€ĞµĞ´Ñ�Ñ‚Ğ°Ğ²Ğ»ĞµĞ½Ğ¸Ğµ Ğ¾ Ñ‚Ğ¾Ğ¼, ĞºĞ°ĞºĞ¸Ğµ Ñ�Ğ»Ğ¾Ğ²Ğ° Ğ¸Ğ»Ğ¸ Ğ¸Ñ… Ñ‡Ğ°Ñ�Ñ‚Ğ¸ Ğ¾ĞºĞ°Ğ·Ñ‹Ğ²Ğ°Ñ�Ñ‚ Ğ½Ğ°Ğ¸Ğ±Ğ¾Ğ»ÑŒÑˆĞµĞµ
# Ğ²Ğ»Ğ¸Ñ�Ğ½Ğ¸Ğµ Ğ½Ğ° Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¹ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸:
categories_unique = clf.classes_
for cat in categories_unique:
    idx = list(categories_unique).index(cat)
    coefs = clf.coef_[idx]
    top7_idx = np.argsort(coefs)[-7:]
    top7_words = [feature_names[i] for i in top7_idx]
    cat_name = cat_map.get(cat, cat)
    print(f"ĞšĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ñ�: {cat_name}, Ñ‚Ğ¾Ğ¿ Ñ�Ğ»Ğ¾Ğ²: {top7_words[::-1]}")


# ĞŸĞ¾Ñ�Ñ‚Ñ€Ğ¾ĞµĞ½Ğ¸Ğµ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğ¹ Ğ´Ğ»Ñ� Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ³Ğ¾ Ğ½Ğ°Ğ±Ğ¾Ñ€Ğ°
test_preds = pipeline.predict(test['Query'])
submission = pd.DataFrame({'ID': test['ID'], 'CategoryID': test_preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)


from lime.lime_text import LimeTextExplainer

# Ğ‘Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞºĞ° LIME (Local Interpretable Model-agnostic Explanations) Ğ¿Ñ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµÑ‚Ñ�Ñ� Ğ´Ğ»Ñ� Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ° Ñ‚Ğ¾Ğ³Ğ¾, 
# ĞºĞ°ĞºĞ¸Ğµ Ñ�Ğ»ĞµĞ¼ĞµĞ½Ñ‚Ñ‹ Ñ‚ĞµĞºÑ�Ñ‚Ğ° Ğ¾ĞºĞ°Ğ·Ğ°Ğ»Ğ¸ Ğ½Ğ°Ğ¸Ğ±Ğ¾Ğ»ÑŒÑˆĞµĞµ Ğ²Ğ»Ğ¸Ñ�Ğ½Ğ¸Ğµ Ğ½Ğ° Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğµ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸.
class_names = [cat_map.get(cls, str(cls)) for cls in clf.classes_]
explainer_lime = LimeTextExplainer(class_names=class_names)

# Ğ’Ñ‹Ğ±Ğ¸Ñ€Ğ°ĞµĞ¼ Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€ Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ° Ğ¸Ğ· Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸
sample_query = X_train.iloc[11540]
print("\nĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ° Ğ´Ğ»Ñ� Ğ¾Ğ±ÑŠÑ�Ñ�Ğ½ĞµĞ½Ğ¸Ñ� (LIME):", sample_query)
exp = explainer_lime.explain_instance(sample_query, pipeline.predict_proba, num_features=7)
# ĞšĞ°Ğº Ñ�Ñ‚Ğ¾ Ñ€Ğ°Ğ±Ğ¾Ñ‚Ğ°ĞµÑ‚:
#ğŸ”¹ Ğ¡Ğ¾Ğ·Ğ´Ğ°Ñ�Ñ‚Ñ�Ñ� Ğ¸Ğ¼ĞµĞ½Ğ° ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² Ğ´Ğ»Ñ� Ğ»ÑƒÑ‡ÑˆĞµĞ¹ Ğ¸Ğ½Ñ‚ĞµÑ€Ğ¿Ñ€ĞµÑ‚Ğ°Ñ†Ğ¸Ğ¸ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğ¹. 
#ğŸ”¹ Ğ’Ñ‹Ğ±Ğ¸Ñ€Ğ°ĞµÑ‚Ñ�Ñ� Ğ¾Ğ´Ğ¸Ğ½ Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ� Ğ¸Ğ· Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸ Ğ´Ğ»Ñ� Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ°. 
#ğŸ”¹ LIME Ğ¾Ğ±ÑŠÑ�Ñ�Ğ½Ñ�ĞµÑ‚ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğµ, Ğ¿Ğ¾ĞºĞ°Ğ·Ñ‹Ğ²Ğ°Ñ�, ĞºĞ°ĞºĞ¸Ğµ Ñ�Ğ»Ğ¾Ğ²Ğ° Ğ¸Ğ»Ğ¸ Ğ¸Ñ… Ñ‡Ğ°Ñ�Ñ‚Ğ¸ Ğ±Ñ‹Ğ»Ğ¸ Ğ½Ğ°Ğ¸Ğ±Ğ¾Ğ»ĞµĞµ Ğ·Ğ½Ğ°Ñ‡Ğ¸Ğ¼Ñ‹Ğ¼Ğ¸ Ğ´Ğ»Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸. 
#ğŸ”¹ Ğ“Ñ€Ğ°Ñ„Ğ¸Ñ‡ĞµÑ�ĞºĞ¾Ğµ Ğ¿Ñ€ĞµĞ´Ñ�Ñ‚Ğ°Ğ²Ğ»ĞµĞ½Ğ¸Ğµ Ğ¿Ğ¾Ğ·Ğ²Ğ¾Ğ»Ñ�ĞµÑ‚ Ğ²Ğ¸Ğ·ÑƒĞ°Ğ»ÑŒĞ½Ğ¾ Ğ¾Ñ†ĞµĞ½Ğ¸Ñ‚ÑŒ, ĞºĞ°ĞºĞ¸Ğµ Ñ„Ñ€Ğ°Ğ³Ğ¼ĞµĞ½Ñ‚Ñ‹ Ñ‚ĞµĞºÑ�Ñ‚Ğ° Ğ¿Ğ¾Ğ²Ğ»Ğ¸Ñ�Ğ»Ğ¸ Ğ½Ğ° Ñ€ĞµÑˆĞµĞ½Ğ¸Ğµ.
exp.show_in_notebook(text=True)

