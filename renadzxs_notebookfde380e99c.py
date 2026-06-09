import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
train = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')
test = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')

# Ù†Ù…ÙˆØ°Ø¬ Ø¨Ø³ÙŠØ· Ø¨Ø§Ø³ØªØ®Ø¯Ø§Ù… TF-IDF Ùˆ Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000)),
])
print(train.columns)

# Ø§Ù„ØªØ¯Ø±ÙŠØ¨
pipeline.fit(train['Question'], train['label'])

# Ø§Ù„ØªÙ†Ø¨Ø¤ Ø¹Ù„Ù‰ Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±
preds = pipeline.predict(test['Question'])

# ØªØ¬Ù‡ÙŠØ² Ù…Ù„Ù� Ø§Ù„ØªØ³Ù„ÙŠÙ…
submission = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv')
submission['label'] = preds
submission.to_csv('submission.csv', index=False)

print("âœ… ØªÙ… ØªØ¬Ù‡ÙŠØ² Ø§Ù„Ù…Ù„Ù�! Ø­Ø§ÙˆÙ„ ØªØ³ÙˆÙŠ Submit ÙˆØ´ÙˆÙ� ØªØ±ØªÙŠØ¨Ùƒ ğŸ”¥")


