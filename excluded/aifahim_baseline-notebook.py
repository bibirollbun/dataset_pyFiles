import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


train_path = '/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv'
train_df = pd.read_csv(train_path)

print(f"Training data shape: {train_df.shape}")
print(f"Class distribution:")
print(train_df['sentiment'].value_counts())



model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 3))),
    ('classifier', MultinomialNB())
])

print("\nTraining model on full dataset...")
model.fit(train_df['text'], train_df['sentiment'])
print("Model training complete")




predictions = model.predict(train_df['text'])


submission = pd.DataFrame({
    'id': train_df['id'],
    'sentiment': predictions
})


submission.to_csv('submission.csv', index=False)
print(f"\nCreated submission.csv with {len(submission)} rows")


print("\nDistribution of predictions:")
print(submission['sentiment'].value_counts())


print("\nFirst 5 rows of submission.csv:")
print(submission.head())

