!pip install sentence-transformers 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import re
import warnings
warnings.filterwarnings('ignore')


class RedditRuleViolationClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.sentence_model = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.model = None

    def preprocess_text(self, text):
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text)
        text = re.sub(r'@\w+|/u/\w+', '[USER]', text)
        text = re.sub(r'/r/\w+', '[SUBREDDIT]', text)
        text = ' '.join(text.split())
        return text

    def extract_features(self, df):
        df['processed_body'] = df['body'].apply(self.preprocess_text)
        embeddings = self.sentence_model.encode(df['processed_body'].tolist(), show_progress_bar=True)
        text_length = df['body'].str.len().fillna(0).values.reshape(-1, 1)
        word_count = df['body'].str.split().str.len().fillna(0).values.reshape(-1, 1)
        caps_ratio = df['body'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0).values.reshape(-1, 1)
        exclamation_count = df['body'].str.count('!').fillna(0).values.reshape(-1, 1)
        question_count = df['body'].str.count('\?').fillna(0).values.reshape(-1, 1)
        rule_encoded = pd.Categorical(df['rule']).codes.reshape(-1, 1)
        subreddit_encoded = pd.Categorical(df['subreddit']).codes.reshape(-1, 1)
        return np.hstack([embeddings, text_length, word_count, caps_ratio,
                          exclamation_count, question_count, rule_encoded, subreddit_encoded])

    def train(self, df):
        print("Data shape:", df.shape)
        print("Target distribution:", df['rule_violation'].value_counts(normalize=True).to_dict())
        print("Rules:", df['rule'].nunique(), "unique rules")
        print("Subreddits:", df['subreddit'].nunique(), "unique subreddits")
        print("\nExtracting features...")
        X = self.extract_features(df)
        y = df['rule_violation'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        }
        best_score, best_model, best_name = 0, None, None
        print("\nModel evaluation:")
        for name, model in models.items():
            scores = cross_val_score(model, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc')
            mean_score = scores.mean()
            print(f"{name}: {mean_score:.4f} (+/- {scores.std() * 2:.4f})")
            if mean_score > best_score:
                best_score, best_model, best_name = mean_score, model, name
        print(f"\nUsing {best_name} (AUC: {best_score:.4f})")
        self.model = best_model
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        print("\nTest Results:")
        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        # Store the test data corresponding to the split
        self.test_data = (df.iloc[y_train.shape[0]:].reset_index(drop=True), y_test, y_pred, y_prob)
        return self

    def predict(self, df):
        X = self.extract_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return predictions, probabilities

    def show_examples(self, n=5):
        if not hasattr(self, 'test_data'):
            print("No test data available")
            return
        test_df, y_true, y_pred, y_prob = self.test_data
        indices = np.random.choice(len(test_df), min(n, len(test_df)), replace=False)
        print(f"\n{n} Random Predictions:")
        for i, idx in enumerate(indices):
            # Use the index to access elements from the test_df and the corresponding arrays
            text = test_df.iloc[idx]['body']
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"\n{i+1}. Text: {text}")
            print(f"   Rule: {test_df.iloc[idx]['rule']}")
            print(f"   Subreddit: {test_df.iloc[idx]['subreddit']}")
            print(f"   Actual: {y_true[idx]}, Predicted: {y_pred[idx]}, Prob: {y_prob[idx]:.3f}")


def main():
    train_df = pd.read_csv('/kaggle/input/jigsaw-kaggle/train.csv')
    print("Loaded train.csv successfully")
    classifier = RedditRuleViolationClassifier()
    classifier.train(train_df)
    classifier.show_examples()

    test_df = pd.read_csv('/kaggle/input/jigsaw-kaggle/test.csv')
    print(f"\nLoaded test.csv with {len(test_df)} samples")
    predictions, probabilities = classifier.predict(test_df)
    predictions, probabilities = classifier.predict(test_df)

        # Ensure all test data is predicted and saved
    submission = pd.DataFrame({
            'row_id': test_df.index,
            'rule_violation': probabilities
        })

    print(submission)
    submission.to_csv('submission.csv', index=False)

    return classifier


classifier = main()

