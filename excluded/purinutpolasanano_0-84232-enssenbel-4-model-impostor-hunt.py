!pip install scikit-learn==1.4.2


import pandas as pd
import os
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(data_dir, train_csv_path):
    """Load and process training data with better error handling"""
    train_df = pd.read_csv(train_csv_path)
    data = []

    for _, row in train_df.iterrows():
        folder_id = str(row.iloc[0])
        real_text_id = str(row.iloc[1])

        folder_name = f"article_{folder_id.zfill(4)}"
        folder_path = os.path.join(data_dir, folder_name)

        for file_id in ["1", "2"]:
            file_path = os.path.join(folder_path, f"file_{file_id}.txt")

            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read().strip()
                if not text:
                    continue

                label = 1 if file_id == real_text_id else 0
                data.append({'text': text, 'real': label, 'folder': folder_id})

            except (FileNotFoundError, UnicodeDecodeError) as e:
                print(f"⚠️ Error loading {file_path}: {str(e)}")
                continue

    df = pd.DataFrame(data)
    return df.dropna(subset=['text'])


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.punct_table = str.maketrans('', '', string.punctuation)

    def preprocess(self, text):
        text = text.lower()

        text = text.translate(self.punct_table)

        tokens = [self.lemmatizer.lemmatize(word)
                 for word in text.split()
                 if word not in self.stop_words]

        return ' '.join(tokens)


def create_features(df, vectorizer=None, mode='train'):
    """Create features with TF-IDF and additional text statistics"""
    preprocessor = TextPreprocessor()

    df['clean_text'] = df['text'].apply(preprocessor.preprocess)

    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['text'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)

    if mode == 'train':
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        X_tfidf = vectorizer.fit_transform(df['clean_text'])
        return X_tfidf, vectorizer, df
    else:
        if vectorizer is None:
            raise ValueError("Vectorizer must be provided for test mode")
        X_tfidf = vectorizer.transform(df['clean_text'])
        return X_tfidf, df


def train_models(X, y):
    """Train models with cross-validation and hyperparameter tuning"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=0.1,
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.5
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced_subsample',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ),
        'SVM': CalibratedClassifierCV(
            SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True
            ),
            cv=3
        )
    }

    trained_models = {}
    val_scores = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model

        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)
        val_scores[name] = {'accuracy': acc, 'f1': f1}

        print(f"{name} Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        print(classification_report(y_val, val_preds))

    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft',
        n_jobs=-1
    )
    voting_clf.fit(X_train_res, y_train_res)
    trained_models['Ensemble'] = voting_clf

    val_preds = voting_clf.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    val_scores['Ensemble'] = {'accuracy': acc, 'f1': f1}
    print(f"\nEnsemble Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print(classification_report(y_val, val_preds))

    return trained_models, val_scores


def predict_test(models, val_scores, X_test, test_df):
    """Make predictions with fallback strategies"""

    proba_dfs = []
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_test)[:, 1]
                proba_dfs.append(pd.DataFrame({
                    'folder': test_df['folder'].values,
                    'file_id': test_df['file_id'].values,
                    f'proba_{name}': proba
                }))
            except Exception as e:
                print(f"⚠️ Error getting probabilities from {name}: {str(e)}")
                continue

    if not proba_dfs:
        raise ValueError("No probability data was generated from any model")

    proba_df = proba_dfs[0]
    for df in proba_dfs[1:]:
        proba_df = proba_df.merge(df, on=['folder', 'file_id'])

    try:
        proba_df['file_number'] = proba_df['file_id'].str.extract(r'file_(\d+)').astype(int)
        proba_df['article_id'] = proba_df['folder'].str.extract(r'article_(\d+)').astype(int)
    except Exception as e:
        print(f"⚠️ Error extracting file numbers or article IDs: {str(e)}")
        proba_df['file_number'] = proba_df['file_id'].apply(lambda x: int(x.split('_')[-1]))
        proba_df['article_id'] = proba_df.index

    best_model = max(val_scores.items(), key=lambda x: x[1]['f1'])[0]

    final_selection = []
    for article_id in proba_df['article_id'].unique():
        try:
            article_files = proba_df[proba_df['article_id'] == article_id].copy()

            if len(article_files) == 0:
                print(f"⚠️ No files found for article {article_id}")
                continue

            article_files['best_model_rank'] = article_files[f'proba_{best_model}'].rank(ascending=False)

            proba_cols = [col for col in article_files.columns if col.startswith('proba_')]
            article_files['ensemble_proba'] = article_files[proba_cols].mean(axis=1)
            article_files['ensemble_rank'] = article_files['ensemble_proba'].rank(ascending=False)

            selected = None
            try:
                best_model_choice = article_files[article_files['best_model_rank'] == 1].iloc[0]

                if best_model_choice[f'proba_{best_model}'] > 0.6:
                    selected = best_model_choice
                else:
                    ensemble_choice = article_files[article_files['ensemble_rank'] == 1].iloc[0]
                    selected = ensemble_choice
            except IndexError:
                selected = article_files.iloc[0]
                print(f"⚠️ Used fallback selection for article {article_id}")

            final_selection.append({
                'id': int(selected['article_id']),
                'real_text_id': int(selected['file_number']),
                'confidence': max(selected[f'proba_{best_model}'], selected.get('ensemble_proba', 0))
            })

        except Exception as e:
            print(f"⚠️ Error processing article {article_id}: {str(e)}")
            continue

    if not final_selection:
        raise ValueError("No articles were processed successfully")

    submission_df = pd.DataFrame(final_selection)
    return submission_df.sort_values('id')


if __name__ == "__main__":
    data_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
    train_csv_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
    test_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

    print("Loading training data...")
    df = load_data(data_dir, train_csv_path)

    print("Loading test data...")
    test_data = []
    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)
        if os.path.isdir(folder_path):
            for file_id in ["1", "2"]:
                file_path = os.path.join(folder_path, f"file_{file_id}.txt")
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        text = f.read().strip()
                    if text:
                        test_data.append({'folder': folder, 'file_id': f'file_{file_id}', 'text': text})
                except (FileNotFoundError, UnicodeDecodeError) as e:
                    print(f"⚠️ Error loading test file {file_path}: {str(e)}")
                    continue
    test_df = pd.DataFrame(test_data)

    print("Creating features...")
    X, vectorizer, df = create_features(df, mode='train')
    y = df['real'].astype(int)

    if test_df.empty:
        print("⚠️ Warning: No test data was loaded!")
        submission_df = pd.DataFrame(columns=['id', 'real_text_id'])
    else:
        X_test, test_df = create_features(test_df, vectorizer=vectorizer, mode='test')

        print("\nTraining models...")
        trained_models, val_scores = train_models(X, y)

        print("\nMaking predictions...")
        submission_df = predict_test(trained_models, val_scores, X_test, test_df)

    submission_df[['id', 'real_text_id']].to_csv("submission.csv", index=False)
    print("\n✅ Enhanced submission created: submission.csv")

