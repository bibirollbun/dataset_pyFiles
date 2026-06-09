!pip install sentence-transformers





# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# 1. Load the data
train_df = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')

# Define the correct paths
base_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")

# 2. Map numeric IDs from train.csv to actual article directory names
if os.path.exists(train_dir):
    actual_articles = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"Found {len(actual_articles)} article directories in train folder")
    print(f"First 10 articles: {actual_articles[:10]}")
    
    # Create a mapping from numeric ID to article directory name
    id_to_article = {}
    article_to_id = {}
    
    for i, article in enumerate(actual_articles):
        if i < len(train_df):
            id_to_article[i] = article
            article_to_id[article] = i
    
    print(f"Created mapping for {len(id_to_article)} articles")
    print(f"Sample mapping: 0 -> {id_to_article.get(0, 'N/A')}, 1 -> {id_to_article.get(1, 'N/A')}")

# 3. Create sample submission from test directory structure
print("\nCreating sample submission file...")
if os.path.exists(test_dir):
    test_articles = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    print(f"Found {len(test_articles)} test articles")
    
    # Create numeric IDs for test articles (0, 1, 2, ...)
    test_numeric_ids = list(range(len(test_articles)))
    sample_sub = pd.DataFrame({'id': test_numeric_ids, 'real_text_id': [1] * len(test_articles)})
    
    # Create mapping for test articles
    test_id_to_article = {i: article for i, article in enumerate(test_articles)}
    test_article_to_id = {article: i for i, article in enumerate(test_articles)}
    
    print(f"Created sample submission with {len(test_articles)} test articles")
else:
    print("Test directory not found, creating empty submission")
    sample_sub = pd.DataFrame(columns=['id', 'real_text_id'])

# 4. Load the Sentence Transformer model
print("\nLoading Sentence Transformer model...")
embedder = SentenceTransformer('all-mpnet-base-v2')
print("Model loaded successfully!")

# 5. Function to read text files
def read_text_file(file_path):
    """Reads the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read().strip()
                return content
        except:
            print(f"Failed to read: {file_path}")
            return ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# 6. Function to get embeddings for a list of texts
def get_embeddings(texts):
    """Converts a list of texts into sentence embeddings."""
    return embedder.encode(texts, show_progress_bar=True, batch_size=16)

# 7. Process the TRAIN data using the correct mapping
print("\nProcessing TRAIN data...")
train_texts_1 = []
train_texts_2 = []
groups = []
train_labels = []

# Let's try a different approach - scan directory directly
print("Scanning train directory directly...")
for article in actual_articles:
    article_dir = os.path.join(train_dir, article)
    file1_path = os.path.join(article_dir, 'file_1.txt')
    file2_path = os.path.join(article_dir, 'file_2.txt')
    
    if os.path.exists(file1_path) and os.path.exists(file2_path):
        text1 = read_text_file(file1_path)
        text2 = read_text_file(file2_path)
        
        if text1 and text2 and len(text1) > 50 and len(text2) > 50:
            train_texts_1.append(text1)
            train_texts_2.append(text2)
            groups.append(article)
            
            # Get the numeric ID for this article to find the label
            numeric_id = article_to_id.get(article)
            if numeric_id is not None and numeric_id < len(train_df):
                label = train_df.iloc[numeric_id]['real_text_id']
                train_labels.append(label)
            else:
                # Default to 1 if we can't find the label
                train_labels.append(1)

if train_texts_1:
    print(f"Found {len(train_texts_1)} articles via direct scan")
    y = np.array(train_labels)
    groups = np.array(groups)
else:
    raise ValueError("No valid training data found!")

# Only proceed if we have training data
if len(train_texts_1) > 0:
    print(f"Generating embeddings for {len(train_texts_1)} text pairs...")
    embeddings_1 = get_embeddings(train_texts_1)
    embeddings_2 = get_embeddings(train_texts_2)

    # 8. Create feature vectors for each pair
    train_features = []
    for emb1, emb2 in zip(embeddings_1, embeddings_2):
        difference = np.abs(emb1 - emb2)
        product = emb1 * emb2
        combined_features = np.hstack((emb1, emb2, difference, product))
        train_features.append(combined_features)

    X = np.array(train_features)
    
    print(f"Feature matrix shape: {X.shape}")

    # 9. Train and validate using GroupKFold
    print("\nStarting Cross-Validation...")
    n_splits = min(5, len(np.unique(groups)))
    print(f"Using {n_splits} folds for cross-validation")
    gkf = GroupKFold(n_splits=n_splits)
    fold_accuracies = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Fold {fold+1}: Train size={len(X_train)}, Validation size={len(X_val)}")
        
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # Train without early stopping for simplicity
        model.fit(X_train, y_train)
        
        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        fold_accuracies.append(acc)
        models.append(model)
        print(f"Fold {fold+1} | Accuracy: {acc:.5f}")
    
    print(f"\nAverage CV Accuracy: {np.mean(fold_accuracies):.5f} (+/- {np.std(fold_accuracies):.5f})")

    # 10. Retrain a final model on ALL training data
    print("\nTraining final model on full dataset...")
    # Create a new model without early stopping parameters
    final_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X, y)
    print("Final model trained successfully!")

    # 11. Process the TEST data for submission
    print("\nProcessing TEST data...")
    test_texts_1 = []
    test_texts_2 = []
    test_numeric_ids = []
    valid_test_indices = []

    for numeric_id, article_id in test_id_to_article.items():
        article_dir = os.path.join(test_dir, article_id)
        
        if not os.path.exists(article_dir):
            print(f"Warning: Test directory not found for article {article_id}")
            continue
            
        file1_path = os.path.join(article_dir, 'file_1.txt')
        file2_path = os.path.join(article_dir, 'file_2.txt')
        
        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            print(f"Warning: Files not found for test article {article_id}")
            continue
        
        text1 = read_text_file(file1_path)
        text2 = read_text_file(file2_path)
        
        if text1 and text2 and len(text1) > 50 and len(text2) > 50:
            test_texts_1.append(text1)
            test_texts_2.append(text2)
            test_numeric_ids.append(numeric_id)
            valid_test_indices.append(numeric_id)

    print(f"Found {len(test_texts_1)} valid test pairs out of {len(sample_sub)} total")

    if len(test_texts_1) > 0:
        print(f"Generating embeddings for {len(test_texts_1)} test pairs...")
        test_embeddings_1 = get_embeddings(test_texts_1)
        test_embeddings_2 = get_embeddings(test_texts_2)

        test_features = []
        for emb1, emb2 in zip(test_embeddings_1, test_embeddings_2):
            difference = np.abs(emb1 - emb2)
            product = emb1 * emb2
            combined_features = np.hstack((emb1, emb2, difference, product))
            test_features.append(combined_features)

        X_test = np.array(test_features)
        test_predictions = final_model.predict(X_test)
        
        # Create the submission with the correct format (numeric IDs)
        all_predictions = np.ones(len(sample_sub), dtype=int)
        for numeric_id, pred in zip(test_numeric_ids, test_predictions):
            all_predictions[numeric_id] = pred
            
        submission_df = pd.DataFrame({
            'id': range(len(all_predictions)),
            'real_text_id': all_predictions
        })
    else:
        print("No valid test data found, using default predictions")
        submission_df = pd.DataFrame({
            'id': range(len(sample_sub)),
            'real_text_id': [1] * len(sample_sub)
        })

    # 12. Create the submission file
    submission_df.to_csv('submission.csv', index=False)
    print("\nSubmission file 'submission.csv' created successfully!")

    print("\nSample submission head:")
    print(submission_df.head())
    print(f"\nSubmission shape: {submission_df.shape}")
    print(f"Prediction distribution: {submission_df['real_text_id'].value_counts().to_dict()}")

else:
    print("No training data available, cannot proceed with model training")

