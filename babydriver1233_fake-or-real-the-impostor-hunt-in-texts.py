# COMPLETE CODE FOR FAKE/REAL TEXT CLASSIFICATION COMPETITION
import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

# Set paths
TRAIN_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
TEST_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
TRAIN_CSV = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"

# ==================== DATA LOADING FUNCTIONS ====================
def load_data_corrected(train_dir, train_csv):
    """Load data with corrected directory structure handling"""
    
    # Load ground truth
    df_truth = pd.read_csv(train_csv)
    print(f"Ground truth shape: {df_truth.shape}")
    
    # Check what's actually in the train directory
    train_contents = os.listdir(train_dir)
    print(f"Train directory contents: {train_contents[:10]}...")
    
    # Create mapping from article_XXXX to numerical ID
    article_to_id = {}
    for folder in train_contents:
        if folder.startswith('article_'):
            try:
                numerical_id = int(folder.split('_')[1])
                article_to_id[folder] = numerical_id
            except:
                continue
    
    print(f"Found {len(article_to_id)} article folders with numerical IDs")
    
    train_data = []
    successful = 0
    
    for folder_name, numerical_id in article_to_id.items():
        if numerical_id in df_truth['id'].values:
            real_text_id = df_truth[df_truth['id'] == numerical_id]['real_text_id'].values[0]
            
            article_path = os.path.join(train_dir, folder_name)
            file1_path = os.path.join(article_path, "file_1.txt")
            file2_path = os.path.join(article_path, "file_2.txt")
            
            if os.path.exists(file1_path) and os.path.exists(file2_path):
                try:
                    with open(file1_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text1 = f.read().strip()
                    with open(file2_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text2 = f.read().strip()
                    
                    train_data.append({
                        'article_id': numerical_id,
                        'folder_name': folder_name,
                        'text1': text1,
                        'text2': text2,
                        'real_text_id': real_text_id
                    })
                    successful += 1
                except Exception as e:
                    print(f"Error reading files for {folder_name}: {e}")
    
    print(f"Successfully loaded {successful} out of {len(df_truth)} articles")
    return pd.DataFrame(train_data), df_truth

def load_test_data_corrected(test_dir):
    """Load test data with corrected directory handling"""
    test_data = []
    test_ids = []
    
    test_contents = os.listdir(test_dir)
    print(f"Test directory contents: {test_contents[:10]}...")
    
    for folder_name in test_contents:
        if folder_name.startswith('article_'):
            article_path = os.path.join(test_dir, folder_name)
            if os.path.isdir(article_path):
                file1_path = os.path.join(article_path, "file_1.txt")
                file2_path = os.path.join(article_path, "file_2.txt")
                
                if os.path.exists(file1_path) and os.path.exists(file2_path):
                    try:
                        with open(file1_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text1 = f.read().strip()
                        with open(file2_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text2 = f.read().strip()
                        
                        test_data.append({
                            'article_id': folder_name,
                            'text1': text1,
                            'text2': text2
                        })
                        test_ids.append(folder_name)
                    except Exception as e:
                        print(f"Error loading test article {folder_name}: {e}")
    
    print(f"Loaded {len(test_data)} test samples")
    return pd.DataFrame(test_data), test_ids

# ==================== FEATURE EXTRACTION ====================
def extract_text_features(text):
    """Extract various linguistic features from text"""
    features = {}
    
    # Basic statistics
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['sentence_count'] = len(re.split(r'[.!?]+', text))
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
    
    # Lexical features
    words = text.lower().split()
    features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
    
    # Special characters
    features['digit_count'] = sum(1 for char in text if char.isdigit())
    features['punctuation_count'] = sum(1 for char in text if char in '.,!?;:')
    
    return features

# ==================== BERT PREDICTION FUNCTION ====================
def predict_with_bert(model, tokenizer, texts, batch_size=16):
    """Predict using the trained BERT model"""
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='tf'
        )
        
        batch_preds = model.predict([encodings['input_ids'], encodings['attention_mask']], verbose=0)
        predictions.extend(batch_preds.flatten())
    
    return np.array(predictions)

# ==================== MAIN EXECUTION ====================
def main():
    print("=== FAKE/REAL TEXT CLASSIFICATION COMPETITION SOLUTION ===")
    
    # Load data
    print("\n1. Loading training data...")
    train_df, truth_df = load_data_corrected(TRAIN_DIR, TRAIN_CSV)
    
    print("\n2. Loading test data...")
    test_df, test_ids = load_test_data_corrected(TEST_DIR)
    
    print(f"\nTraining data shape: {train_df.shape}")
    if len(train_df) > 0:
        print(f"Sample text lengths: {len(train_df.iloc[0]['text1'])}, {len(train_df.iloc[0]['text2'])}")
    
    # Feature extraction
    if len(train_df) > 0:
        print("\n3. Extracting features...")
        
        features_list = []
        labels = []
        
        for idx, row in train_df.iterrows():
            feat1 = extract_text_features(row['text1'])
            feat1['text_id'] = 1
            feat2 = extract_text_features(row['text2'])
            feat2['text_id'] = 2
            
            features_list.extend([feat1, feat2])
            
            if row['real_text_id'] == 1:
                labels.extend([1, 0])
            else:
                labels.extend([0, 1])
        
        feature_df = pd.DataFrame(features_list)
        labels = np.array(labels)
        
        print(f"Feature matrix shape: {feature_df.shape}")
        print(f"Labels shape: {labels.shape}")
    
    # Train Random Forest
    if len(train_df) > 0:
        print("\n4. Training Random Forest model...")
        
        feature_columns = ['char_count', 'word_count', 'sentence_count', 
                          'avg_word_length', 'avg_sentence_length',
                          'unique_word_ratio', 'digit_count', 'punctuation_count']
        
        X = feature_df[feature_columns].values
        y = labels
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Random Forest accuracy: {accuracy:.4f}")
    
    # Train BERT model
    if len(train_df) > 0:
        print("\n5. Training BERT model...")
        
        all_texts = []
        all_labels = []
        
        for idx, row in train_df.iterrows():
            all_texts.extend([row['text1'], row['text2']])
            if row['real_text_id'] == 1:
                all_labels.extend([1, 0])
            else:
                all_labels.extend([0, 1])
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        encodings = tokenizer(
            all_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='tf'
        )
        
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            },
            all_labels
        )).shuffle(1000).batch(16)
        
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        
        input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name='attention_mask')
        
        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        
        x = tf.keras.layers.Dropout(0.3)(pooled_output)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(dataset, epochs=2, verbose=1)
        
        # Evaluate BERT
        bert_preds = predict_with_bert(model, tokenizer, all_texts)
        bert_preds_binary = (bert_preds > 0.5).astype(int)
        bert_accuracy = accuracy_score(all_labels, bert_preds_binary)
        print(f"BERT model accuracy: {bert_accuracy:.4f}")
    
    # Create submission
    print("\n6. Creating final submission...")
    
    submission_data = []
    
    for idx, row in test_df.iterrows():
        article_id = row['article_id']
        numerical_id = article_id.split('_')[1]
        text1, text2 = row['text1'], row['text2']
        
        # BERT predictions
        bert_probs = predict_with_bert(model, tokenizer, [text1, text2])
        bert_pred1, bert_pred2 = bert_probs[0], bert_probs[1]
        
        # Feature-based predictions
        features1 = extract_text_features(text1)
        features2 = extract_text_features(text2)
        feat_vec1 = np.array([features1[col] for col in feature_columns]).reshape(1, -1)
        feat_vec2 = np.array([features2[col] for col in feature_columns]).reshape(1, -1)
        feat_vec1_scaled = scaler.transform(feat_vec1)
        feat_vec2_scaled = scaler.transform(feat_vec2)
        rf_prob1 = rf_model.predict_proba(feat_vec1_scaled)[0][1]
        rf_prob2 = rf_model.predict_proba(feat_vec2_scaled)[0][1]
        
        # Ensemble prediction (weight BERT more heavily)
        final_prob1 = 0.7 * bert_pred1 + 0.3 * rf_prob1
        final_prob2 = 0.7 * bert_pred2 + 0.3 * rf_prob2
        
        predicted_real = 1 if final_prob1 > final_prob2 else 2
        submission_data.append({'id': numerical_id, 'real_text_id': predicted_real})
    
    # Create final submission
    final_submission = pd.DataFrame(submission_data)
    final_submission.to_csv('/kaggle/working/submission.csv', index=False)
    
    print(f"✓ Submission created with {len(final_submission)} predictions")
    print("✓ First few predictions:")
    print(final_submission.head())
    
    # Save models
    print("\n7. Saving trained models...")
    model.save('/kaggle/working/bert_model')
    joblib.dump(rf_model, '/kaggle/working/random_forest_model.joblib')
    joblib.dump(scaler, '/kaggle/working/scaler.joblib')
    print("✓ All models saved successfully!")
    
    # Analysis
    print("\n8. Analyzing data patterns...")
    real_texts, fake_texts = [], []
    
    for idx, row in train_df.iterrows():
        if row['real_text_id'] == 1:
            real_texts.append(row['text1'])
            fake_texts.append(row['text2'])
        else:
            real_texts.append(row['text2'])
            fake_texts.append(row['text1'])
    
    def calculate_avg_features(texts):
        features = {'length': [], 'word_count': [], 'sentence_count': [],
                   'avg_word_length': [], 'unique_ratio': []}
        for text in texts:
            feat = extract_text_features(text)
            features['length'].append(feat['char_count'])
            features['word_count'].append(feat['word_count'])
            features['sentence_count'].append(feat['sentence_count'])
            features['avg_word_length'].append(feat['avg_word_length'])
            features['unique_ratio'].append(feat['unique_word_ratio'])
        return {k: np.mean(v) for k, v in features.items()}
    
    real_avg = calculate_avg_features(real_texts)
    fake_avg = calculate_avg_features(fake_texts)
    
    print("Average characteristics:")
    print("Real texts:", real_avg)
    print("Fake texts:", fake_avg)
    
    print("\n=== SOLUTION COMPLETE ===")
    print(f"Training samples: {len(train_df)}")
    print(f"Test predictions: {len(final_submission)}")
    print("Final submission: /kaggle/working/submission.csv")

# Run the complete solution
if __name__ == "__main__":
    main()




