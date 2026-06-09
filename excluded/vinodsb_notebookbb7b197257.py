import pandas as pd
import numpy as np
import re
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

class AdvancedDataReferenceDetector:
    def __init__(self):
        self.vectorizers = {}
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Enhanced data-related patterns and keywords
        self.data_keywords = {
            'data_general': ['data', 'dataset', 'database', 'corpus', 'collection', 'repository', 'archive'],
            'data_types': ['genomic', 'clinical', 'survey', 'experimental', 'observational', 'longitudinal'],
            'data_sources': ['ncbi', 'genbank', 'pubmed', 'who', 'cdc', 'nih', 'nasa', 'noaa', 'eurostat'],
            'data_actions': ['collected', 'generated', 'obtained', 'derived', 'analyzed', 'processed', 'compiled'],
            'data_access': ['publicly', 'openly', 'freely', 'available', 'accessible', 'downloadable'],
            'citations': ['doi', 'url', 'http', 'www', 'reference', 'cite', 'citation'],
            'measurements': ['sample', 'measurement', 'observation', 'record', 'entry', 'variable'],
            'formats': ['csv', 'json', 'xml', 'xlsx', 'sql', 'api', 'ftp']
        }
        
        # Advanced regex patterns for data references
        self.patterns = {
            'data_mention': r'\b(?:data|dataset|database|corpus)\s+(?:from|in|of|available|obtained|used|analyzed)',
            'data_generation': r'\b(?:we|authors|researchers)\s+(?:collected|generated|created|compiled|gathered)\s+(?:data|dataset)',
            'data_reuse': r'\b(?:used|utilized|analyzed|reused|derived)\s+(?:data|dataset)\s+(?:from|available)',
            'public_data': r'\b(?:publicly|openly|freely)\s+available\s+(?:data|dataset)',
            'data_source': r'\b(?:data|dataset)\s+(?:from|obtained|available)\s+(?:at|from)\s+(?:http|www|\w+\.(?:gov|org|edu))',
            'primary_indicators': r'\b(?:collected|generated|created|original|new|novel)\s+(?:data|dataset|measurements)',
            'secondary_indicators': r'\b(?:existing|previous|published|available|public)\s+(?:data|dataset)',
            'citation_pattern': r'\b(?:doi|DOI):\s*[\d\.\-\/]+|\b(?:http|https|www)[\w\.\-\/\:]+',
            'table_figure': r'\b(?:Table|Figure|Fig\.?)\s+\d+',
            'database_names': r'\b(?:GenBank|PubMed|NCBI|WHO|CDC|NIH|NASA|NOAA|Eurostat|OECD)\b',
            'file_formats': r'\b\w+\.(?:csv|json|xml|xlsx|xls|txt|dat|sql)\b',
            'statistical_terms': r'\b(?:sample|population|cohort|survey|study|trial|experiment)\b'
        }
    
    def extract_linguistic_features(self, text):
        """Extract advanced linguistic features"""
        if pd.isna(text):
            return {}
        
        text = str(text)
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Punctuation features
        features['period_count'] = text.count('.')
        features['comma_count'] = text.count(',')
        features['semicolon_count'] = text.count(';')
        features['colon_count'] = text.count(':')
        features['parentheses_count'] = text.count('(') + text.count(')')
        features['brackets_count'] = text.count('[') + text.count(']')
        features['quotes_count'] = text.count('"') + text.count("'")
        
        # Advanced punctuation ratios
        word_count = features['word_count'] if features['word_count'] > 0 else 1
        features['punctuation_ratio'] = (features['period_count'] + features['comma_count']) / word_count
        features['citation_punctuation'] = (features['parentheses_count'] + features['brackets_count']) / word_count
        
        return features
    
    def extract_semantic_features(self, text):
        """Extract semantic and contextual features"""
        if pd.isna(text):
            return {}
        
        text = str(text).lower()
        features = {}
        
        # Keyword category counts
        for category, keywords in self.data_keywords.items():
            count = sum(text.count(keyword) for keyword in keywords)
            features[f'{category}_count'] = count
            features[f'{category}_present'] = 1 if count > 0 else 0
        
        # Pattern matching
        for pattern_name, pattern in self.patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            features[f'pattern_{pattern_name}'] = matches
            features[f'pattern_{pattern_name}_present'] = 1 if matches > 0 else 0
        
        # N-gram features for specific phrases
        bigrams = ['data collection', 'data analysis', 'publicly available', 'primary data', 
                  'secondary data', 'original data', 'existing data', 'survey data']
        for bigram in bigrams:
            features[f'bigram_{bigram.replace(" ", "_")}'] = text.count(bigram)
        
        # Advanced trigrams for context
        trigrams = ['we collected data', 'data was obtained', 'publicly available data',
                   'data from the', 'analysis of data', 'generated new data']
        for trigram in trigrams:
            features[f'trigram_{trigram.replace(" ", "_")}'] = text.count(trigram)
        
        # Verb tense analysis for data usage
        usage_verbs = {
            'past': ['collected', 'generated', 'obtained', 'analyzed', 'used', 'gathered'],
            'present': ['collect', 'generate', 'obtain', 'analyze', 'use', 'gather'],
            'passive': ['was collected', 'were obtained', 'is available', 'are analyzed']
        }
        
        for tense, verbs in usage_verbs.items():
            count = sum(text.count(verb) for verb in verbs)
            features[f'verb_{tense}_count'] = count
        
        # Data source confidence indicators
        confidence_indicators = {
            'high_confidence': ['doi:', 'http://', 'https://', 'ncbi', 'genbank', 'pubmed'],
            'medium_confidence': ['available at', 'obtained from', 'downloaded from'],
            'low_confidence': ['personal communication', 'unpublished data', 'internal data']
        }
        
        for confidence_level, indicators in confidence_indicators.items():
            count = sum(text.count(indicator) for indicator in indicators)
            features[f'confidence_{confidence_level}'] = count
        
        # Section context clues
        section_keywords = {
            'methods': ['methodology', 'methods', 'materials', 'procedure', 'protocol'],
            'results': ['results', 'findings', 'outcomes', 'analysis', 'statistics'],
            'discussion': ['discussion', 'conclusion', 'limitations', 'implications']
        }
        
        for section, keywords in section_keywords.items():
            count = sum(text.count(keyword) for keyword in keywords)
            features[f'section_{section}_context'] = count
        
        return features
    
    def extract_named_entities(self, text):
        """Extract named entity features"""
        if pd.isna(text):
            return {}
        
        features = {}
        try:
            # Tokenize and tag
            tokens = word_tokenize(str(text))
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            
            # Count entity types
            entity_types = {}
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    entity_type = chunk.label()
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            # Common entity types relevant to data references
            relevant_entities = ['ORGANIZATION', 'PERSON', 'GPE', 'DATE', 'CARDINAL']
            for entity in relevant_entities:
                features[f'entity_{entity.lower()}'] = entity_types.get(entity, 0)
            
            features['total_entities'] = sum(entity_types.values())
            features['unique_entity_types'] = len(entity_types)
            
        except Exception as e:
            print(f"Warning: NER extraction failed: {e}")
            for entity in ['organization', 'person', 'gpe', 'date', 'cardinal']:
                features[f'entity_{entity}'] = 0
            features['total_entities'] = 0
            features['unique_entity_types'] = 0
        
        return features
    
    def extract_position_features(self, text, context_window=50):
        """Extract position-based features"""
        if pd.isna(text):
            return {}
        
        text = str(text).lower()
        features = {}
        
        # Position of key terms
        data_terms = ['data', 'dataset', 'database']
        for term in data_terms:
            positions = [m.start() for m in re.finditer(term, text)]
            if positions:
                features[f'{term}_first_position'] = positions[0] / len(text) if len(text) > 0 else 0
                features[f'{term}_last_position'] = positions[-1] / len(text) if len(text) > 0 else 0
                features[f'{term}_count'] = len(positions)
            else:
                features[f'{term}_first_position'] = -1
                features[f'{term}_last_position'] = -1
                features[f'{term}_count'] = 0
        
        # Beginning/end bias
        text_start = text[:context_window] if len(text) > context_window else text
        text_end = text[-context_window:] if len(text) > context_window else text
        
        features['data_in_start'] = 1 if any(term in text_start for term in data_terms) else 0
        features['data_in_end'] = 1 if any(term in text_end for term in data_terms) else 0
        
        return features
    
    def create_tfidf_features(self, texts, fit=True):
        """Create TF-IDF features with multiple configurations"""
        tfidf_configs = {
            'word_1_3': TfidfVectorizer(ngram_range=(1, 3), max_features=5000, stop_words='english', min_df=2),
            'char_3_5': TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=3000, min_df=2),
            'word_unigram': TfidfVectorizer(ngram_range=(1, 1), max_features=3000, stop_words='english', min_df=2)
        }
        
        tfidf_features = {}
        for name, vectorizer in tfidf_configs.items():
            if fit:
                self.vectorizers[name] = vectorizer
                features = vectorizer.fit_transform(texts)
            else:
                features = self.vectorizers[name].transform(texts)
            
            # Convert to DataFrame for easier handling
            feature_names = [f'{name}_{i}' for i in range(features.shape[1])]
            tfidf_features[name] = pd.DataFrame(features.toarray(), columns=feature_names)
        
        return tfidf_features
    
    def prepare_features(self, df, text_column='text', fit=True):
        """Prepare comprehensive feature set"""
        print("Extracting comprehensive features...")
        
        # Extract all feature types
        linguistic_features = []
        semantic_features = []
        entity_features = []
        position_features = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            
            linguistic_features.append(self.extract_linguistic_features(text))
            semantic_features.append(self.extract_semantic_features(text))
            entity_features.append(self.extract_named_entities(text))
            position_features.append(self.extract_position_features(text))
        
        # Convert to DataFrames
        linguistic_df = pd.DataFrame(linguistic_features).fillna(0)
        semantic_df = pd.DataFrame(semantic_features).fillna(0)
        entity_df = pd.DataFrame(entity_features).fillna(0)
        position_df = pd.DataFrame(position_features).fillna(0)
        
        # Create TF-IDF features
        texts = df[text_column].fillna('').astype(str)
        tfidf_features = self.create_tfidf_features(texts, fit=fit)
        
        # Combine all features
        feature_dfs = [linguistic_df, semantic_df, entity_df, position_df]
        feature_dfs.extend(tfidf_features.values())
        
        combined_features = pd.concat(feature_dfs, axis=1)
        
        # Scale numerical features
        if fit:
            combined_features = pd.DataFrame(
                self.scaler.fit_transform(combined_features),
                columns=combined_features.columns
            )
        else:
            combined_features = pd.DataFrame(
                self.scaler.transform(combined_features),
                columns=combined_features.columns
            )
        
        print(f"Created {combined_features.shape[1]} features")
        return combined_features
    
    def create_ensemble_model(self):
        """Create an advanced ensemble of different models"""
        # Individual models with optimized hyperparameters
        models = [
            ('rf', RandomForestClassifier(
                n_estimators=300, 
                max_depth=25, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200, 
                learning_rate=0.1, 
                max_depth=8,
                subsample=0.8,
                random_state=42
            )),
            ('lr', LogisticRegression(
                random_state=42, 
                max_iter=2000, 
                C=1.0,
                class_weight='balanced',
                solver='liblinear'
            )),
            ('svm', SVC(
                probability=True, 
                random_state=42, 
                C=1.0, 
                kernel='rbf',
                class_weight='balanced',
                gamma='scale'
            ))
        ]
        
        # Use soft voting for probability-based ensemble
        ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
        return ensemble
    
    def train(self, X, y):
        """Train the ensemble model"""
        print("Training ensemble model...")
        
        # Create and train ensemble
        self.model = self.create_ensemble_model()
        self.model.fit(X, y)
        
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)

def load_competition_data():
    """Load the actual competition data"""
    print("Loading competition data...")
    
    data_path = Path('/kaggle/input/make-data-count-finding-data-references')
    
    train_df = None
    test_df = None
    sample_sub = None
    
    try:
        if (data_path / 'train.csv').exists():
            train_df = pd.read_csv(data_path / 'train.csv')
            print(f"Training data loaded: {train_df.shape}")
            print(f"Columns: {train_df.columns.tolist()}")
            
            # Check for label distribution
            label_cols = [col for col in train_df.columns if col.lower() in ['label', 'target', 'class', 'category']]
            if label_cols:
                label_col = label_cols[0]
                print(f"\nLabel distribution ({label_col}):")
                print(train_df[label_col].value_counts())
        
        if (data_path / 'test.csv').exists():
            test_df = pd.read_csv(data_path / 'test.csv')
            print(f"Test data loaded: {test_df.shape}")
        
        if (data_path / 'sample_submission.csv').exists():
            sample_sub = pd.read_csv(data_path / 'sample_submission.csv')
            print(f"Sample submission loaded: {sample_sub.shape}")
            
    except Exception as e:
        print(f"Error loading competition data: {e}")
    
    return train_df, test_df, sample_sub

def create_enhanced_synthetic_data():
    """Create more realistic synthetic data for testing"""
    print("Creating enhanced synthetic data...")
    
    # More diverse and realistic examples with edge cases
    texts = [
        # Primary data examples (Class 1)
        "We collected primary survey data from 1,247 participants across five clinical sites between January 2022 and December 2023.",
        "Original experimental data was generated using RNA sequencing protocols on tissue samples from 156 patients.",
        "The study team gathered longitudinal health data through structured interviews conducted every six months.",
        "We created a novel dataset by combining genomic sequencing results with clinical phenotype information.",
        "Primary data collection involved field observations recorded using standardized measurement protocols over 18 months.",
        "The research generated new epidemiological data through a population-based cross-sectional survey (n=2,341).",
        "We established an original cohort dataset by recruiting participants from three metropolitan areas.",
        "Experimental data was produced through controlled laboratory experiments using mass spectrometry analysis.",
        "Our team collected fresh tissue samples and generated proteomic data using LC-MS/MS techniques.",
        "We conducted original behavioral experiments and recorded response times for 500 participants.",
        
        # Secondary data examples (Class 2)
        "The analysis utilized publicly available genomic data from the NCBI GenBank database (accession numbers provided).",
        "We obtained climate data from NOAA weather stations covering the period 1990-2020 (DOI: 10.5194/data-2021).",
        "The study reused data from the European Social Survey Wave 9, freely available at www.europeansocialsurvey.org.",
        "Secondary analysis was performed on existing cohort data from the Framingham Heart Study database.",
        "We analyzed publicly accessible COVID-19 surveillance data from the WHO Global Health Observatory.",
        "The research utilized open-source financial data downloaded from the World Bank Open Data portal.",
        "Data was obtained from the Cancer Genome Atlas (TCGA) project via the Genomic Data Commons API.",
        "We reused demographic data from the U.S. Census Bureau's American Community Survey (2015-2019).",
        "The analysis incorporated existing satellite imagery data from NASA's Earth Observing System Data.",
        "We downloaded and analyzed historical stock market data from Yahoo Finance API (2010-2023).",
        
        # No data reference examples (Class 0)
        "This systematic review synthesizes findings from 47 published studies on cardiovascular disease prevention.",
        "The theoretical framework builds upon established principles of cognitive behavioral therapy.",
        "Our methodology follows standard protocols described in the literature for qualitative research design.",
        "The discussion section addresses limitations and implications of the current research findings.",
        "This paper presents a comprehensive review of machine learning applications in healthcare settings.",
        "The introduction provides background context on recent developments in renewable energy policy.",
        "Results demonstrate significant improvements in computational efficiency compared to baseline methods.",
        "The conclusion summarizes key contributions and suggests directions for future research endeavors.",
        "We propose a novel algorithm for optimizing neural network architectures in deep learning.",
        "The literature review identified gaps in current understanding of protein folding mechanisms."
    ]
    
    # Labels: 0 = no data reference, 1 = primary data, 2 = secondary data
    labels = ([1] * 10 +    # Primary data (10 examples)
             [2] * 10 +     # Secondary data (10 examples)
             [0] * 10)      # No data reference (10 examples)
    
    # Create training data
    train_df = pd.DataFrame({
        'id': range(len(texts)),
        'text': texts,
        'label': labels
    })
    
    # Create test data with challenging examples
    test_texts = [
        "We conducted original field studies to collect biodiversity data from pristine forest ecosystems.",
        "The analysis incorporated existing satellite imagery data from NASA's Earth Observing System.",
        "This meta-analysis reviews 23 randomized controlled trials published between 2010-2023.",
        "Primary survey data was gathered from healthcare workers during the pandemic response period.",
        "We utilized open-access genetic variant data from the 1000 Genomes Project database.",
        "The study methodology adheres to CONSORT guidelines for reporting clinical trial results.",
        "Original behavioral data was collected through controlled experiments with college student volunteers.",
        "We analyzed historical economic indicators obtained from the Federal Reserve Economic Database.",
        "Data collection procedures followed IRB-approved protocols for human subjects research.",
        "The research team developed new algorithms for processing large-scale genomic datasets.",
        "We accessed publicly available climate data through the Global Climate Data Portal.",
        "This review synthesizes evidence from systematic reviews and meta-analyses in the field."
    ]
    
    test_df = pd.DataFrame({
        'id': range(len(texts), len(texts) + len(test_texts)),
        'text': test_texts
    })
    
    sample_sub = pd.DataFrame({
        'id': test_df['id'],
        'prediction': [0] * len(test_df)
    })
    
    return train_df, test_df, sample_sub

def evaluate_model(detector, X_train, y_train, cv_folds=5):
    """Perform cross-validation evaluation"""
    print("\nPerforming cross-validation...")
    
    # Determine appropriate CV strategy
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = np.min(class_counts)
    n_splits = min(cv_folds, min_class_count, len(y_train) // 2)
    
    print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    print(f"Using {n_splits} folds for cross-validation")
    
    if n_splits < 2:
        print("Insufficient data for cross-validation")
        return None
    
    # Choose CV strategy
    if len(unique_classes) > 1:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_iterator = cv.split(X_train, y_train)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_iterator = cv.split(X_train)
    
    cv_scores = []
    cv_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_iterator):
        print(f"Fold {fold + 1}/{n_splits}")
        
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Create and train fold model
        fold_detector = AdvancedDataReferenceDetector()
        fold_detector.vectorizers = detector.vectorizers
        fold_detector.scaler = detector.scaler
        fold_detector.label_encoder = detector.label_encoder
        fold_detector.train(X_fold_train, y_fold_train)
        
        # Predict and evaluate
        y_pred = fold_detector.predict(X_fold_val)
        f1 = f1_score(y_fold_val, y_pred, average='weighted')
        acc = accuracy_score(y_fold_val, y_pred)
        
        cv_scores.append(f1)
        cv_accuracies.append(acc)
        print(f"  F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")
    
    print(f"\nCross-validation Results:")
    print(f"F1 Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
    print(f"Accuracy: {np.mean(cv_accuracies):.4f} (+/- {np.std(cv_accuracies)*2:.4f})")
    
    return cv_scores

def main():
    """Main execution function"""
    print("Advanced Make Data Count Solution")
    print("=" * 50)
    
    # Load data
    train_df, test_df, sample_sub = load_competition_data()
    
    if train_df is None or test_df is None:
        print("Competition data not found. Using synthetic data for demonstration...")
        train_df, test_df, sample_sub = create_enhanced_synthetic_data()
    
    # Determine column names
    text_column = 'text'
    id_column = 'id'
    
    # Find label column
    label_column = None
    for col in ['label', 'target', 'class', 'category']:
        if col in train_df.columns:
            label_column = col
            break
    
    if label_column is None:
        print("Warning: No label column found. Using 'label'.")
        label_column = 'label'
    
    print(f"\nUsing columns: text='{text_column}', label='{label_column}', id='{id_column}'")
    
    # Initialize detector
    detector = AdvancedDataReferenceDetector()
    
    # Prepare features
    print("\nPreparing training features...")
    X_train = detector.prepare_features(train_df, text_column, fit=True)
    
    # Encode labels
    y_train = detector.label_encoder.fit_transform(train_df[label_column])
    
    # Cross-validation evaluation
    cv_scores = evaluate_model(detector, X_train, y_train)
    
    # Train final model
    print("\nTraining final model on full dataset...")
    detector.train(X_train, y_train)
    
    # Prepare test features
    print("\nPreparing test features...")
    X_test = detector.prepare_features(test_df, text_column, fit=False)
    
    # Make predictions
    print("Making predictions...")
    predictions = detector.predict(X_test)
    prediction_probs = detector.predict_proba(X_test)
    
    # Decode predictions
    decoded_predictions = detector.label_encoder.inverse_transform(predictions)
    
    # Create submission
    print("Creating submission...")
    submission = test_df[[id_column]].copy()
    
    # Determine prediction column name from sample submission
    pred_column = 'prediction'
    if sample_sub is not None and len(sample_sub.columns) > 1:
        pred_column = sample_sub.columns[1]
    
    submission[pred_column] = decoded_predictions
    
    # Add confidence scores
    max_probs = np.max(prediction_probs, axis=1)
    submission['confidence'] = max_probs
    
    print(f"\nSubmission shape: {submission.shape}")
    print("Sample predictions:")
    print(submission.head(10))
    
    print(f"\nPrediction distribution:")
    print(pd.Series(decoded_predictions).value_counts())
    
    # Save submission
    submission_file = 'advanced_submission.csv'
    submission[[id_column, pred_column]].to_csv(submission_file, index=False)
    print(f"\nSubmission saved to '{submission_file}'")
    
    # Save detailed results
    detailed_results = submission.copy()
    detailed_results['text_preview'] = test_df[text_column].str[:100] + '...'
    detailed_results.to_csv('detailed_predictions.csv', index=False)
    
    # Save model
    print("Saving model...")
    model_data = {
        'detector': detector,
        'text_column': text_column,
        'label_column': label_column,
        'cv_scores': cv_scores
    }
    
    with open('advanced_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved to 'advanced_model.pkl'")
    print("\nExecution completed successfully!")

if __name__ == "__main__":
    main()

