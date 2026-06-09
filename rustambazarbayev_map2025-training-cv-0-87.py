# training.py
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import gc
import joblib
from typing import Dict, List, Tuple
from tqdm import tqdm
from fractions import Fraction
import re
import warnings
warnings.filterwarnings('ignore')

# ============= CONFIGURATION =============
CONFIG = {
    'model_paths': [
        "/kaggle/input/map-exp-14-full/MAP_EXP_14_FULL",  # DeBERTa-like
        "/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL",  # Qwen
        "/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL"  # DeepSeekMath
    ],
    'max_length': 256,
    'batch_size': 1,  # Conservative for large models
    'pooling_strategy': 'mean',
    'use_solution_features': True,
    'use_question_solving': True,
    'meta_learner': 'xgboost',
    'xgb_params': {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
    },
    'n_folds': 5,
    'random_seed': 42,
    'early_stopping_rounds': 50,
}

# ============= QUESTION SOLUTIONS =============
class QuestionSolutions:
    def __init__(self):
        self.solutions = {
            31772: {'answer': '1/3', 'solution': 'Total triangles: 9. Shaded: 6. Unshaded: 9-6=3. Fraction: 3/9 = 1/3', 'concepts': ['fractions', 'simplification', 'visual_counting'], 'common_errors': ['not_simplifying', 'counting_shaded', 'wrong_total']},
            31774: {'answer': '1/12', 'solution': '1/2 ÷ 6 = 1/2 × 1/6 = 1/12. Division by whole number = multiply by reciprocal.', 'concepts': ['fraction_division', 'reciprocal', 'multiplication'], 'common_errors': ['divide_denominator_only', 'multiply_by_6']},
            31777: {'answer': '72', 'solution': '120 counters × 3/5 = (120 × 3)/5 = 360/5 = 72 red counters', 'concepts': ['fraction_of_amount', 'multiplication', 'division'], 'common_errors': ['divide_instead_multiply', 'wrong_operation']},
            31778: {'answer': '6', 'solution': 'Cross multiply: A×15 = 9×10, 15A = 90, A = 6', 'concepts': ['equivalent_fractions', 'cross_multiplication', 'equations'], 'common_errors': ['add_instead_multiply', 'wrong_cross_multiply']},
            32829: {'answer': '12', 'solution': '2y = 24, divide both sides by 2, y = 12', 'concepts': ['linear_equations', 'inverse_operations', 'division'], 'common_errors': ['subtract_2', 'wrong_inverse']},
            32833: {'answer': '10/3', 'solution': '2/3 × 5 = (2×5)/3 = 10/3 = 3⅓', 'concepts': ['fraction_multiplication', 'mixed_numbers'], 'common_errors': ['multiply_denominator', 'add_instead']},
            32835: {'answer': 'largest_value', 'solution': 'Compare all numbers considering place value and decimal positions', 'concepts': ['number_comparison', 'place_value', 'ordering'], 'common_errors': ['ignore_decimals', 'compare_wrong_place']},
            33471: {'answer': '15', 'solution': 'Yellow: 24 × 3/8 = 9. Green: 24 - 9 = 15', 'concepts': ['fraction_of_amount', 'subtraction', 'complement'], 'common_errors': ['find_yellow_only', 'wrong_fraction']},
            33472: {'answer': '11/15', 'solution': 'LCD(3,5)=15. 1/3=5/15, 2/5=6/15. 5/15+6/15=11/15', 'concepts': ['fraction_addition', 'common_denominator', 'LCD'], 'common_errors': ['add_denominators', 'no_common_denominator']},
            33474: {'answer': '2/3 × 1/3', 'solution': 'Robert eats 1/3 of Sally\'s 2/3. This is 1/3 OF 2/3 = multiply', 'concepts': ['fraction_of_fraction', 'multiplication', 'word_problems'], 'common_errors': ['use_addition', 'use_division']},
            76870: {'answer': '10', 'solution': 'Interior angle = 144°. Exterior = 180-144 = 36°. Sides = 360/36 = 10', 'concepts': ['polygons', 'interior_angles', 'exterior_angles'], 'common_errors': ['use_interior_directly', 'wrong_formula']},
            89443: {'answer': '-3', 'solution': '(-8)-(-5) = (-8)+5 = -3. Subtracting negative = adding positive.', 'concepts': ['negative_numbers', 'integer_operations', 'subtraction'], 'common_errors': ['subtract_both', 'wrong_sign']},
            91695: {'answer': '26', 'solution': 'Pattern: 6,10,14,18... (+4 each). Rule: 4n+2. Pattern 6: 4(6)+2=26', 'concepts': ['sequences', 'linear_patterns', 'nth_term'], 'common_errors': ['wrong_difference', 'wrong_formula']},
            104665: {'answer': '48', 'solution': 'Total work: 3×192=576 person-hours. 12 people: 576÷12=48 hours', 'concepts': ['inverse_proportion', 'rates', 'work_problems'], 'common_errors': ['direct_proportion', 'multiply_instead']},
            109465: {'answer': 'Very likely', 'solution': 'P=0.9 is close to 1 (certain). Best description: Very likely/Almost certain', 'concepts': ['probability', 'likelihood_scale', 'interpretation'], 'common_errors': ['misread_scale', 'wrong_description']}
        }
    
    def get_solution(self, question_id: int) -> Dict:
        return self.solutions.get(question_id, {})
    
    def get_correct_answer(self, question_id: int) -> str:
        solution = self.get_solution(question_id)
        return solution.get('answer', '')

# ============= SOLUTION-AWARE PROMPTS =============
class SolutionPrompts:
    def __init__(self, solutions_db):
        self.solutions_db = solutions_db
    
    def standard_prompt(self, row):
        correctness = "CORRECT" if row['is_correct'] else "INCORRECT"
        return f"""Question: {row['QuestionText']}
Answer: {row['MC_Answer']} ({correctness})
Student Explanation: {row['StudentExplanation']}
Identify misconception:"""
    
    def solution_aware_prompt(self, row):
        solution = self.solutions_db.get_solution(row['QuestionId'])
        correctness = "CORRECT" if row['is_correct'] else "INCORRECT"
        
        if solution:
            concepts = ', '.join(solution.get('concepts', []))
            correct_answer = solution.get('answer', '')
            
            return f"""Mathematical Problem Analysis

Question: {row['QuestionText']}
Key Concepts: {concepts}
Correct Answer: {correct_answer}

Student's Answer: {row['MC_Answer']} (Status: {correctness})
Student's Explanation: {row['StudentExplanation']}

Task: Identify the specific mathematical misconception or error pattern."""
        else:
            return self.standard_prompt(row)

# ============= SOLUTION FEATURE EXTRACTOR =============
class SolutionFeatureExtractor:
    def __init__(self, solutions_db):
        self.solutions_db = solutions_db
    
    def extract_numbers(self, text: str) -> List[float]:
        pattern = r'-?\d+\.?\d*|(?:\d+/\d+)'
        matches = re.findall(pattern, str(text))
        numbers = []
        for match in matches:
            try:
                if '/' in match:
                    numbers.append(float(Fraction(match)))
                else:
                    numbers.append(float(match))
            except:
                pass
        return numbers
    
    def keyword_match_score(self, text: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0
        text_lower = str(text).lower()
        matches = sum(1 for kw in keywords if kw.replace('_', ' ') in text_lower)
        return matches / len(keywords)
    
    def answer_matches_correct(self, student_answer: str, correct_answer: str) -> bool:
        student_nums = self.extract_numbers(student_answer)
        correct_nums = self.extract_numbers(correct_answer)
        
        if not student_nums or not correct_nums:
            return str(student_answer).strip().lower() == str(correct_answer).strip().lower()
        
        for snum in student_nums:
            for cnum in correct_nums:
                if abs(snum - cnum) < 0.01:
                    return True
        return False
    
    def extract_features(self, row: pd.Series) -> np.ndarray:
        question_id = row['QuestionId']
        solution = self.solutions_db.get_solution(question_id)
        
        features = []
        features.append(float(row.get('is_correct', 0)))
        features.append(len(str(row['StudentExplanation'])) / 200.0)
        
        if solution:
            correct_answer = solution.get('answer', '')
            features.append(float(self.answer_matches_correct(row['MC_Answer'], correct_answer)))
            
            concepts = solution.get('concepts', [])
            features.append(self.keyword_match_score(row['StudentExplanation'], concepts))
            
            errors = solution.get('common_errors', [])
            features.append(self.keyword_match_score(row['StudentExplanation'], errors))
            
            solution_text = solution.get('solution', '')
            solution_words = set(re.findall(r'\b\w{4,}\b', solution_text.lower()))
            explanation_words = set(re.findall(r'\b\w{4,}\b', str(row['StudentExplanation']).lower()))
            
            if solution_words:
                overlap = len(solution_words & explanation_words) / len(solution_words)
                features.append(overlap)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        nums_in_explanation = self.extract_numbers(row['StudentExplanation'])
        features.append(min(len(nums_in_explanation) / 5.0, 1.0))
        
        operations = ['add', 'subtract', 'multiply', 'divide', 'times', 'plus', 'minus']
        features.append(self.keyword_match_score(row['StudentExplanation'], operations))
        
        features.append(float(question_id % 1000) / 1000.0)
        
        return np.array(features)
    
    def extract_batch(self, df: pd.DataFrame) -> np.ndarray:
        features = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting solution features"):
            features.append(self.extract_features(row))
        return np.vstack(features)

# ============= EMBEDDING EXTRACTOR =============
class EmbeddingExtractor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
    
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
    
    def pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    
    def extract(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            self.load()
        
        embeddings = []
        for i in tqdm(range(0, len(texts), CONFIG['batch_size'])):
            batch = texts[i:i+CONFIG['batch_size']]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=CONFIG['max_length'],
                return_tensors='pt'
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state
                emb = self.pool(hidden, inputs['attention_mask'])
                embeddings.append(emb.cpu().numpy())
            
            del inputs, outputs, hidden
            torch.cuda.empty_cache()
        
        return np.vstack(embeddings)
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

def smart_deduplication(df):
    df_sorted = df.sort_values(['text', 'Misconception'], key=lambda x: x.replace('NA', 'zzz') if x.name == 'Misconception' else x)
    return df_sorted.drop_duplicates(subset=['text'], keep='first')

def main():
    train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
    
    le = LabelEncoder()
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category + ':' + train.Misconception
    train['label'] = le.fit_transform(train['target'])
    original_le = le
    original_n_classes = len(le.classes_)
    
    idx = train.Category.str.startswith('True', na=False)
    correct = train.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
    correct = correct.sort_values('c', ascending=False).drop_duplicates(['QuestionId'])
    correct = correct[['QuestionId', 'MC_Answer']]
    correct['is_correct'] = 1
    
    train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)
    
    class_counts = train['label'].value_counts()
    good_classes = class_counts[class_counts >= CONFIG['n_folds']].index
    original_good_labels = sorted(good_classes.tolist())
    
    train = train[train['label'].isin(good_classes)].reset_index(drop=True)
    
    label_map = {old: new for new, old in enumerate(original_good_labels)}
    train['label'] = train['label'].map(label_map)
    n_classes = len(original_good_labels)
    
    joblib.dump({'le': original_le, 'original_good_labels': original_good_labels, 'label_map': label_map}, 'label_info.pkl')
    
    solutions_db = QuestionSolutions()
    prompt_creator = SolutionPrompts(solutions_db)
    train['text'] = train.apply(prompt_creator.solution_aware_prompt if CONFIG['use_question_solving'] else prompt_creator.standard_prompt, axis=1)
    
    train = smart_deduplication(train)
    
    embeddings_list = []
    for model_path in CONFIG['model_paths']:
        extractor = EmbeddingExtractor(model_path)
        embeddings = extractor.extract(train['text'].tolist())
        embeddings_list.append(embeddings)
        extractor.cleanup()
    
    train_embeddings = np.hstack(embeddings_list)
    
    if CONFIG['use_solution_features']:
        feature_extractor = SolutionFeatureExtractor(solutions_db)
        train_solution_features = feature_extractor.extract_batch(train)
        X_train = np.hstack([train_embeddings, train_solution_features])
    else:
        X_train = train_embeddings
    
    y_train = train['label'].values
    
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_seed'])
    
    oof_preds = np.zeros((len(train), n_classes))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train[tr_idx]
        y_tr = y_train[tr_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        model = xgb.XGBClassifier(
            **CONFIG['xgb_params'],
            objective='multi:softprob',
            num_class=n_classes,
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            random_state=CONFIG['random_seed']
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=CONFIG['early_stopping_rounds'],
            verbose=False
        )
        
        val_preds = model.predict_proba(X_val)
        oof_preds[val_idx] = val_preds
    
    cv_map3 = np.mean(np.any(np.argsort(-oof_preds, axis=1)[:, :3] == y_train[:, None], axis=1))
    print(f"OVERALL CV MAP@3: {cv_map3:.6f}")
    
    final_model = xgb.XGBClassifier(
        **CONFIG['xgb_params'],
        objective='multi:softprob',
        num_class=n_classes,
        tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
        random_state=CONFIG['random_seed']
    )
    
    final_model.fit(X_train, y_train)
    final_model.save_model('best_model.json')
    
    print("Training complete")

if __name__ == "__main__":
    main()

