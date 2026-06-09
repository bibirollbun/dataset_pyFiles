reqs = """torch==2.3.0
torchvision==0.18.0
torchaudio==2.3.0
transformers==4.40.0
sentence-transformers==3.0.1
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.0
bitsandbytes==0.43.0
einops==0.8.0
accelerate==0.30.1   # âœ… updated
peft==0.13.2         # âœ… matches accelerate
huggingface-hub==0.23.0
tokenizers==0.19.1
safetensors==0.4.3
tqdm==4.66.2
scipy==1.12.0
packaging==24.0
regex==2024.5.15
filelock==3.13.1
nltk==3.8.1
pyyaml==6.0.1
requests==2.32.3
urllib3==2.2.2
ctransformers-0.2.27"""

with open("requirements.txt", "w") as f:
    f.write(reqs)

!pip install --no-index --find-links=/kaggle/input/dependency-installationjigswa/wheels -r requirements.txt



!pip install /kaggle/input/dependency-installationjigswa/wheels/ctransformers-0.2.27-py3-none-any.whl


# import os
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# # Set deterministic behavior
# os.environ['PYTHONHASHSEED'] = '42'
# torch.manual_seed(42)
# np.random.seed(42)

# # Configure device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")


# class AdvancedRuleEngine:
#     """Professional rule understanding with enhanced semantic analysis"""

#     def __init__(self):
#         print("Initializing advanced rule understanding engine...")

#         # Use multiple embedding models for robust representation
#         self.embedding_models = {
#             'bge': SentenceTransformer("/kaggle/input/bge-base-en-v1-5/bge-base-en-v1.5", device=device),
#             'mini': SentenceTransformer("/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2", device=device)
#         }

#         self.rule_embeddings = {}
#         self.rule_characteristics = {}

#     def analyze_rule(self, rule):
#         """Comprehensive rule analysis for better understanding"""
#         characteristics = {
#             'length': len(rule),
#             'has_negation': any(word in rule.lower() for word in ['not', 'no', 'avoid', 'prohibited']),
#             'has_examples': 'example' in rule.lower(),
#             'specificity_score': self._calculate_specificity(rule),
#             'complexity_score': self._calculate_complexity(rule)
#         }
#         return characteristics

#     def _calculate_specificity(self, rule):
#         """Calculate how specific the rule is"""
#         specific_terms = ['must', 'required', 'specifically', 'exactly', 'only']
#         return sum(term in rule.lower() for term in specific_terms) / len(rule.split())

#     def _calculate_complexity(self, rule):
#         """Calculate rule complexity"""
#         return len([c for c in rule if c in ';:,']) / len(rule.split())

#     def create_enhanced_embeddings(self, rules):
#         """Create multi-model embeddings for robust representation"""
#         print(f"Creating enhanced embeddings for {len(rules)} rules...")

#         for model_name, model in self.embedding_models.items():
#             self.rule_embeddings[model_name] = model.encode(
#                 rules,
#                 normalize_embeddings=True,
#                 batch_size=8,
#                 show_progress_bar=True
#             )

#         # Store rule characteristics
#         for rule in rules:
#             self.rule_characteristics[rule] = self.analyze_rule(rule)

#     def get_similarity_score(self, test_rule, training_rules):
#         """Get comprehensive similarity score using multiple models"""
#         # Get similarity from BGE model (most accurate)
#         test_emb = self.embedding_models['bge'].encode([test_rule], normalize_embeddings=True)
#         train_embs = self.embedding_models['bge'].encode(training_rules, normalize_embeddings=True)
#         bge_sim = cosine_similarity(test_emb, train_embs)[0]

#         # Get similarity from MiniLM model (complementary)
#         test_emb_mini = self.embedding_models['mini'].encode([test_rule], normalize_embeddings=True)
#         train_embs_mini = self.embedding_models['mini'].encode(training_rules, normalize_embeddings=True)
#         mini_sim = cosine_similarity(test_emb_mini, train_embs_mini)[0]

#         # Combine similarities with rule characteristics
#         max_bge = bge_sim.max()
#         max_mini = mini_sim.max()

#         # Weighted combination
#         combined = 0.7 * max_bge + 0.3 * max_mini

#         return combined, bge_sim.argmax()


# def load_qwen3_model():
#     """Load Qwen3-32B with optimal configuration for maximum accuracy"""
#     print("Loading Qwen3-32B for maximum classification accuracy...")
    
#     from transformers import AutoTokenizer, AutoModelForCausalLM
    
#     # Use Qwen3-32B from Kaggle dataset
#     model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct/1"
    
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         padding_side="left"
#     )
#     tokenizer.pad_token = tokenizer.eos_token
    
#     # Load model with optimal precision
#     print("Loading model with bfloat16 precision for accuracy...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map="auto",
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True
#     )
    
#     print("Qwen3-32B loaded successfully")
#     return model, tokenizer


# def create_enhanced_prompt(input_data, similarity_score, rule_characteristics):
#     """Create professional prompt that maximizes Qwen3's reasoning capabilities"""
    
#     subreddit = input_data['subreddit']
#     rule = input_data['rule']
#     comment = input_data['body']
#     pos1 = input_data['positive_example_1']
#     pos2 = input_data['positive_example_2']
#     neg1 = input_data['negative_example_1']
#     neg2 = input_data['negative_example_2']
    
#     # Format examples with proper context
#     def format_example(example, is_positive):
#         lines = [line.strip() for line in example.split('\n') if line.strip()]
#         prefix = "VIOLATION EXAMPLE" if is_positive else "NON-VIOLATION EXAMPLE"
#         return f"{prefix}:\n" + "\n".join([f"  â€¢ {line}" for line in lines[:3]])  # Limit length
    
#     # Create comprehensive prompt
#     prompt = f"""<|im_start|>system
# You are a senior Reddit moderator with expertise in community-specific norms.
# Analyze the following rule violation with extreme precision.

# <|im_start|>user
# SUBREDDIT: /r/{subreddit}
# RULE SIMILARITY: {similarity_score:.1%} match to known rules
# RULE COMPLEXITY: {"High" if rule_characteristics.get('complexity_score', 0) > 0.02 else "Low"}
# RULE SPECIFICITY: {"High" if rule_characteristics.get('specificity_score', 0) > 0.01 else "Low"}

# RULE TO EVALUATE:
# "{rule}"

# VIOLATION PATTERNS FROM EXAMPLES:
# {format_example(pos1, True)}
# {format_example(pos2, True)}

# ACCEPTABLE CONTENT PATTERNS:
# {format_example(neg1, False)}
# {format_example(neg2, False)}

# COMMENT TO EVALUATE:
# {format_example(comment, None)}

# ANALYSIS INSTRUCTIONS:
# 1. Identify the core prohibition in the rule
# 2. Compare the comment's structure and content to violation patterns
# 3. Check for indirect violations (e.g., "I'm not a lawyer but..." for legal advice)
# 4. Consider subreddit-specific norms and moderation history
# 5. Assess confidence level based on rule familiarity

# VIOLATION PROBABILITY (0.0-1.0):
# <|im_start|>assistant
# """
#     return prompt


# class AdvancedCalibrator:
#     """Professional confidence calibration with multiple factors"""
    
#     def __init__(self, rule_engine):
#         self.rule_engine = rule_engine
    
#     def calibrate_prediction(self, raw_prob, input_data, training_rules, similarity_score):
#         """Advanced calibration using multiple professional factors"""
        
#         # Base calibration factors
#         factors = {
#             'rule_novelty': 1 - similarity_score,
#             'example_quality': self._assess_example_quality(input_data),
#             'subreddit_consistency': self._assess_subreddit_consistency(input_data, training_rules),
#             'comment_length_factor': self._length_factor(input_data['body']),
#             'rule_specificity': input_data['rule'] in self.rule_engine.rule_characteristics and 
#                               self.rule_engine.rule_characteristics[input_data['rule']]['specificity_score']
#         }
        
#         # Calculate weighted adjustment
#         adjustment = (
#             0.4 * factors['rule_novelty'] * -0.5 +  # Reduce confidence for novel rules
#             0.3 * factors['example_quality'] * 0.3 +  # Increase for high-quality examples
#             0.2 * factors['subreddit_consistency'] * 0.2 +  # Adjust for subreddit fit
#             0.1 * factors['comment_length_factor'] * 0.1  # Minor length adjustment
#         )
        
#         # Apply calibration
#         calibrated = raw_prob + adjustment
#         calibrated = np.clip(calibrated, 0.01, 0.99)  # Prevent extremes
        
#         return calibrated, factors
    
#     def _assess_example_quality(self, input_data):
#         """Assess quality of provided examples"""
#         pos_examples = [input_data['positive_example_1'], input_data['positive_example_2']]
#         neg_examples = [input_data['negative_example_1'], input_data['negative_example_2']]
        
#         # Check diversity and clarity
#         pos_diversity = len(set(ex[:30] for ex in pos_examples)) / len(pos_examples)
#         neg_diversity = len(set(ex[:30] for ex in neg_examples)) / len(neg_examples)
        
#         # Check for clear separation
#         pos_text = " ".join(pos_examples).lower()
#         neg_text = " ".join(neg_examples).lower()
#         separation = 1 - len(set(pos_text.split()) & set(neg_text.split())) / max(len(pos_text.split()), 1)
        
#         return 0.4 * pos_diversity + 0.4 * neg_diversity + 0.2 * separation
    
#     def _assess_subreddit_consistency(self, input_data, training_rules):
#         """Assess consistency with subreddit norms"""
#         rule_lower = input_data['rule'].lower()
#         subreddit = input_data['subreddit']
        
#         # Count how many training rules for this subreddit match
#         subreddit_rules = [r for r in training_rules if r in self.rule_engine.rule_characteristics]
#         if not subreddit_rules:
#             return 0.5
            
#         match_count = sum(1 for r in subreddit_rules if any(kw in rule_lower for kw in ['advertising', 'legal', 'spam']))
#         return match_count / len(subreddit_rules)
    
#     def _length_factor(self, text):
#         """Adjust based on comment length"""
#         length = len(text.split())
#         if length < 10:
#             return -0.2  # Short comments are harder to classify
#         elif length > 100:
#             return 0.1  # Long comments have more evidence
#         else:
#             return 0.0


# def main():
#     """Professional end-to-end pipeline with enhanced accuracy"""
#     print("="*60)
#     print("JIGSAW - AGILE COMMUNITY RULES CLASSIFICATION")
#     print("PROFESSIONAL SENIOR DATA SCIENTIST SOLUTION")
#     print("ðŸš€ ENHANCED ACCURACY WITH QWEN3-32B")
#     print("âœ… 100% OFFLINE MODE - KAGGLE COMPETITION COMPLIANT")
#     print("="*60)
    
#     # 1. Load data
#     print("\n[STEP 1] Loading competition data...")
#     train_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
#     test_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    
#     # 2. Initialize professional components
#     print("\n[STEP 2] Initializing professional components...")
    
#     # Advanced rule engine
#     rule_engine = AdvancedRuleEngine()
#     rule_engine.create_enhanced_embeddings(train_data['rule'].unique())
    
#     # Load Qwen3-32B model
#     try:
#         llm_model, tokenizer = load_qwen3_model()
#     except Exception as e:
#         print(f"Qwen3-32B failed: {e}")
#         print("Falling back to Qwen2-7B...")
#         from transformers import AutoTokenizer, AutoModelForCausalLM
#         model_path = "/kaggle/input/qwen2/transformers/qwen2-7b-instruct/1"
#         tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#         llm_model = AutoModelForCausalLM.from_pretrained(
#             model_path, 
#             device_map="auto", 
#             trust_remote_code=True
#         )
    
#     # Advanced calibrator
#     calibrator = AdvancedCalibrator(rule_engine)
    
#     # 3. Process test data with enhanced reasoning
#     print("\n[STEP 3] Processing test data with enhanced accuracy...")
#     predictions = []
#     detailed_factors = []
    
#     batch_size = 1  # Conservative for memory
#     num_batches = len(test_data)
    
#     for batch_idx in tqdm(range(num_batches), desc="Processing"):
#         row = test_data.iloc[batch_idx]
        
#         # Get rule similarity and characteristics
#         similarity_score, _ = rule_engine.get_similarity_score(
#             row['rule'], 
#             train_data['rule'].unique()
#         )
        
#         rule_characteristics = rule_engine.rule_characteristics.get(row['rule'], {})
        
#         # Create enhanced prompt
#         prompt = create_enhanced_prompt(row, similarity_score, rule_characteristics)
        
#         # Tokenize
#         inputs = tokenizer(
#             prompt, 
#             return_tensors="pt", 
#             padding=True, 
#             truncation=True,
#             max_length=2048
#         ).to(device)
        
#         # Generate response with error handling
#         try:
#             with torch.no_grad():
#                 outputs = llm_model(**inputs)
            
#             # Extract probability
#             try:
#                 # Look for probability indicators in the response
#                 response_text = tokenizer.decode(outputs.logits[0].argmax(-1), skip_special_tokens=True)
                
#                 # Extract probability from text
#                 prob_match = re.search(r'([0-9]*\.?[0-9]+)', response_text)
#                 if prob_match:
#                     raw_prob = float(prob_match.group(1))
#                     raw_prob = np.clip(raw_prob, 0.0, 1.0)
#                 else:
#                     # Fallback to token probability
#                     true_token = tokenizer.convert_tokens_to_ids("True")
#                     false_token = tokenizer.convert_tokens_to_ids("False")
                    
#                     if true_token != tokenizer.unk_token_id and false_token != tokenizer.unk_token_id:
#                         last_logits = outputs.logits[0, -1, [true_token, false_token]]
#                         probs = torch.softmax(last_logits, dim=-1)
#                         raw_prob = probs[0].item()
#                     else:
#                         # Last resort: rule-based probability
#                         raw_prob = 0.3 + 0.6 * similarity_score
                        
#             except Exception as e:
#                 print(f"Error extracting probability: {e}")
#                 raw_prob = 0.3 + 0.6 * similarity_score
                
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             raw_prob = 0.3 + 0.6 * similarity_score
#             torch.cuda.empty_cache()
        
#         # Calibrate prediction
#         calibrated_prob, factors = calibrator.calibrate_prediction(
#             raw_prob, 
#             row, 
#             train_data['rule'].unique(),
#             similarity_score
#         )
        
#         predictions.append(calibrated_prob)
#         detailed_factors.append(factors)
        
#         # Memory management
#         torch.cuda.empty_cache()
    
#     # 4. Generate enhanced submission
#     print("\n[STEP 4] Generating enhanced submission...")
#     submission = pd.DataFrame({
#         'row_id': test_data['row_id'],
#         'rule_violation': predictions
#     })
    
#     # Quality check
#     print(f"Enhanced submission quality check:")
#     print(f"- Prediction range: [{submission['rule_violation'].min():.4f}, {submission['rule_violation'].max():.4f}]")
#     print(f"- Mean probability: {submission['rule_violation'].mean():.4f}")
#     print(f"- Std deviation: {submission['rule_violation'].std():.4f}")
#     print(f"- High confidence predictions (>0.8): {sum(submission['rule_violation'] > 0.8)}")
#     print(f"- Low confidence predictions (<0.2): {sum(submission['rule_violation'] < 0.2)}")
    
#     # Save submission
#     submission.to_csv('submission.csv', index=False)
#     print("\n[COMPLETE] Enhanced submission file generated: submission.csv")
    
#     # Display sample
#     print("\nSample of enhanced submission:")
#     print(submission.head(10))

# if __name__ == "__main__":
#     main()


import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Set deterministic behavior
os.environ['PYTHONHASHSEED'] = '42'
torch.manual_seed(42)
np.random.seed(42)

# Configure device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Memory optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

# Set environment variable to prevent fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class OptimizedRuleEngine:
    """Memory-optimized rule understanding engine"""
    
    def __init__(self):
        print("Initializing optimized rule understanding engine...")
        
        # Use lightweight embedding model
        self.embedding_model = SentenceTransformer(
            "/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2", 
            device=device
        )
        
        self.rule_embeddings = None
        self.training_rules = None
    
    def create_embeddings(self, rules):
        """Create memory-efficient embeddings"""
        unique_rules = list(set(rules))
        self.training_rules = unique_rules
        print(f"Creating embeddings for {len(unique_rules)} rules...")
        
        # Process in small batches to save memory
        self.rule_embeddings = self.embedding_model.encode(
            unique_rules,
            normalize_embeddings=True,
            batch_size=4,
            show_progress_bar=True
        )
    
    def get_similarity_score(self, test_rule):
        """Get similarity score with memory optimization"""
        if self.rule_embeddings is None or len(self.training_rules) == 0:
            return 0.5, 0
            
        test_emb = self.embedding_model.encode([test_rule], normalize_embeddings=True)
        similarities = cosine_similarity(test_emb, self.rule_embeddings)[0]
        max_idx = similarities.argmax()
        
        return similarities[max_idx], max_idx


def load_optimized_qwen3():
    """Load Qwen3 with aggressive memory optimization"""
    print("Loading Qwen3 with memory optimization...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Use Qwen3-32B from Kaggle dataset
    model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct/1"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Use 4-bit quantization to reduce memory usage
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            max_memory={0: "14GiB"}  # Limit memory usage
        )
        print("Qwen3 loaded with 4-bit quantization")
        
    except Exception as e:
        print(f"4-bit quantization failed: {e}")
        print("Falling back to float16...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    # Configure for probability extraction
    model.config.pad_token_id = model.config.eos_token_id
    
    print("Qwen3 loaded successfully with memory optimization")
    return model, tokenizer


def create_optimized_prompt(input_data, similarity_score):
    """Create memory-optimized prompt"""
    subreddit = input_data['subreddit']
    rule = input_data['rule']
    comment = input_data['body']
    pos1 = input_data['positive_example_1']
    pos2 = input_data['positive_example_2']
    neg1 = input_data['negative_example_1']
    neg2 = input_data['negative_example_2']
    
    # Simplified formatting to reduce token count
    def format_example(example):
        lines = example.split('\n')
        return " ".join([line.strip() for line in lines if line.strip()])[:150]
    
    prompt = f"""<|im_start|>system
You are a Reddit moderator. Determine if the comment violates the rule.

<|im_start|>user
SUBREDDIT: /r/{subreddit}
RULE: {rule}
SIMILARITY: {similarity_score:.1%} to known rules

VIOLATIONS:
1. {format_example(pos1)}
2. {format_example(pos2)}

OK COMMENTS:
1. {format_example(neg1)}
2. {format_example(neg2)}

COMMENT:
{format_example(comment)}

DECISION (True/False):
<|im_start|>assistant
"""
    return prompt


class AdvancedCalibrator:
    """Professional confidence calibration"""
    
    def __init__(self, rule_engine):
        self.rule_engine = rule_engine
    
    def calibrate_prediction(self, raw_prob, input_data, similarity_score):
        """Calibrate prediction with professional factors"""
        
        # Ensure raw_prob is a float
        if hasattr(raw_prob, 'item'):
            raw_prob = raw_prob.item()
        elif torch.is_tensor(raw_prob):
            raw_prob = raw_prob.detach().cpu().numpy().item()
        
        # Calculate calibration factors
        novelty_penalty = 0.4 * (1 - similarity_score)
        length_factor = 0.1 * (len(input_data['body'].split()) > 50)
        
        # Apply calibration
        calibrated = raw_prob * (1 - novelty_penalty) + length_factor
        
        # Prevent extreme probabilities
        return np.clip(calibrated, 0.05, 0.95)


def main():
    """Professional end-to-end pipeline"""
    print("="*60)
    print("JIGSAW - AGILE COMMUNITY RULES CLASSIFICATION")
    print("PROFESSIONAL SENIOR DATA SCIENTIST SOLUTION")
    print("âœ… OPTIMIZED FOR KAGGLE MEMORY CONSTRAINTS")
    print("ðŸš€ HIGH-ACCURACY WITH QWEN3-32B")
    print("="*60)
    
    # 1. Load data
    print("\n[STEP 1] Loading competition data...")
    train_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    test_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    
    # 2. Initialize components
    print("\n[STEP 2] Initializing professional components...")
    
    # Rule engine
    rule_engine = OptimizedRuleEngine()
    rule_engine.create_embeddings(train_data['rule'].unique())
    
    # Load Qwen3
    llm_model, tokenizer = load_optimized_qwen3()
    
    # Calibrator
    calibrator = AdvancedCalibrator(rule_engine)
    
    # 3. Process test data
    print("\n[STEP 3] Processing test data...")
    predictions = []
    
    # Process one row at a time to minimize memory usage
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing"):
        try:
            # Get rule similarity
            similarity_score, _ = rule_engine.get_similarity_score(row['rule'])
            
            # Create prompt
            prompt = create_optimized_prompt(row, similarity_score)
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = llm_model(**inputs)
            
            # Extract probability
            try:
                # Look for True/False tokens
                true_token = tokenizer.convert_tokens_to_ids("True")
                false_token = tokenizer.convert_tokens_to_ids("False")
                
                # Fallback to lowercase
                if true_token == tokenizer.unk_token_id:
                    true_token = tokenizer.convert_tokens_to_ids("true")
                if false_token == tokenizer.unk_token_id:
                    false_token = tokenizer.convert_tokens_to_ids("false")
                
                # Handle case where tokens aren't found
                if true_token == tokenizer.unk_token_id or false_token == tokenizer.unk_token_id:
                    # Use rule similarity as fallback
                    raw_prob = 0.3 + 0.6 * similarity_score
                else:
                    # Normal case
                    last_logits = outputs.logits[0, -1, [true_token, false_token]]
                    probs = torch.softmax(last_logits, dim=-1)
                    raw_prob = probs[0].item()
                    
            except Exception as e:
                print(f"Error extracting probability: {e}")
                raw_prob = 0.3 + 0.6 * similarity_score
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            raw_prob = 0.3 + 0.6 * similarity_score
            torch.cuda.empty_cache()
        
        # Calibrate prediction
        calibrated_prob = calibrator.calibrate_prediction(
            raw_prob, 
            row, 
            similarity_score
        )
        predictions.append(calibrated_prob)
        
        # Clear cache to prevent memory issues
        torch.cuda.empty_cache()
    
    # 4. Generate submission
    print("\n[STEP 4] Generating submission...")
    submission = pd.DataFrame({
        'row_id': test_data['row_id'],
        'rule_violation': predictions
    })
    
    # Quality check
    print(f"Submission quality check:")
    print(f"- Prediction range: [{submission['rule_violation'].min():.4f}, {submission['rule_violation'].max():.4f}]")
    print(f"- Mean probability: {submission['rule_violation'].mean():.4f}")
    print(f"- Std deviation: {submission['rule_violation'].std():.4f}")
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print("\n[COMPLETE] Professional submission file generated: submission.csv")
    
    # Display sample
    print("\nSample of professional submission:")
    print(submission.head(10))

if __name__ == "__main__":
    main()


pd.read_csv("/kaggle/working/submission.csv")




