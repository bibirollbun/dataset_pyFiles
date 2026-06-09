import subprocess
import sys

subprocess.run([sys.executable, "/kaggle/input/gemma3n-dependencies/install_gemma3n.py"])


%%writefile main_inference.py
"""Gemma 3n inference script for MAP competition.
Must be run via subprocess with correct LD_LIBRARY_PATH.
"""
import os
import gc
import sys
import traceback

def main():
    try:
        print("="*60)
        print("Starting Gemma 3n Inference")
        print("="*60)
        
        # Print environment info for debugging
        print(f"\nPython: {sys.executable}")
        print(f"Working dir: {os.getcwd()}")
        ld_path = os.environ.get('LD_LIBRARY_PATH', 'NOT SET')
        print(f"LD_LIBRARY_PATH includes nccl: {'nccl' in ld_path}")
        
        # Memory cleanup before imports
        gc.collect()
        
        print("\nImporting pandas...")
        import pandas as pd
        
        print("Importing torch...")
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Determine device
        DEVICE = "cpu"  # Default to CPU
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU: {gpu_name}")
                # Check compute capability
                major, minor = torch.cuda.get_device_capability(0)
                print(f"Compute capability: {major}.{minor}")
                if major >= 7:
                    DEVICE = "cuda"
                    torch.cuda.empty_cache()
                    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB used")
                else:
                    print(f"GPU compute capability {major}.{minor} < 7.0, using CPU")
            except Exception as e:
                print(f"GPU check failed: {e}, using CPU")
        
        print(f"\nUsing device: {DEVICE}")
        
        print("\nImporting transformers...")
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        from tqdm.auto import tqdm
        
        # --- Configuration ---
        MODEL_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e4b-it/2"
        TEST_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
        SUBMISSION_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv"
        MAX_NEW_TOKENS = 128
        TEMPERATURE = 0.5
        
        # Valid categories and misconceptions
        CATEGORIES = [
            "True_Correct", "True_Neither", "True_Misconception",
            "False_Correct", "False_Neither", "False_Misconception"
        ]
        
        MISCONCEPTIONS = [
            "NA", "Incomplete", "Additive", "Duplication", "Subtraction", 
            "Positive", "Wrong_term", "Irrelevant", "Wrong_fraction", 
            "Inversion", "Mult", "Denominator-only_change", "Whole_numbers_larger",
            "Adding_across", "WNB", "Tacking", "Unknowable", "Wrong_Fraction",
            "SwapDividend", "Scale", "Not_variable", "Firstterm", "Adding_terms",
            "Multiplying_by_4", "FlipChange", "Division", "Definition", "Interior",
            "Longer_is_bigger", "Ignores_zeroes", "Base_rate", "Shorter_is_bigger",
            "Inverse_operation", "Certainty", "Incorrect_equivalent_fraction_addition",
            "Wrong_Operation"
        ]
        
        def create_prompt(row):
            """Create prompt for the model to classify student response."""
            prompt = f"""Analyze this student's math response and classify it.

Question: {row['QuestionText']}
Student's Answer Choice: {row['MC_Answer']}
Student's Explanation: {row['StudentExplanation']}

Classify into one of these categories: {', '.join(CATEGORIES)}

If the category contains "Misconception", also identify the misconception type.

Respond in exactly this format:
Category: [category]
Misconception: [misconception or NA]"""
            return prompt
        
        def parse_model_output(output_text):
            """Parse model output into Category:Misconception format."""
            category = "True_Neither"
            misconception = "NA"
            
            lines = output_text.strip().split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'category:' in line_lower:
                    cat_part = line.split(':', 1)[-1].strip()
                    for cat in CATEGORIES:
                        if cat.lower() in cat_part.lower():
                            category = cat
                            break
                elif 'misconception:' in line_lower:
                    misc_part = line.split(':', 1)[-1].strip()
                    if misc_part.lower() not in ('na', 'none', 'n/a'):
                        for misc in MISCONCEPTIONS:
                            if misc.lower() in misc_part.lower():
                                misconception = misc
                                break
            
            if "Misconception" not in category:
                misconception = "NA"
            
            return f"{category}:{misconception}"
        
        # 1. Load Data
        print("\nLoading data...")
        test_df = pd.read_csv(TEST_PATH)
        submission_df = pd.read_csv(SUBMISSION_PATH)
        print(f"Test samples: {len(test_df)}")
        print(f"Submission template rows: {len(submission_df)}")
        
        # 2. Load Model
        print("\nLoading Gemma 3n model...")
        print(f"Model path: {MODEL_PATH}")
        
        if DEVICE == "cuda":
            print("Using 4-bit quantization for GPU...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_PATH,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            print("Loading model on CPU (no quantization)...")
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_PATH,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print("Model loaded successfully!")
        
        model.eval()
        
        # 3. Inference Loop
        predictions = []
        print("\nStarting inference...")
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating Predictions"):
            try:
                prompt = create_prompt(row)
                inputs = processor(text=prompt, return_tensors="pt")
                if DEVICE == "cuda":
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                    )
                
                generated_text = processor.decode(outputs[0], skip_special_tokens=True)
                answer = generated_text.replace(prompt, "").strip()
                formatted_prediction = parse_model_output(answer)
                predictions.append(formatted_prediction)
                
                # Memory cleanup
                del inputs, outputs
                if idx % 5 == 0:
                    gc.collect()
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                print(f"\nError on row {idx}: {e}")
                predictions.append("True_Neither:NA")  # Fallback prediction
        
        # 4. Create Submission
        print("\nCreating submission file...")
        submission_df["Category:Misconception"] = predictions
        
        output_path = "/kaggle/working/submission.csv"
        submission_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"SUCCESS! Submission saved to {output_path}")
        print(f"{'='*60}")
        print(f"\nSubmission shape: {submission_df.shape}")
        print(submission_df.head())
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("FATAL ERROR")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())


import subprocess
import sys
import os

print("ğŸš€ Running inference via run_inference.py wrapper...")
print("This wrapper sets LD_LIBRARY_PATH before loading torch.")
print("="*60)

# Use the wrapper script provided by the dependencies dataset
wrapper_script = "/kaggle/input/gemma3n-dependencies/run_inference.py"
inference_script = "/kaggle/working/main_inference.py"

# Run with output capture for debugging
result = subprocess.run(
    [sys.executable, wrapper_script, inference_script],
    cwd="/kaggle/working",
    capture_output=True,
    text=True
)

# Print all output
print("\nSTDOUT:")
print("="*60)
print(result.stdout if result.stdout else "(empty)")

if result.stderr:
    print("\nSTDERR:")
    print("="*60)
    print(result.stderr)

print(f"\nReturn code: {result.returncode}")

# Check for submission file
if os.path.exists("/kaggle/working/submission.csv"):
    print("\nâœ… submission.csv created successfully!")
    import pandas as pd
    sub = pd.read_csv("/kaggle/working/submission.csv")
    print(f"Shape: {sub.shape}")
    print(sub.head())
else:
    print("\nâ�Œ submission.csv not found!")
    print("Files in /kaggle/working:", os.listdir("/kaggle/working"))
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed with return code {result.returncode}")

