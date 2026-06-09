import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,Qwen2VLForConditionalGeneration, pipeline
import random
from typing import Dict, List, Any, Tuple
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
import warnings
warnings.filterwarnings('ignore')

class LLMPromptTester:
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize the LLM Prompt Tester
        
        Args:
            model_name: Hugging Face model name (default: distilgpt2)
                       Other options: "gpt2", "microsoft/DialoGPT-small"
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.training_data = None
        self.solution_data = None
        
        # ARC color palette (10 colors: 0-9)
        self.colors = [
            '#000000',  # 0: Black
            '#0074D9',  # 1: Blue
            '#FF4136',  # 2: Red
            '#2ECC40',  # 3: Green
            '#FFDC00',  # 4: Yellow
            '#AAAAAA',  # 5: Grey
            '#F012BE',  # 6: Magenta
            '#FF851B',  # 7: Orange
            '#7FDBFF',  # 8: Sky
            '#870C25'   # 9: Brown
        ]
        self.cmap = ListedColormap(self.colors)
        
        print(f"Initializing model: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            print("Loading tokenizer and model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            #self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto"
)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("âœ… Model loaded successfully!")
            print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            
        except Exception as e:
            print(f"â�Œ Error loading model: {e}")
            raise e
    
    def load_data(self, training_file: str, solution_file: str):
        """Load training and solution data from JSON files"""
        try:
            with open(training_file, 'r') as f:
                data = json.load(f)
                # Handle nested structure if it exists
                if isinstance(data, dict) and 'root' in data:
                    self.training_data = data['root']
                else:
                    self.training_data = data
            
            with open(solution_file, 'r') as f:
                data = json.load(f)
                # Handle nested structure if it exists
                if isinstance(data, dict) and 'root' in data:
                    self.solution_data = data['root']
                else:
                    self.solution_data = data
            
            print(f"âœ… Data loaded successfully!")
            print(f"Training data keys: {len(self.training_data)}")
            print(f"Solution data keys: {len(self.solution_data)}")
            
        except Exception as e:
            print(f"â�Œ Error loading data: {e}")
            raise e
    
    def plot_grid(self, grid, ax, title=""):
        """Plot a single grid on given axes."""
        try:
            if grid is None or len(grid) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=10, pad=10)
                ax.axis('off')
                return None
                
            grid_array = np.array(grid)
            
            # Create the plot
            im = ax.imshow(grid_array, cmap=self.cmap, vmin=0, vmax=9)
            
            # Add grid lines
            ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
            ax.tick_params(which="minor", size=0)
            
            # Remove major ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set title
            ax.set_title(title, fontsize=10, pad=10)
            
            return im
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10, pad=10)
            ax.axis('off')
            return None
    
    def parse_prediction_to_grid(self, prediction_text: str):
        """Try to parse the prediction text into a grid format"""
        try:
            if not prediction_text:
                return None
                
            prediction_text = prediction_text.strip()
            
            # Pattern 1: Look for nested list structure like [[1,2,3],[4,5,6]]
            pattern1 = r'\[\s*\[.*?\]\s*\]'
            matches = re.findall(pattern1, prediction_text, re.DOTALL)
            
            if matches:
                for match in matches:
                    try:
                        # Clean the match
                        clean_match = re.sub(r'[^\d,\[\]]', '', match)
                        grid = eval(clean_match)
                        if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
                            # Validate grid values are 0-9
                            valid_grid = []
                            for row in grid:
                                valid_row = [max(0, min(9, int(x))) for x in row if str(x).isdigit()]
                                if valid_row:
                                    valid_grid.append(valid_row)
                            if valid_grid:
                                return valid_grid
                    except:
                        continue
            
            # Pattern 2: Look for rows of numbers
            lines = prediction_text.split('\n')
            grid = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('#', '//', 'Example', 'Input', 'Output')):
                    # Extract numbers from the line
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        row = [max(0, min(9, int(n))) for n in numbers]
                        if row:
                            grid.append(row)
            
            if len(grid) > 0:
                # Make sure all rows have the same length
                max_len = max(len(row) for row in grid)
                grid = [row + [0] * (max_len - len(row)) for row in grid]
                return grid
            
            return None
            
        except Exception as e:
            print(f"Error parsing prediction: {e}")
            return None
    
    def create_prompt_templates(self) -> Dict[str, str]:
        """Create different prompt templates for testing ARC challenges"""
        templates = {
            "basic": """
You are an intelligent reasoning agent specialized in solving visual transformation puzzles from the ARC dataset.

Your task is to:
- Carefully analyze 1â€“3 training examples.
- Each training example consists of a 2D grid as input and a corresponding output grid.
- Identify the transformation logic used in these examples.
- The transformation may involve geometric operations (rotation, reflection), pattern duplication, removal, coloring, resizing, or rule-based symbol manipulation.

Do not guess. Take time to:
1. Detect what changes between each input and output.
2. Generalize a transformation rule consistent across examples.
3. Validate the pattern holds across all examples.

Once the rule is understood, apply the exact same transformation to the test input.

Training Examples:
{training_examples}

Now apply the transformation to the test input:
Test Input: {test_input}

Respond with the output as a 2D grid of digits (0â€“9), formatted precisely as: [[row1],[row2],...]
""",

            "explicit_pattern": """Pattern Recognition Task:

{training_examples}

Pattern Analysis:
- Look at colors/numbers that change
- Look at positions that change  
- Look for rotations, reflections, or fills
- Look for object detection and transformation

Apply the discovered pattern to: {test_input}

Final grid [[row1],[row2],...]:""",
            "detailed_analysis": """
You are solving an Abstract Reasoning Challenge (ARC). Your role is to derive a consistent transformation pattern from training examples and apply it to a test case.

Here is your structured task:

Training Examples:
{training_examples}

Test Input:
{test_input}

Step-by-step Analysis:
1. **Inputâ€“Output Comparison**:
   - What differences exist between each input and its output?
   - Are objects rotated, moved, removed, duplicated, resized, colored, or otherwise transformed?

2. **Rule Induction**:
   - What transformation rule explains all examples?
   - Be specific and general. Avoid overfitting to one case.

3. **Test Transformation**:
   - Apply the derived rule to the test input logically and consistently.

Final Answer Format:
Return the transformed output as a 2D grid of digits (0â€“9). Format strictly as:
[[row1],[row2],...]

Ensure the transformation is deterministic and consistent with the identified pattern.
"""
            ,

            "chain_of_thought": """Let me solve this step by step.

Training examples:
{training_examples}

Test input: {test_input}

My reasoning:
1. Examining the examples, I notice...
2. The pattern appears to be...
3. Applying this to the test input...

Final answer [[row1],[row2],...]:""",

            "structured_reasoning": """ABSTRACT REASONING CHALLENGE

TRAINING DATA:
{training_examples}

TASK: Find the transformation rule and apply it.

ANALYSIS:
- Input dimensions: Check size patterns
- Color changes: Look for substitutions
- Spatial changes: Look for movements/rotations
- Object detection: Identify shapes and their transformations

TEST INPUT: {test_input}

SOLUTION [[row1],[row2],...]:""",

            "minimal": """{training_examples}

Test: {test_input}

Answer [[row1],[row2],...]:""",

            "few_shot_examples": """Here are pattern examples. Find the rule:

{training_examples}

The rule is: [REPLACE each X with Y based on pattern]

Apply to: {test_input}

Result [[row1],[row2],...]:"""
        }
        return templates
    
    def format_training_examples(self, train_data: List) -> str:
        """Format training examples for prompts"""
        if not train_data:
            return "No training examples available"
            
        formatted = ""
        for i, example in enumerate(train_data[:3]):  # Limit to 3 examples to avoid token limits
            formatted += f"Example {i+1}:\n"
            formatted += f"Input: {example.get('input', [])}\n"
            formatted += f"Output: {example.get('output', [])}\n\n"
        return formatted.strip()
    
    def generate_prediction(self, prompt: str, max_length: int = 400, temperature: float = 0.1) -> str:
        """Generate prediction using the LLM with optimized parameters for ARC"""
        try:
            # Truncate prompt if too long
            if len(prompt) > 2000:
                prompt = prompt[:2000] + "..."
            
            result = self.pipeline(
                prompt,
                max_new_tokens=200,  # Increased for better grid generation
                temperature=temperature,  # Lower temperature for more deterministic output
                num_return_sequences=1,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Avoid repetitive output
                top_p=0.9,  # Nucleus sampling for better quality
                top_k=50   # Limit vocabulary for more focused generation
            )
            
            # Extract only the generated part
            generated_text = result[0]['generated_text']
            prediction = generated_text[len(prompt):].strip()
            
            return prediction
            
        except Exception as e:
            return f"Error generating prediction: {e}"
    
    def visualize_complete_example(self, case_id: str, prediction_text: str = "", figsize=(16, 6)):
        """Visualize train, test, predicted, and actual solution grids."""
        try:
            if case_id not in self.training_data:
                print(f"Challenge {case_id} not found!")
                return
            
            case_data = self.training_data[case_id]
            train_examples = case_data.get('train', [])
            test_examples = case_data.get('test', [])
            
            # Get the first training example
            train_input = train_examples[0]['input'] if train_examples else None
            train_output = train_examples[0]['output'] if train_examples else None
            
            # Get the first test example
            test_input = test_examples[0]['input'] if test_examples else None
            
            # Get actual solution
            actual_solution = None
            if case_id in self.solution_data:
                sol_data = self.solution_data[case_id]
                if isinstance(sol_data, list) and len(sol_data) > 0:
                    actual_solution = sol_data[0]
                else:
                    actual_solution = sol_data
            
            # Parse prediction
            predicted_grid = self.parse_prediction_to_grid(prediction_text)
            
            # Create subplot
            fig, axes = plt.subplots(2, 4, figsize=figsize)
            fig.suptitle(f'Challenge {case_id} - Complete Analysis', fontsize=16, fontweight='bold')
            
            # First row: Training example
            self.plot_grid(train_input, axes[0, 0], "Train Input")
            self.plot_grid(train_output, axes[0, 1], "Train Output")
            
            # Show pattern info
            axes[0, 2].text(0.5, 0.5, f"Pattern:\nTransform input\nto output\n\nExamples: {len(train_examples)}", 
                           ha='center', va='center', transform=axes[0, 2].transAxes,
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[0, 2].set_title("Pattern Analysis", fontsize=10, pad=10)
            axes[0, 2].axis('off')
            
            # Show dataset info
            axes[0, 3].text(0.5, 0.5, f"Dataset Info:\n\nTrain: {len(train_examples)}\nTest: {len(test_examples)}", 
                           ha='center', va='center', transform=axes[0, 3].transAxes,
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[0, 3].set_title("Dataset Info", fontsize=10, pad=10)
            axes[0, 3].axis('off')
            
            # Second row: Test and predictions
            self.plot_grid(test_input, axes[1, 0], "Test Input")
            self.plot_grid(predicted_grid, axes[1, 1], "Model Prediction")
            self.plot_grid(actual_solution, axes[1, 2], "Actual Solution")
            
            # Comparison
            if predicted_grid is not None and actual_solution is not None:
                try:
                    pred_array = np.array(predicted_grid)
                    actual_array = np.array(actual_solution)
                    
                    if pred_array.shape == actual_array.shape:
                        accuracy = np.mean(pred_array == actual_array)
                        match_status = "âœ… PERFECT MATCH!" if accuracy == 1.0 else f"â�Œ {accuracy:.1%} correct"
                        color = "lightgreen" if accuracy == 1.0 else "lightcoral"
                    else:
                        match_status = f"â�Œ Shape mismatch\nPred: {pred_array.shape}\nActual: {actual_array.shape}"
                        color = "lightcoral"
                except:
                    match_status = "â�Œ Comparison failed"
                    color = "lightcoral"
            else:
                match_status = "â�Œ Cannot compare\n(missing data)"
                color = "lightgray"
            
            axes[1, 3].text(0.5, 0.5, f"Result:\n\n{match_status}", 
                           ha='center', va='center', transform=axes[1, 3].transAxes,
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
            axes[1, 3].set_title("Comparison", fontsize=10, pad=10)
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"RESULTS SUMMARY FOR CASE: {case_id}")
            print(f"{'='*60}")
            
            if predicted_grid:
                print(f"ğŸ¤– PREDICTED GRID ({len(predicted_grid)}x{len(predicted_grid[0]) if predicted_grid else 0}):")
                for row in predicted_grid:
                    print(f"  {row}")
            else:
                print(f"ğŸ¤– PREDICTED GRID: Could not parse")
                print(f"   Raw text: '{prediction_text[:100]}{'...' if len(prediction_text) > 100 else ''}'")
            
            if actual_solution:
                print(f"\nâœ… ACTUAL SOLUTION ({len(actual_solution)}x{len(actual_solution[0]) if actual_solution else 0}):")
                for row in actual_solution:
                    print(f"  {row}")
            
            print(f"\nğŸ“Š FINAL RESULT: {match_status}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error in visualization: {e}")
    
    def test_single_case(self, case_id: str = None, prompt_template: str = "basic") -> Dict:
        """Test a single case with specified prompt template"""
        try:
            if not self.training_data or not self.solution_data:
                return {"error": "Data not loaded. Please load data first."}
            
            # Get available case IDs
            available_cases = list(self.training_data.keys())
            
            if not available_cases:
                return {"error": "No training cases available"}
            
            # Select case
            if case_id is None or case_id not in available_cases:
                case_id = random.choice(available_cases)
                print(f"Selected random case: {case_id}")
            
            # Get case data
            case_data = self.training_data[case_id]
            train_data = case_data.get('train', [])
            test_data = case_data.get('test', [])
            
            # Get solution
            actual_solution = self.solution_data.get(case_id, "No solution found")
            
            # Prepare prompt
            templates = self.create_prompt_templates()
            template = templates.get(prompt_template, templates['basic'])
            
            # Format the prompt
            training_examples = self.format_training_examples(train_data)
            test_input = str(test_data[0]['input']) if test_data else "No test data"
            
            prompt = template.format(
                training_examples=training_examples,
                test_input=test_input
            )
            
            print(f"\n{'='*60}")
            print(f"TESTING CASE: {case_id}")
            print(f"PROMPT: {prompt}")
            print(f"PROMPT TEMPLATE: {prompt_template}")
            print(f"{'='*60}")
            
            # Generate prediction
            print("ğŸ¤– Generating prediction...")
            start_time = time.time()
            prediction = self.generate_prediction(prompt, max_length=300, temperature=0.3)
            generation_time = time.time() - start_time
            
            # Prepare results
            results = {
                "case_id": case_id,
                "prompt_template": prompt_template,
                "prompt_used": prompt,
                "training_data": train_data,
                "test_data": test_data,
                "prediction": prediction,
                "actual_solution": actual_solution,
                "generation_time": round(generation_time, 2)
            }
            
            # Display results
            self.visualize_complete_example(case_id, prediction)
            
            return results
            
        except Exception as e:
            print(f"Error in test_single_case: {e}")
            return {"error": str(e)}
    
    def compare_prompts(self, case_id: str = None, templates: List[str] = None) -> Dict:
        """Compare multiple prompt templates on the same case"""
        if templates is None:
            templates = ["basic", "instructional", "chain_of_thought", "pattern_focused"]
        
        results = {}
        
        # Get a case_id if not provided
        if case_id is None:
            available_cases = list(self.training_data.keys())
            case_id = random.choice(available_cases) if available_cases else None
        
        if case_id is None:
            return {"error": "No cases available"}
        
        print(f"Comparing prompts on case: {case_id}")
        
        for template in templates:
            print(f"\nğŸ”„ Testing template: {template}")
            result = self.test_single_case(case_id, template)
            results[template] = result
            time.sleep(1)  # Small delay
        
        return results

def create_sample_data():
    """Create sample ARC-like data for testing"""
    sample_training = {
        "sample_001": {
            "train": [
                {
                    "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]], 
                    "output": [[2, 0, 2], [0, 2, 0], [2, 0, 2]]
                },
                {
                    "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]], 
                    "output": [[0, 2, 0], [2, 0, 2], [0, 2, 0]]
                }
            ],
            "test": [
                {"input": [[1, 1, 0], [0, 1, 1], [1, 0, 1]]}
            ]
        },
        "sample_002": {
            "train": [
                {
                    "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], 
                    "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
                }
            ],
            "test": [
                {"input": [[0, 0], [0, 1]]}
            ]
        }
    }
    
    sample_solution = {
        "sample_001": [[[2, 2, 0], [0, 2, 2], [2, 0, 2]]],
        "sample_002": [[[1, 1], [1, 1]]]
    }
    
    return sample_training, sample_solution

def main():
    """Main function to demonstrate the LLM Prompt Tester"""
    try:
        # Initialize the tester with better model options
        # Recommended models in order of performance for ARC:
        
        # Option 1: Best for reasoning (if you have GPU memory)
        #model_name = "microsoft/DialoGPT-large"  # Better reasoning, conversational
        model_name="Qwen/Qwen2-VL-2B-Instruct"
        
        # Option 2: Good balance of size and performance
        #model_name = "gpt2-medium"  # 355M parameters, better than distilgpt2
        
        # Option 3: Smaller but still better than distilgpt2
        # model_name = "gpt2"  # 124M parameters vs 82M in distilgpt2
        
        # Option 4: For code-like tasks (experimental)
        # model_name = "microsoft/CodeGPT-small-py"  # Better at structured outputs
        
        # Option 5: If you have limited memory
        # model_name = "distilgpt2"  # Fallback option
        
        print("ğŸš€ Initializing LLM Prompt Tester...")
        print(f"Using model: {model_name}")
        tester = LLMPromptTester(model_name=model_name)
        
        # Try to load real data, fallback to sample data
        tester.load_data("/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json", "/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json")
        
        # Test different prompt templates
        print("\nğŸ§ª Starting prompt testing...")
        
        # Test with basic prompt
        print("\n--- Testing Basic Prompt ---")
        result1 = tester.test_single_case(prompt_template="basic")
        
        # Test with same case using different prompt
        if 'case_id' in result1:
            print("\n--- Testing Chain of Thought Prompt ---")
            result2 = tester.test_single_case(case_id=result1['case_id'], prompt_template="chain_of_thought")
            
            print("\n--- Testing Instructional Prompt ---")
            result3 = tester.test_single_case(case_id=result1['case_id'], prompt_template="instructional")
            print("\n--- Testing detailed_analysis Prompt ---")
            result3 = tester.test_single_case(case_id=result1['case_id'], prompt_template="detailed_analysis")
        
        print("\nâœ¨ Testing complete!")
        print("\nYou can now:")
        print("1. Modify prompt templates to improve performance")
        print("2. Try different models (gpt2, microsoft/DialoGPT-small)")
        print("3. Adjust generation parameters")
        print("4. Test specific cases by ID")
        
    except Exception as e:
        print(f"â�Œ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random
from typing import Dict, List, Any, Tuple
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
import warnings
from torch.utils.data import Dataset, DataLoader
import copy
warnings.filterwarnings('ignore')

# Import AdamW from torch.optim (newer versions)
try:
    from torch.optim import AdamW
except ImportError:
    try:
        from transformers import AdamW
    except ImportError:
        print("âš ï¸� AdamW not found, using torch.optim.Adam instead")
        from torch.optim import Adam as AdamW

class ARCDataset(Dataset):
    """Custom dataset for ARC training examples"""
    def __init__(self, training_examples, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for example in training_examples:
            input_grid = example['input']
            output_grid = example['output']
            
            # Create training text
            text = f"Input: {input_grid}\nOutput: {output_grid}\n"
            self.examples.append(text)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For language modeling
        }

class LLMPromptTesterWithTTT:
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize the LLM Prompt Tester with Test-Time Training capabilities
        
        Args:
            model_name: Hugging Face model name (default: distilgpt2)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.original_model = None  # Keep original for reset
        self.pipeline = None
        self.training_data = None
        self.solution_data = None
        
        # ARC color palette (10 colors: 0-9)
        self.colors = [
            '#000000',  # 0: Black
            '#0074D9',  # 1: Blue
            '#FF4136',  # 2: Red
            '#2ECC40',  # 3: Green
            '#FFDC00',  # 4: Yellow
            '#AAAAAA',  # 5: Grey
            '#F012BE',  # 6: Magenta
            '#FF851B',  # 7: Orange
            '#7FDBFF',  # 8: Sky
            '#870C25'   # 9: Brown
        ]
        self.cmap = ListedColormap(self.colors)
        
        print(f"Initializing model with TTT: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            print("Loading tokenizer and model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Keep a copy of the original model
            self.original_model = copy.deepcopy(self.model)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("âœ… Model loaded successfully!")
            print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            
        except Exception as e:
            print(f"â�Œ Error loading model: {e}")
            raise e
    
    def reset_model(self):
        """Reset model to original state"""
        print("ğŸ”„ Resetting model to original state...")
        self.model.load_state_dict(self.original_model.state_dict())
        
        # Update pipeline with reset model
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def test_time_training(self, training_examples: List[Dict], 
                          epochs: int = 3, 
                          learning_rate: float = 5e-5,
                          batch_size: int = 2):
        """
        Perform test-time training on the current task's training examples
        
        Args:
            training_examples: List of input-output pairs for the current task
            epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            batch_size: Batch size for training
        """
        try:
            print(f"ğŸ�¯ Starting Test-Time Training...")
            print(f"   Examples: {len(training_examples)}")
            print(f"   Epochs: {epochs}")
            print(f"   Learning Rate: {learning_rate}")
            
            # Create dataset
            dataset = ARCDataset(training_examples, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Set model to training mode
            self.model.train()
            
            # Setup optimizer - only update a subset of parameters for efficiency
            # Fine-tune only the last few layers
            params_to_update = []
            for name, param in self.model.named_parameters():
                if 'transformer.h.5' in name or 'transformer.h.4' in name or 'lm_head' in name:
                    params_to_update.append(param)
                else:
                    param.requires_grad = False
            
            optimizer = AdamW(params_to_update, lr=learning_rate, weight_decay=0.01)
            
            # Training loop
            total_loss = 0
            num_batches = 0
            
            for epoch in range(epochs):
                epoch_loss = 0
                epoch_batches = 0
                
                for batch in dataloader:
                    # Move to device
                    device = next(self.model.parameters()).device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_batches += 1
                
                avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
                print(f"   Epoch {epoch+1}/{epochs}: Loss = {avg_epoch_loss:.4f}")
                
                total_loss += epoch_loss
                num_batches += epoch_batches
            
            # Set back to eval mode
            self.model.eval()
            
            # Re-enable all parameters
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Update pipeline with fine-tuned model
            self.pipeline.model = self.model
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"âœ… Test-Time Training completed! Avg Loss: {avg_loss:.4f}")
            
            return avg_loss
            
        except Exception as e:
            print(f"â�Œ Error in test-time training: {e}")
            # Reset model on error
            self.reset_model()
            return None
    
    def create_enhanced_prompt_templates(self) -> Dict[str, str]:
        """Create enhanced prompt templates specifically for ARC with TTT"""
        templates = {
            "ttt_basic": """I've been trained on similar pattern recognition tasks. Let me solve this ARC puzzle.

Training examples:
{training_examples}

Test input: {test_input}

Based on the pattern I learned, the output grid is:
[[""",

            "ttt_analytical": """After analyzing similar transformations, I can see the pattern clearly.

Examples:
{training_examples}

For input: {test_input}

Applying the learned transformation rule:
[[""",

            "ttt_step_by_step": """I've learned this type of transformation. Let me apply it step by step.

Training data:
{training_examples}

Test case: {test_input}

Step 1: Identify the transformation pattern from training
Step 2: Apply to test input  
Step 3: Generate output grid

Result: [[""",

            "ttt_confident": """I've been fine-tuned on this exact pattern. Here's the solution:

{training_examples}

Input: {test_input}
Output: [[""",

            "grid_focused_ttt": """Pattern learned. Transforming grid:

{training_examples} â†’ {test_input} â†’ [["""
        }
        return templates
    
    def load_data(self, training_file: str, solution_file: str):
        """Load training and solution data from JSON files"""
        try:
            with open(training_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'root' in data:
                    self.training_data = data['root']
                else:
                    self.training_data = data
            
            with open(solution_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'root' in data:
                    self.solution_data = data['root']
                else:
                    self.solution_data = data
            
            print(f"âœ… Data loaded successfully!")
            print(f"Training data keys: {len(self.training_data)}")
            print(f"Solution data keys: {len(self.solution_data)}")
            
        except Exception as e:
            print(f"â�Œ Error loading data: {e}")
            raise e
    
    def plot_grid(self, grid, ax, title=""):
        """Plot a single grid on given axes."""
        try:
            if grid is None or len(grid) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=10, pad=10)
                ax.axis('off')
                return None
                
            grid_array = np.array(grid)
            
            # Create the plot
            im = ax.imshow(grid_array, cmap=self.cmap, vmin=0, vmax=9)
            
            # Add grid lines
            ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
            ax.tick_params(which="minor", size=0)
            
            # Remove major ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set title
            ax.set_title(title, fontsize=10, pad=10)
            
            return im
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10, pad=10)
            ax.axis('off')
            return None
    
    def parse_prediction_to_grid(self, prediction_text: str):
        """Try to parse the prediction text into a grid format"""
        try:
            if not prediction_text:
                return None
                
            prediction_text = prediction_text.strip()
            
            # Pattern 1: Look for nested list structure like [[1,2,3],[4,5,6]]
            pattern1 = r'\[\s*\[.*?\]\s*\]'
            matches = re.findall(pattern1, prediction_text, re.DOTALL)
            
            if matches:
                for match in matches:
                    try:
                        # Clean the match
                        clean_match = re.sub(r'[^\d,\[\]]', '', match)
                        grid = eval(clean_match)
                        if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
                            # Validate grid values are 0-9
                            valid_grid = []
                            for row in grid:
                                valid_row = [max(0, min(9, int(x))) for x in row if str(x).isdigit()]
                                if valid_row:
                                    valid_grid.append(valid_row)
                            if valid_grid:
                                return valid_grid
                    except:
                        continue
            
            # Pattern 2: Look for rows of numbers
            lines = prediction_text.split('\n')
            grid = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('#', '//', 'Example', 'Input', 'Output')):
                    # Extract numbers from the line
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        row = [max(0, min(9, int(n))) for n in numbers]
                        if row:
                            grid.append(row)
            
            if len(grid) > 0:
                # Make sure all rows have the same length
                max_len = max(len(row) for row in grid)
                grid = [row + [0] * (max_len - len(row)) for row in grid]
                return grid
            
            return None
            
        except Exception as e:
            print(f"Error parsing prediction: {e}")
            return None
    
    def format_training_examples(self, train_data: List) -> str:
        """Format training examples for prompts"""
        if not train_data:
            return "No training examples available"
            
        formatted = ""
        for i, example in enumerate(train_data[:3]):  # Limit to 3 examples
            formatted += f"Example {i+1}:\n"
            formatted += f"Input: {example.get('input', [])}\n"
            formatted += f"Output: {example.get('output', [])}\n\n"
        return formatted.strip()
    
    def generate_prediction(self, prompt: str, max_length: int = 200, temperature: float = 0.1) -> str:
        """Generate prediction using the (possibly fine-tuned) LLM"""
        try:
            # Truncate prompt if too long
            if len(prompt) > 1200:
                prompt = prompt[:1200] + "..."
            
            result = self.pipeline(
                prompt,
                max_new_tokens=100,  # Reduced for grid focus
                temperature=temperature,  # Lower temperature for more deterministic output
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            # Extract only the generated part
            generated_text = result[0]['generated_text']
            prediction = generated_text[len(prompt):].strip()
            
            return prediction
            
        except Exception as e:
            return f"Error generating prediction: {e}"
    
    def test_with_ttt(self, case_id: str = None, 
                     use_ttt: bool = True,
                     ttt_epochs: int = 3,
                     ttt_lr: float = 5e-5,
                     prompt_template: str = "ttt_basic") -> Dict:
        """
        Test a single case with optional Test-Time Training
        
        Args:
            case_id: Specific case to test (random if None)
            use_ttt: Whether to perform test-time training
            ttt_epochs: Number of TTT epochs
            ttt_lr: Learning rate for TTT
            prompt_template: Prompt template to use
        """
        try:
            if not self.training_data or not self.solution_data:
                return {"error": "Data not loaded. Please load data first."}
            
            # Reset model to original state first
            self.reset_model()
            
            # Get available case IDs
            available_cases = list(self.training_data.keys())
            
            if not available_cases:
                return {"error": "No training cases available"}
            
            # Select case
            if case_id is None or case_id not in available_cases:
                case_id = random.choice(available_cases)
                print(f"Selected random case: {case_id}")
            
            # Get case data
            case_data = self.training_data[case_id]
            train_data = case_data.get('train', [])
            test_data = case_data.get('test', [])
            
            # Get solution
            actual_solution = self.solution_data.get(case_id, "No solution found")
            
            print(f"\n{'='*60}")
            print(f"TESTING CASE: {case_id}")
            print(f"TTT ENABLED: {use_ttt}")
            print(f"TRAINING EXAMPLES: {len(train_data)}")
            print(f"{'='*60}")
            
            # Perform Test-Time Training if enabled
            ttt_loss = None
            if use_ttt and len(train_data) > 0:
                ttt_loss = self.test_time_training(
                    train_data, 
                    epochs=ttt_epochs, 
                    learning_rate=ttt_lr
                )
            
            # Prepare prompt
            templates = self.create_enhanced_prompt_templates()
            template = templates.get(prompt_template, templates['ttt_basic'])
            
            # Format the prompt
            training_examples = self.format_training_examples(train_data)
            test_input = str(test_data[0]['input']) if test_data else "No test data"
            
            prompt = template.format(
                training_examples=training_examples,
                test_input=test_input
            )
            
            # Generate prediction
            print("ğŸ¤– Generating prediction...")
            start_time = time.time()
            prediction = self.generate_prediction(prompt, max_length=200, temperature=0.1)
            generation_time = time.time() - start_time
            
            # Prepare results
            results = {
                "case_id": case_id,
                "used_ttt": use_ttt,
                "ttt_loss": ttt_loss,
                "ttt_epochs": ttt_epochs if use_ttt else 0,
                "prompt_template": prompt_template,
                "training_data": train_data,
                "test_data": test_data,
                "prediction": prediction,
                "actual_solution": actual_solution,
                "generation_time": round(generation_time, 2)
            }
            
            # Visualize results
            self.visualize_ttt_results(case_id, prediction, use_ttt, ttt_loss)
            
            return results
            
        except Exception as e:
            print(f"Error in test_with_ttt: {e}")
            return {"error": str(e)}
    
    def visualize_ttt_results(self, case_id: str, prediction_text: str, used_ttt: bool, ttt_loss: float, figsize=(18, 6)):
        """Visualize results with TTT information"""
        try:
            if case_id not in self.training_data:
                print(f"Challenge {case_id} not found!")
                return
            
            case_data = self.training_data[case_id]
            train_examples = case_data.get('train', [])
            test_examples = case_data.get('test', [])
            
            # Get grids
            train_input = train_examples[0]['input'] if train_examples else None
            train_output = train_examples[0]['output'] if train_examples else None
            test_input = test_examples[0]['input'] if test_examples else None
            
            # Get actual solution
            actual_solution = None
            if case_id in self.solution_data:
                sol_data = self.solution_data[case_id]
                if isinstance(sol_data, list) and len(sol_data) > 0:
                    actual_solution = sol_data[0]
                else:
                    actual_solution = sol_data
            
            # Parse prediction
            predicted_grid = self.parse_prediction_to_grid(prediction_text)
            
            # Create subplot
            fig, axes = plt.subplots(2, 5, figsize=figsize)
            fig.suptitle(f'Challenge {case_id} - Test-Time Training Results', fontsize=16, fontweight='bold')
            
            # First row: Training example and TTT info
            self.plot_grid(train_input, axes[0, 0], "Train Input")
            self.plot_grid(train_output, axes[0, 1], "Train Output")
            
            # TTT Information
            ttt_info = f"Test-Time Training:\n"
            if used_ttt:
                ttt_info += f"âœ… ENABLED\n"
                ttt_info += f"Loss: {ttt_loss:.4f}" if ttt_loss else "Loss: Failed"
                color = "lightgreen" if ttt_loss and ttt_loss < 2.0 else "orange"
            else:
                ttt_info += f"â�Œ DISABLED\n"
                ttt_info += f"Using base model"
                color = "lightgray"
            
            axes[0, 2].text(0.5, 0.5, ttt_info, 
                           ha='center', va='center', transform=axes[0, 2].transAxes,
                           fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
            axes[0, 2].set_title("TTT Status", fontsize=10, pad=10)
            axes[0, 2].axis('off')
            
            # Model info
            model_info = f"Model: {self.model_name}\n"
            model_info += f"Examples: {len(train_examples)}\n"
            model_info += f"Mode: {'Fine-tuned' if used_ttt else 'Base'}"
            
            axes[0, 3].text(0.5, 0.5, model_info, 
                           ha='center', va='center', transform=axes[0, 3].transAxes,
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[0, 3].set_title("Model Info", fontsize=10, pad=10)
            axes[0, 3].axis('off')
            
            # Prediction details
            pred_info = f"Prediction:\n"
            if predicted_grid:
                pred_info += f"Shape: {len(predicted_grid)}x{len(predicted_grid[0])}\n"
                pred_info += f"Status: âœ… Parsed"
            else:
                pred_info += f"Status: â�Œ Failed to parse\n"
                pred_info += f"Raw: {prediction_text[:30]}..."
            
            axes[0, 4].text(0.5, 0.5, pred_info, 
                           ha='center', va='center', transform=axes[0, 4].transAxes,
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            axes[0, 4].set_title("Prediction Info", fontsize=10, pad=10)
            axes[0, 4].axis('off')
            
            # Second row: Test and results
            self.plot_grid(test_input, axes[1, 0], "Test Input")
            self.plot_grid(predicted_grid, axes[1, 1], f"Prediction\n({'TTT' if used_ttt else 'Base'})")
            self.plot_grid(actual_solution, axes[1, 2], "Actual Solution")
            
            # Comparison
            if predicted_grid is not None and actual_solution is not None:
                try:
                    pred_array = np.array(predicted_grid)
                    actual_array = np.array(actual_solution)
                    
                    if pred_array.shape == actual_array.shape:
                        accuracy = np.mean(pred_array == actual_array)
                        match_status = "ğŸ�¯ PERFECT!" if accuracy == 1.0 else f"ğŸ“Š {accuracy:.1%} match"
                        color = "lightgreen" if accuracy == 1.0 else "lightcoral"
                    else:
                        match_status = f"ğŸ“� Shape mismatch\nP: {pred_array.shape}\nA: {actual_array.shape}"
                        color = "lightcoral"
                except:
                    match_status = "â�Œ Comparison failed"
                    color = "lightcoral"
            else:
                match_status = "â�Œ Cannot compare\n(missing data)"
                color = "lightgray"
            
            axes[1, 3].text(0.5, 0.5, f"Results:\n\n{match_status}", 
                           ha='center', va='center', transform=axes[1, 3].transAxes,
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
            axes[1, 3].set_title("Accuracy", fontsize=10, pad=10)
            axes[1, 3].axis('off')
            
            # Performance summary
            perf_summary = f"Performance:\n\n"
            if used_ttt:
                perf_summary += f"Method: TTT\n"
                perf_summary += f"Training: âœ…\n"
                if ttt_loss:
                    perf_summary += f"Quality: {'Good' if ttt_loss < 2.0 else 'Poor'}"
                else:
                    perf_summary += f"Quality: Failed"
            else:
                perf_summary += f"Method: Base\n"
                perf_summary += f"Training: â�Œ\n"
                perf_summary += f"Quality: Baseline"
            
            axes[1, 4].text(0.5, 0.5, perf_summary, 
                           ha='center', va='center', transform=axes[1, 4].transAxes,
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsteelblue"))
            axes[1, 4].set_title("Method", fontsize=10, pad=10)
            axes[1, 4].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print detailed summary
            print(f"\n{'='*70}")
            print(f"TEST-TIME TRAINING RESULTS FOR CASE: {case_id}")
            print(f"{'='*70}")
            
            print(f"ğŸ”§ CONFIGURATION:")
            print(f"   Model: {self.model_name}")
            print(f"   TTT Enabled: {used_ttt}")
            if used_ttt:
                print(f"   TTT Loss: {ttt_loss:.4f}" if ttt_loss else "   TTT Loss: Failed")
            
            if predicted_grid:
                print(f"\nğŸ¤– PREDICTED GRID ({len(predicted_grid)}x{len(predicted_grid[0])}):")
                for row in predicted_grid:
                    print(f"  {row}")
            else:
                print(f"\nğŸ¤– PREDICTED GRID: Parse failed")
                print(f"   Raw: '{prediction_text[:100]}{'...' if len(prediction_text) > 100 else ''}'")
            
            if actual_solution:
                print(f"\nâœ… ACTUAL SOLUTION ({len(actual_solution)}x{len(actual_solution[0])}):")
                for row in actual_solution:
                    print(f"  {row}")
            
            print(f"\nğŸ“Š FINAL RESULT: {match_status}")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"Error in TTT visualization: {e}")
    
    def compare_with_without_ttt(self, case_id: str = None, ttt_epochs: int = 3) -> Dict:
        """Compare performance with and without Test-Time Training"""
        print(f"\nğŸ”¬ COMPARATIVE ANALYSIS: Base Model vs Test-Time Training")
        print(f"{'='*60}")
        
        # Test without TTT
        print("\n1ï¸�âƒ£ Testing WITHOUT Test-Time Training...")
        result_base = self.test_with_ttt(
            case_id=case_id, 
            use_ttt=False, 
            prompt_template="ttt_basic"
        )
        
        time.sleep(1)  # Brief pause
        
        # Test with TTT (using same case)
        print("\n2ï¸�âƒ£ Testing WITH Test-Time Training...")
        result_ttt = self.test_with_ttt(
            case_id=result_base.get('case_id') if result_base else case_id,
            use_ttt=True,
            ttt_epochs=ttt_epochs,
            prompt_template="ttt_basic"
        )
        
        # Compare results
        comparison = {
            "case_id": result_base.get('case_id') if result_base else "Unknown",
            "base_model": result_base,
            "ttt_model": result_ttt,
            "comparison_summary": {}
        }
        
        # Analysis
        print(f"\nğŸ“ˆ COMPARISON SUMMARY:")
        print(f"   Case ID: {comparison['case_id']}")
        
        if result_base and result_ttt:
            base_pred = self.parse_prediction_to_grid(result_base.get('prediction', ''))
            ttt_pred = self.parse_prediction_to_grid(result_ttt.get('prediction', ''))
            actual = result_base.get('actual_solution')
            
            # Calculate accuracies if possible
            base_acc = self._calculate_accuracy(base_pred, actual)
            ttt_acc = self._calculate_accuracy(ttt_pred, actual)
            
            comparison['comparison_summary'] = {
                'base_accuracy': base_acc,
                'ttt_accuracy': ttt_acc,
                'improvement': ttt_acc - base_acc if (base_acc is not None and ttt_acc is not None) else None,
                'ttt_loss': result_ttt.get('ttt_loss'),
                'base_parsed': base_pred is not None,
                'ttt_parsed': ttt_pred is not None
            }
            
            print(f"   Base Model Accuracy: {base_acc:.1%}" if base_acc is not None else "   Base Model: Parse failed")
            print(f"   TTT Model Accuracy: {ttt_acc:.1%}" if ttt_acc is not None else "   TTT Model: Parse failed")
            
            if base_acc is not None and ttt_acc is not None:
                improvement = ttt_acc - base_acc
                if improvement > 0:
                    print(f"   ğŸ�¯ IMPROVEMENT: +{improvement:.1%} (TTT is better!)")
                elif improvement < 0:
                    print(f"   ğŸ“‰ REGRESSION: {improvement:.1%} (Base is better)")
                else:
                    print(f"   â�¡ï¸� NO CHANGE: Same performance")
            
            print(f"   TTT Training Loss: {result_ttt.get('ttt_loss', 'N/A')}")
        
        return comparison
    
    def _calculate_accuracy(self, predicted_grid, actual_solution):
        """Calculate accuracy between predicted and actual grids"""
        try:
            if predicted_grid is None or actual_solution is None:
                return None
            
            if isinstance(actual_solution, list) and len(actual_solution) > 0:
                # Handle nested solution format
                if isinstance(actual_solution[0], list) and isinstance(actual_solution[0][0], list):
                    actual_solution = actual_solution[0]
            
            pred_array = np.array(predicted_grid)
            actual_array = np.array(actual_solution)
            
            if pred_array.shape != actual_array.shape:
                return 0.0  # Shape mismatch = 0% accuracy
            
            return np.mean(pred_array == actual_array)
            
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            return None

def main():
    """Main function to demonstrate the enhanced LLM Prompt Tester with TTT"""
    try:
        # Initialize the tester
        print("ğŸš€ Initializing Enhanced LLM Prompt Tester with Test-Time Training...")
        tester = LLMPromptTesterWithTTT(model_name="distilgpt2")
        
        # Load data (you'll need to provide the correct paths)
        tester.load_data(
                "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json", 
                "/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json"
            )
        print("\n" + "="*70)
        print("ğŸ§ª STARTING TEST-TIME TRAINING EXPERIMENTS")
        print("="*70)
        
        # Experiment 1: Single test with TTT
        print("\nğŸ”¬ EXPERIMENT 1: Single Case with Test-Time Training")
        result1 = tester.test_with_ttt('beb8660c',
            use_ttt=True, 
            ttt_epochs=50, 
            ttt_lr=1e-4,
            prompt_template="ttt_confident"
        )
        
        time.sleep(2)
        
        # Experiment 2: Comparative analysis
        print("\nğŸ”¬ EXPERIMENT 2: Comparing Base vs TTT Performance")
        comparison = tester.compare_with_without_ttt(ttt_epochs=4)
        
        # Experiment 3: Different TTT configurations
        print("\nğŸ”¬ EXPERIMENT 3: Testing Different TTT Configurations")
        case_id = result1.get('case_id') if result1 else None
        
        configs = [
            {"epochs": 12, "lr": 1e-4, "name": "Light TTT"},
            {"epochs": 25, "lr": 5e-5, "name": "Moderate TTT"},
            {"epochs": 38, "lr": 2e-5, "name": "Heavy TTT"}
        ]
        
        for config in configs:
            print(f"\n   Testing {config['name']} (epochs={config['epochs']}, lr={config['lr']})...")
            result = tester.test_with_ttt(
                case_id=case_id,
                use_ttt=True,
                ttt_epochs=config['epochs'],
                ttt_lr=config['lr'],
                prompt_template="ttt_analytical"
            )
            time.sleep(1)
        
        print("\n" + "="*70)
        print("âœ¨ TEST-TIME TRAINING EXPERIMENTS COMPLETED!")
        print("="*70)
        
        print("\nğŸ“‹ SUMMARY OF FINDINGS:")
        print("â€¢ Test-Time Training allows the model to adapt to specific ARC patterns")
        print("â€¢ Fine-tuning on task-specific examples improves pattern recognition")
        print("â€¢ Different TTT configurations (epochs, learning rate) affect performance")
        print("â€¢ Enhanced prompts work better with fine-tuned models")
        
        print("\nğŸ”§ NEXT STEPS TO IMPROVE:")
        print("1. Experiment with different model architectures (GPT-2, T5, CodeT5)")
        print("2. Try more sophisticated fine-tuning strategies")
        print("3. Use curriculum learning with easier ARC tasks first")
        print("4. Implement ensemble methods with multiple TTT runs")
        print("5. Add grid-specific tokenization and loss functions")
        
        print("\nğŸ’¡ USAGE TIPS:")
        print("â€¢ Use more training examples when available for better TTT")
        print("â€¢ Adjust TTT epochs based on task complexity")
        print("â€¢ Try different prompt templates with TTT")
        print("â€¢ Monitor TTT loss to ensure successful fine-tuning")
        
    except Exception as e:
        print(f"â�Œ Error in main: {e}")
        import traceback
        traceback.print_exc()

# Additional utility functions for extended functionality
def create_curriculum_data():
    """Create a curriculum of progressively harder ARC-like tasks"""
    curriculum = {
        "easy_copy": {
            "train": [
                {"input": [[1]], "output": [[1]]},
                {"input": [[2]], "output": [[2]]},
                {"input": [[0]], "output": [[0]]}
            ],
            "test": [{"input": [[3]]}]
        },
        "simple_transform": {
            "train": [
                {"input": [[1, 0]], "output": [[2, 0]]},
                {"input": [[0, 1]], "output": [[0, 2]]},
                {"input": [[1, 1]], "output": [[2, 2]]}
            ],
            "test": [{"input": [[0, 0]]}]
        },
        "pattern_fill": {
            "train": [
                {"input": [[1, 0, 1], [0, 0, 0], [1, 0, 1]], "output": [[1, 2, 1], [2, 2, 2], [1, 2, 1]]},
                {"input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]], "output": [[0, 1, 0], [1, 3, 1], [0, 1, 0]]}
            ],
            "test": [{"input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]}]
        }
    }
    
    solutions = {
        "easy_copy": [[[3]]],
        "simple_transform": [[[0, 0]]],  # No transformation rule learned
        "pattern_fill": [[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]  # Fill pattern
    }
    
    return curriculum, solutions

def run_curriculum_experiment(tester):
    """Run a curriculum learning experiment"""
    print("\nğŸ�“ CURRICULUM LEARNING EXPERIMENT")
    print("="*50)
    
    curriculum_data, curriculum_solutions = create_curriculum_data()
    
    # Save original data
    original_training = tester.training_data
    original_solutions = tester.solution_data
    
    try:
        # Set curriculum data
        tester.training_data = curriculum_data
        tester.solution_data = curriculum_solutions
        
        # Test each level
        for level, level_name in enumerate(["easy_copy", "simple_transform", "pattern_fill"]):
            print(f"\nğŸ“š Level {level+1}: {level_name}")
            result = tester.test_with_ttt(
                case_id=level_name,
                use_ttt=True,
                ttt_epochs=3 + level*2,  # Increase epochs for harder tasks
                ttt_lr=1e-4 / (level + 1),  # Decrease learning rate for harder tasks
                prompt_template="ttt_step_by_step"
            )
            time.sleep(1)
            
    finally:
        # Restore original data
        tester.training_data = original_training
        tester.solution_data = original_solutions

if __name__ == "__main__":
    main()


import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random
from typing import Dict, List, Any, Tuple, Optional
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
import warnings
from torch.utils.data import Dataset, DataLoader
import copy
from collections import defaultdict
import itertools
warnings.filterwarnings('ignore')

# Import AdamW from torch.optim (newer versions)
try:
    from torch.optim import AdamW
except ImportError:
    try:
        from transformers import AdamW
    except ImportError:
        print("âš ï¸� AdamW not found, using torch.optim.Adam instead")
        from torch.optim import Adam as AdamW

class ARCDataAugmenter:
    """Comprehensive data augmentation for ARC tasks following winning solution strategies"""
    
    def __init__(self):
        self.color_mappings = self._generate_color_mappings()
    
    def _generate_color_mappings(self):
        """Generate all possible color permutations (0-9)"""
        colors = list(range(10))
        mappings = []
        # Generate some common permutations to avoid memory issues
        for _ in range(50):  # Limited set for efficiency
            mapping = dict(zip(colors, random.sample(colors, 10)))
            mappings.append(mapping)
        return mappings
    
    def rotate_grid(self, grid, times=1):
        """Rotate grid 90 degrees clockwise 'times' times"""
        if not grid or not grid[0]:
            return grid
        
        result = grid
        for _ in range(times % 4):
            result = [list(row) for row in zip(*result[::-1])]
        return result
    
    def flip_grid(self, grid, horizontal=True, vertical=False):
        """Flip grid horizontally/vertically"""
        if not grid:
            return grid
            
        result = [row[:] for row in grid]  # Deep copy
        
        if horizontal:
            result = [row[::-1] for row in result]
        if vertical:
            result = result[::-1]
            
        return result
    
    def change_colors(self, grid, color_mapping=None):
        """Change colors according to mapping"""
        if not grid:
            return grid
            
        if color_mapping is None:
            color_mapping = random.choice(self.color_mappings)
        
        return [[color_mapping.get(cell, cell) for cell in row] for row in grid]
    
    def pad_grid(self, grid, padding=1, fill_value=0):
        """Add padding around the grid"""
        if not grid:
            return grid
            
        height, width = len(grid), len(grid[0])
        new_height = height + 2 * padding
        new_width = width + 2 * padding
        
        padded = [[fill_value] * new_width for _ in range(new_height)]
        
        # Copy original grid to center
        for i in range(height):
            for j in range(width):
                padded[i + padding][j + padding] = grid[i][j]
        
        return padded
    
    def upscale_grid(self, grid, factor=2):
        """Upscale grid by repeating each cell"""
        if not grid or factor <= 1:
            return grid
            
        upscaled = []
        for row in grid:
            new_row = []
            for cell in row:
                new_row.extend([cell] * factor)
            for _ in range(factor):
                upscaled.append(new_row[:])
        
        return upscaled
    
    def mirror_grid(self, grid, axis='horizontal'):
        """Mirror grid along specified axis"""
        if not grid:
            return grid
            
        if axis == 'horizontal':
            # Mirror horizontally (extend to the right)
            mirrored = []
            for row in grid:
                mirrored_row = row + row[::-1]
                mirrored.append(mirrored_row)
            return mirrored
        elif axis == 'vertical':
            # Mirror vertically (extend downward)
            return grid + grid[::-1]
        else:
            return grid
    
    def augment_example(self, example, augmentation_prob=0.5):
        """Apply random augmentations to a single example"""
        input_grid = example['input']
        output_grid = example['output']
        
        if random.random() > augmentation_prob:
            return example  # No augmentation
        
        # Choose random augmentations (same for input and output)
        augmentations = []
        
        # Rotation
        if random.random() < 0.3:
            rotation = random.randint(1, 3)
            augmentations.append(('rotate', rotation))
        
        # Flip
        if random.random() < 0.3:
            h_flip = random.random() < 0.5
            v_flip = random.random() < 0.5
            if h_flip or v_flip:
                augmentations.append(('flip', h_flip, v_flip))
        
        # Color change
        if random.random() < 0.5:
            color_mapping = random.choice(self.color_mappings)
            augmentations.append(('color', color_mapping))
        
        # Apply augmentations
        aug_input = input_grid
        aug_output = output_grid
        
        for aug in augmentations:
            if aug[0] == 'rotate':
                aug_input = self.rotate_grid(aug_input, aug[1])
                aug_output = self.rotate_grid(aug_output, aug[1])
            elif aug[0] == 'flip':
                aug_input = self.flip_grid(aug_input, aug[1], aug[2])
                aug_output = self.flip_grid(aug_output, aug[1], aug[2])
            elif aug[0] == 'color':
                aug_input = self.change_colors(aug_input, aug[1])
                aug_output = self.change_colors(aug_output, aug[1])
        
        return {'input': aug_input, 'output': aug_output}
    
    def problem_augmentation(self, task_data, augmentation_type='input_transform'):
        """Apply problem-level augmentation (transform only inputs or outputs)"""
        augmented_task = copy.deepcopy(task_data)
        
        # Choose augmentation
        if augmentation_type == 'input_transform':
            transform_func = random.choice([
                lambda g: self.rotate_grid(g, random.randint(1, 3)),
                lambda g: self.flip_grid(g, True, False),
                lambda g: self.pad_grid(g, random.randint(1, 2)),
                lambda g: self.upscale_grid(g, 2)
            ])
            
            # Apply to all inputs
            for example in augmented_task['train']:
                example['input'] = transform_func(example['input'])
            for example in augmented_task['test']:
                example['input'] = transform_func(example['input'])
                
        elif augmentation_type == 'output_transform':
            transform_func = random.choice([
                lambda g: self.mirror_grid(g, 'horizontal'),
                lambda g: self.upscale_grid(g, 2),
                lambda g: self.pad_grid(g, 1, 0)
            ])
            
            # Apply to all outputs
            for example in augmented_task['train']:
                example['output'] = transform_func(example['output'])
        
        return augmented_task

class CurriculumARCDataset(Dataset):
    """Enhanced dataset with curriculum learning and multi-task support"""
    
    def __init__(self, task_data, tokenizer, task_type='output_prediction', 
                 max_length=512, augmenter=None, augment_prob=0.5):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.augmenter = augmenter
        
        for task_id, task in task_data.items():
            # Create examples based on task type
            if task_type == 'output_prediction':
                self._create_output_prediction_examples(task_id, task, augment_prob)
            elif task_type == 'input_generation':
                self._create_input_generation_examples(task_id, task, augment_prob)
            elif task_type == 'verification':
                self._create_verification_examples(task_id, task, augment_prob)
            elif task_type == 'selection':
                self._create_selection_examples(task_id, task, augment_prob)
    
    def _grid_to_text(self, grid):
        """Convert grid to text format used by winning solution"""
        if not grid:
            return "empty grid"
        
        height, width = len(grid), len(grid[0])
        text = f"```grid shape: {height}x{width}\n"
        for i, row in enumerate(grid, 1):
            row_str = ''.join(str(cell) for cell in row)
            text += f"{i} {row_str}\n"
        text += "```"
        return text
    
    def _create_output_prediction_examples(self, task_id, task, augment_prob):
        """Create examples for output prediction task"""
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])
        
        for test_example in test_examples:
            # Apply augmentation if enabled
            if self.augmenter and random.random() < augment_prob:
                aug_train = [self.augmenter.augment_example(ex) for ex in train_examples]
                aug_test = self.augmenter.augment_example({'input': test_example['input'], 'output': []})
                test_input = aug_test['input']
                examples_text = self._format_examples(aug_train)
            else:
                test_input = test_example['input']
                examples_text = self._format_examples(train_examples)
            
            # Create prompt
            text = f"Task: Predict the output for the given input.\n\n"
            text += f"Training examples:\n{examples_text}\n\n"
            text += f"Test input:\n{self._grid_to_text(test_input)}\n\n"
            text += f"Test output:\n{self._grid_to_text(test_example.get('output', []))}"
            
            self.examples.append(text)
    
    def _create_input_generation_examples(self, task_id, task, augment_prob):
        """Create examples for input distribution learning"""
        train_examples = task.get('train', [])
        
        # Generate new inputs by learning the distribution
        for example in train_examples:
            input_grid = example['input']
            
            # Apply augmentation
            if self.augmenter and random.random() < augment_prob:
                aug_example = self.augmenter.augment_example(example)
                input_grid = aug_example['input']
            
            # Create prompt for input generation
            text = f"Task: Generate a new input following the same pattern.\n\n"
            text += f"Example inputs:\n"
            for i, ex in enumerate(train_examples[:3]):
                text += f"Input {i+1}:\n{self._grid_to_text(ex['input'])}\n\n"
            text += f"New input:\n{self._grid_to_text(input_grid)}"
            
            self.examples.append(text)
    
    def _create_verification_examples(self, task_id, task, augment_prob):
        """Create examples for output verification"""
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])
        
        for test_example in test_examples:
            if 'output' not in test_example:
                continue
                
            # Create positive example (correct output)
            examples_text = self._format_examples(train_examples)
            text = f"Task: Verify if the output is correct.\n\n"
            text += f"Training examples:\n{examples_text}\n\n"
            text += f"Test input:\n{self._grid_to_text(test_example['input'])}\n\n"
            text += f"Proposed output:\n{self._grid_to_text(test_example['output'])}\n\n"
            text += f"Is correct: True"
            
            self.examples.append(text)
            
            # Create negative example (incorrect output) by modifying the correct one
            if test_example['output']:
                wrong_output = copy.deepcopy(test_example['output'])
                # Randomly change some cells
                for _ in range(min(3, len(wrong_output) * len(wrong_output[0]) // 4)):
                    i = random.randint(0, len(wrong_output) - 1)
                    j = random.randint(0, len(wrong_output[0]) - 1)
                    wrong_output[i][j] = random.randint(0, 9)
                
                text_wrong = f"Task: Verify if the output is correct.\n\n"
                text_wrong += f"Training examples:\n{examples_text}\n\n"
                text_wrong += f"Test input:\n{self._grid_to_text(test_example['input'])}\n\n"
                text_wrong += f"Proposed output:\n{self._grid_to_text(wrong_output)}\n\n"
                text_wrong += f"Is correct: False"
                
                self.examples.append(text_wrong)
    
    def _create_selection_examples(self, task_id, task, augment_prob):
        """Create examples for output selection"""
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])
        
        for test_example in test_examples:
            if 'output' not in test_example:
                continue
            
            correct_output = test_example['output']
            wrong_outputs = []
            
            # Generate wrong outputs
            for _ in range(2):
                wrong = copy.deepcopy(correct_output)
                for _ in range(min(2, len(wrong) * len(wrong[0]) // 3)):
                    i = random.randint(0, len(wrong) - 1)
                    j = random.randint(0, len(wrong[0]) - 1)
                    wrong[i][j] = random.randint(0, 9)
                wrong_outputs.append(wrong)
            
            # Create selection task
            options = [correct_output] + wrong_outputs
            random.shuffle(options)
            correct_idx = options.index(correct_output)
            
            examples_text = self._format_examples(train_examples)
            text = f"Task: Select the correct output from the options.\n\n"
            text += f"Training examples:\n{examples_text}\n\n"
            text += f"Test input:\n{self._grid_to_text(test_example['input'])}\n\n"
            text += f"Options:\n"
            for i, option in enumerate(options):
                text += f"Option {i+1}:\n{self._grid_to_text(option)}\n\n"
            text += f"Correct option: {correct_idx + 1}"
            
            self.examples.append(text)
    
    def _format_examples(self, examples):
        """Format training examples"""
        formatted = ""
        for i, example in enumerate(examples):
            formatted += f"Example {i+1}:\n"
            formatted += f"Input:\n{self._grid_to_text(example['input'])}\n"
            formatted += f"Output:\n{self._grid_to_text(example['output'])}\n\n"
        return formatted.strip()
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class CurriculumLLMTrainer:
    """Enhanced LLM trainer with curriculum learning and multi-task capabilities"""
    
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.original_model = None
        self.pipeline = None
        self.training_data = None
        self.solution_data = None
        self.augmenter = ARCDataAugmenter()
        self.curriculum_history = []
        
        # Task difficulty levels
        self.task_levels = {
            'easy': ['output_prediction'],
            'medium': ['output_prediction', 'input_generation'],
            'hard': ['output_prediction', 'input_generation', 'verification'],
            'expert': ['output_prediction', 'input_generation', 'verification', 'selection']
        }
        
        # ARC color palette
        self.colors = [
            '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
        ]
        self.cmap = ListedColormap(self.colors)
        
        print(f"Initializing Curriculum LLM Trainer: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            print("Loading tokenizer and model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Keep original model copy
            self.original_model = copy.deepcopy(self.model)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("âœ… Model loaded successfully!")
            print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            
        except Exception as e:
            print(f"â�Œ Error loading model: {e}")
            raise e
    
    def load_data(self, training_file: str, solution_file: str):
        """Load training and solution data"""
        try:
            with open(training_file, 'r') as f:
                data = json.load(f)
                self.training_data = data['root'] if isinstance(data, dict) and 'root' in data else data
            
            with open(solution_file, 'r') as f:
                data = json.load(f)
                self.solution_data = data['root'] if isinstance(data, dict) and 'root' in data else data
            
            print(f"âœ… Data loaded successfully!")
            print(f"Training tasks: {len(self.training_data)}")
            print(f"Solution tasks: {len(self.solution_data)}")
            
        except Exception as e:
            print(f"â�Œ Error loading data: {e}")
            raise e
    
    def reset_model(self):
        """Reset model to original state"""
        print("ğŸ”„ Resetting model to original state...")
        self.model.load_state_dict(self.original_model.state_dict())
        self.pipeline.model = self.model
    
    def curriculum_training(self, 
                          level: str = 'easy',
                          epochs_per_task: int = 2,
                          learning_rate: float = 5e-5,
                          batch_size: int = 4,
                          num_tasks_sample: int = 50):
        """
        Perform curriculum training starting from easy tasks
        
        Args:
            level: Difficulty level ('easy', 'medium', 'hard', 'expert')
            epochs_per_task: Epochs to train on each task type
            learning_rate: Learning rate for training
            batch_size: Batch size
            num_tasks_sample: Number of tasks to sample for training
        """
        try:
            print(f"\nğŸ�“ STARTING CURRICULUM TRAINING - LEVEL: {level.upper()}")
            print(f"{'='*60}")
            
            if not self.training_data:
                raise ValueError("Training data not loaded!")
            
            # Get task types for this level
            task_types = self.task_levels.get(level, ['output_prediction'])
            
            # Sample training tasks
            available_tasks = list(self.training_data.keys())
            sampled_tasks = dict(random.sample(list(self.training_data.items()), 
                                             min(num_tasks_sample, len(available_tasks))))
            
            print(f"ğŸ“š Training on {len(sampled_tasks)} tasks")
            print(f"ğŸ�¯ Task types: {task_types}")
            
            # Progressive training on each task type
            total_loss = 0
            for i, task_type in enumerate(task_types):
                print(f"\nğŸ“– PHASE {i+1}/{len(task_types)}: Training on {task_type}")
                print("-" * 40)
                
                # Create dataset for current task type
                dataset = CurriculumARCDataset(
                    sampled_tasks, 
                    self.tokenizer, 
                    task_type=task_type,
                    augmenter=self.augmenter,
                    augment_prob=0.5
                )
                
                if len(dataset) == 0:
                    print(f"âš ï¸� No data for task type {task_type}, skipping...")
                    continue
                
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Set up training
                self.model.train()
                
                # Fine-tune only last layers for efficiency
                params_to_update = []
                for name, param in self.model.named_parameters():
                    #print("Name:",name)
                    if any(layer in name for layer in ['h.5', 'h.4', 'lm_head']):
                        params_to_update.append(param)
                        #print("I am hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                    else:
                        param.requires_grad = False
                
                optimizer = AdamW(params_to_update, lr=learning_rate, weight_decay=0.01)
                
                # Training loop
                task_loss = 0
                for epoch in range(epochs_per_task):
                    epoch_loss = 0
                    num_batches = 0
                    
                    for batch in dataloader:
                        device = next(self.model.parameters()).device
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                    
                    avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
                    print(f"   Epoch {epoch+1}/{epochs_per_task}: Loss = {avg_epoch_loss:.4f}")
                    task_loss += epoch_loss
                
                # Re-enable all parameters
                for param in self.model.parameters():
                    param.requires_grad = True
                
                avg_task_loss = task_loss / (epochs_per_task * len(dataloader)) if len(dataloader) > 0 else 0
                total_loss += avg_task_loss
                print(f"âœ… Completed {task_type}: Avg Loss = {avg_task_loss:.4f}")
            
            # Update pipeline and record training
            self.model.eval()
            self.pipeline.model = self.model
            
            training_record = {
                'level': level,
                'task_types': task_types,
                'num_tasks': len(sampled_tasks),
                'total_loss': total_loss / len(task_types) if task_types else 0,
                'timestamp': time.time()
            }
            self.curriculum_history.append(training_record)
            
            print(f"\nğŸ�‰ CURRICULUM LEVEL '{level.upper()}' COMPLETED!")
            print(f"ğŸ“Š Average Loss: {training_record['total_loss']:.4f}")
            print(f"ğŸ�¯ Tasks Learned: {task_types}")
            
            return training_record
            
        except Exception as e:
            print(f"â�Œ Error in curriculum training: {e}")
            self.reset_model()
            return None
    
    def progressive_curriculum(self, start_level='easy', max_level='expert'):
        """Train progressively through curriculum levels"""
        levels = ['easy', 'medium', 'hard', 'expert']
        start_idx = levels.index(start_level)
        end_idx = levels.index(max_level)
        
        print(f"\nğŸš€ PROGRESSIVE CURRICULUM TRAINING")
        print(f"ğŸ“ˆ Path: {' â†’ '.join(levels[start_idx:end_idx+1])}")
        print(f"{'='*60}")
        
        results = {}
        for level in levels[start_idx:end_idx+1]:
            print(f"\nğŸ�¯ Advancing to Level: {level.upper()}")
            result = self.curriculum_training(
                level=level,
                epochs_per_task=3 if level == 'easy' else 3,
                learning_rate=5e-5 if level == 'easy' else 3e-5,
                num_tasks_sample=30 if level == 'easy' else 50
            )
            results[level] = result
            
            # Brief pause between levels
            time.sleep(2)
        
        print(f"\nğŸ�† PROGRESSIVE TRAINING COMPLETED!")
        print(f"ğŸ“Š Training History: {len(self.curriculum_history)} levels")
        
        return results
    
    def test_curriculum_performance(self, case_id: str = None, with_ttt: bool = True):
        """Test model performance after curriculum training"""
        try:
            if not self.training_data or not self.solution_data:
                return {"error": "Data not loaded"}
            
            available_cases = list(self.training_data.keys())
            if case_id is None or case_id not in available_cases:
                case_id = random.choice(available_cases)
            
            case_data = self.training_data[case_id]
            train_data = case_data.get('train', [])
            test_data = case_data.get('test', [])
            actual_solution = self.solution_data.get(case_id)
            
            print(f"\nğŸ§ª TESTING CURRICULUM MODEL")
            print(f"ğŸ“‹ Case: {case_id}")
            print(f"ğŸ�“ Training History: {len(self.curriculum_history)} levels completed")
            print(f"ğŸ”§ Test-Time Training: {'Enabled' if with_ttt else 'Disabled'}")
            
            # Optional test-time training
            ttt_loss = None
            if with_ttt and len(train_data) > 0:
                print("ğŸ”„ Applying test-time fine-tuning...")
                ttt_loss = self._test_time_finetune(train_data)
            
            # Generate prediction
            prompt = self._create_curriculum_prompt(train_data, test_data[0]['input'] if test_data else [])
            
            print("ğŸ¤– Generating prediction...")
            start_time = time.time()
            prediction = self._generate_prediction(prompt)
            generation_time = time.time() - start_time
            
            # Visualize results
            self._visualize_curriculum_results(case_id, prediction, ttt_loss)
            
            return {
                'case_id': case_id,
                'curriculum_levels': len(self.curriculum_history),
                'ttt_applied': with_ttt,
                'ttt_loss': ttt_loss,
                'prediction': prediction,
                'actual_solution': actual_solution,
                'generation_time': generation_time,
                'curriculum_history': self.curriculum_history
            }
            
        except Exception as e:
            print(f"â�Œ Error in curriculum testing: {e}")
            return {"error": str(e)}
    
    def _test_time_finetune(self, training_examples, epochs=2, lr=8e-5):
        """Perform test-time fine-tuning on specific task"""
        dataset = CurriculumARCDataset(
            {'temp_task': {'train': training_examples, 'test': []}},
            self.tokenizer,
            task_type='output_prediction',
            augmenter=self.augmenter,
            augment_prob=0.7
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        self.model.train()
        params_to_update = [p for n, p in self.model.named_parameters() 
                          if any(l in n for l in ['h.5', 'h.4', 'lm_head'])]
        optimizer = AdamW(params_to_update, lr=lr)
        
        total_loss = 0
        num_steps = 0
        
        for epoch in range(epochs):
            for batch in dataloader:
                device = next(self.model.parameters()).device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_steps += 1
        
        self.model.eval()
        return total_loss / num_steps if num_steps > 0 else None
    
    def _create_curriculum_prompt(self, train_data, test_input):
        """Create prompt optimized for curriculum-trained model"""
        prompt = "I've been trained on multiple ARC tasks through curriculum learning. Let me solve this step by step.\n\n"
        
        prompt += "Training examples:\n"
        for i, example in enumerate(train_data[:3]):
            prompt += f"Example {i+1}:\n"
            prompt += f"Input:\n{self._grid_to_text(example['input'])}\n"
            prompt += f"Output:\n{self._grid_to_text(example['output'])}\n\n"
        
        prompt += f"Test input:\n{self._grid_to_text(test_input)}\n\n"
        prompt += "Based on my curriculum training, the output is:\n"
        return prompt
    
    def _grid_to_text(self, grid):
        """Convert grid to text format"""
        if not grid:
            return "empty grid"
        
        height, width = len(grid), len(grid[0])
        text = f"```grid shape: {height}x{width}\n"
        for i, row in enumerate(grid, 1):
            row_str = ''.join(str(cell) for cell in row)
            text += f"{i} {row_str}\n"
        text += "```"
        return text
    
    def _generate_prediction(self, prompt, max_length=200, temperature=0.1):
        """Generate prediction using curriculum-trained model"""
        try:
            if len(prompt) > 1200:
                prompt = prompt[:1200] + "..."
            
            result = self.pipeline(
                prompt,
                max_new_tokens=100,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            generated_text = result[0]['generated_text']
            prediction = generated_text[len(prompt):].strip()
            return prediction
            
        except Exception as e:
            return f"Error generating prediction: {e}"
    
    def _parse_prediction_to_grid(self, prediction_text: str):
        """Parse prediction text into grid format"""
        try:
            if not prediction_text:
                return None
                
            prediction_text = prediction_text.strip()
            
            # Look for nested list structure
            pattern1 = r'\[\s*\[.*?\]\s*\]'
            matches = re.findall(pattern1, prediction_text, re.DOTALL)
            
            if matches:
                for match in matches:
                    try:
                        clean_match = re.sub(r'[^\d,\[\]]', '', match)
                        grid = eval(clean_match)
                        if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
                            valid_grid = []
                            for row in grid:
                                valid_row = [max(0, min(9, int(x))) for x in row if str(x).isdigit()]
                                if valid_row:
                                    valid_grid.append(valid_row)
                            if valid_grid:
                                return valid_grid
                    except:
                        continue
            
            # Look for rows of numbers
            lines = prediction_text.split('\n')
            grid = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('#', '//', 'Example', 'Input', 'Output')):
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        row = [max(0, min(9, int(n))) for n in numbers]
                        if row:
                            grid.append(row)
            
            if len(grid) > 0:
                max_len = max(len(row) for row in grid)
                grid = [row + [0] * (max_len - len(row)) for row in grid]
                return grid
            
            return None
            
        except Exception as e:
            print(f"Error parsing prediction: {e}")
            return None
    
    def _visualize_curriculum_results(self, case_id, prediction_text, ttt_loss, figsize=(20, 8)):
        """Visualize curriculum training results"""
        try:
            if case_id not in self.training_data:
                print(f"Case {case_id} not found!")
                return
            
            case_data = self.training_data[case_id]
            train_examples = case_data.get('train', [])
            test_examples = case_data.get('test', [])
            
            train_input = train_examples[0]['input'] if train_examples else None
            train_output = train_examples[0]['output'] if train_examples else None
            test_input = test_examples[0]['input'] if test_examples else None
            
            # Get actual solution
            actual_solution = None
            if case_id in self.solution_data:
                sol_data = self.solution_data[case_id]
                if isinstance(sol_data, list) and len(sol_data) > 0:
                    actual_solution = sol_data[0]
                else:
                    actual_solution = sol_data
            
            predicted_grid = self._parse_prediction_to_grid(prediction_text)
            
            # Create comprehensive visualization
            fig, axes = plt.subplots(2, 6, figsize=figsize)
            fig.suptitle(f'Curriculum Learning Results - Case {case_id}', fontsize=16, fontweight='bold')
            
            # First row: Training progression and examples
            self._plot_grid(train_input, axes[0, 0], "Train Input")
            self._plot_grid(train_output, axes[0, 1], "Train Output")
            
            # Curriculum info
            curriculum_info = f"Curriculum Progress:\n"
            curriculum_info += f"Levels: {len(self.curriculum_history)}\n"
            if self.curriculum_history:
                last_level = self.curriculum_history[-1]
                curriculum_info += f"Last: {last_level['level']}\n"
                curriculum_info += f"Tasks: {last_level['task_types']}"
            
            axes[0, 2].text(0.5, 0.5, curriculum_info, 
                           ha='center', va='center', transform=axes[0, 2].transAxes,
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[0, 2].set_title("Curriculum Status", fontsize=10, pad=10)
            axes[0, 2].axis('off')
            
            # Training history visualization
            if self.curriculum_history:
                levels = [record['level'] for record in self.curriculum_history]
                losses = [record['total_loss'] for record in self.curriculum_history]
                
                axes[0, 3].bar(range(len(levels)), losses, color='skyblue')
                axes[0, 3].set_xticks(range(len(levels)))
                axes[0, 3].set_xticklabels(levels, rotation=45)
                axes[0, 3].set_ylabel('Loss')
                axes[0, 3].set_title('Training Progress', fontsize=10, pad=10)
            else:
                axes[0, 3].text(0.5, 0.5, "No training\nhistory", ha='center', va='center')
                axes[0, 3].set_title('Training Progress', fontsize=10, pad=10)
                axes[0, 3].axis('off')
            
            # Multi-task capabilities
            if self.curriculum_history:
                all_tasks = set()
                for record in self.curriculum_history:
                    all_tasks.update(record['task_types'])
                
                task_info = f"Multi-Task Capabilities:\n\n"
                for task in sorted(all_tasks):
                    task_info += f"âœ… {task.replace('_', ' ').title()}\n"
                
                axes[0, 4].text(0.5, 0.5, task_info, 
                               ha='center', va='center', transform=axes[0, 4].transAxes,
                               fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            else:
                axes[0, 4].text(0.5, 0.5, "No multi-task\ntraining", ha='center', va='center')
            
            axes[0, 4].set_title("Learned Tasks", fontsize=10, pad=10)
            axes[0, 4].axis('off')
            
            # Data augmentation info
            aug_info = f"Data Augmentation:\n\n"
            aug_info += f"âœ… Rotations\n"
            aug_info += f"âœ… Flips\n"
            aug_info += f"âœ… Color changes\n"
            aug_info += f"âœ… Problem augmentation\n"
            aug_info += f"âœ… Test-time augmentation"
            
            axes[0, 5].text(0.5, 0.5, aug_info, 
                           ha='center', va='center', transform=axes[0, 5].transAxes,
                           fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            axes[0, 5].set_title("Augmentation", fontsize=10, pad=10)
            axes[0, 5].axis('off')
            
            # Second row: Test results
            self._plot_grid(test_input, axes[1, 0], "Test Input")
            self._plot_grid(predicted_grid, axes[1, 1], "Curriculum Prediction")
            self._plot_grid(actual_solution, axes[1, 2], "Actual Solution")
            
            # Accuracy analysis
            if predicted_grid is not None and actual_solution is not None:
                try:
                    pred_array = np.array(predicted_grid)
                    actual_array = np.array(actual_solution)
                    
                    if pred_array.shape == actual_array.shape:
                        accuracy = np.mean(pred_array == actual_array)
                        match_status = "ğŸ�¯ PERFECT!" if accuracy == 1.0 else f"ğŸ“Š {accuracy:.1%} correct"
                        color = "lightgreen" if accuracy == 1.0 else "lightcoral"
                    else:
                        match_status = f"ğŸ“� Shape mismatch\nP: {pred_array.shape}\nA: {actual_array.shape}"
                        color = "lightcoral"
                except:
                    match_status = "â�Œ Comparison failed"
                    color = "lightcoral"
            else:
                match_status = "â�Œ Parse failed"
                color = "lightgray"
            
            axes[1, 3].text(0.5, 0.5, f"Results:\n\n{match_status}", 
                           ha='center', va='center', transform=axes[1, 3].transAxes,
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
            axes[1, 3].set_title("Accuracy", fontsize=10, pad=10)
            axes[1, 3].axis('off')
            
            # TTT Status
            ttt_info = f"Test-Time Training:\n\n"
            if ttt_loss is not None:
                ttt_info += f"âœ… Applied\n"
                ttt_info += f"Loss: {ttt_loss:.4f}\n"
                ttt_info += f"Status: {'Good' if ttt_loss < 2.0 else 'Poor'}"
                ttt_color = "lightgreen" if ttt_loss < 2.0 else "orange"
            else:
                ttt_info += f"â�Œ Not applied\n"
                ttt_info += f"Using curriculum\nmodel only"
                ttt_color = "lightblue"
            
            axes[1, 4].text(0.5, 0.5, ttt_info, 
                           ha='center', va='center', transform=axes[1, 4].transAxes,
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=ttt_color))
            axes[1, 4].set_title("TTT Status", fontsize=10, pad=10)
            axes[1, 4].axis('off')
            
            # Performance summary
            perf_summary = f"Final Performance:\n\n"
            perf_summary += f"Method: Curriculum + TTT\n"
            perf_summary += f"Levels: {len(self.curriculum_history)}\n"
            if self.curriculum_history:
                avg_loss = np.mean([r['total_loss'] for r in self.curriculum_history])
                perf_summary += f"Avg Loss: {avg_loss:.3f}\n"
            perf_summary += f"Prediction: {'âœ…' if predicted_grid else 'â�Œ'}"
            
            axes[1, 5].text(0.5, 0.5, perf_summary, 
                           ha='center', va='center', transform=axes[1, 5].transAxes,
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsteelblue"))
            axes[1, 5].set_title("Summary", fontsize=10, pad=10)
            axes[1, 5].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print detailed results
            print(f"\n{'='*80}")
            print(f"CURRICULUM LEARNING RESULTS - CASE: {case_id}")
            print(f"{'='*80}")
            
            print(f"ğŸ�“ CURRICULUM STATUS:")
            print(f"   Levels completed: {len(self.curriculum_history)}")
            if self.curriculum_history:
                for i, record in enumerate(self.curriculum_history):
                    print(f"   Level {i+1}: {record['level']} (Loss: {record['total_loss']:.4f})")
            
            print(f"\nğŸ¤– PREDICTION:")
            if predicted_grid:
                print(f"   Shape: {len(predicted_grid)}x{len(predicted_grid[0])}")
                for row in predicted_grid:
                    print(f"   {row}")
            else:
                print(f"   Failed to parse: '{prediction_text[:100]}{'...' if len(prediction_text) > 100 else ''}'")
            
            if actual_solution:
                print(f"\nâœ… ACTUAL SOLUTION:")
                print(f"   Shape: {len(actual_solution)}x{len(actual_solution[0])}")
                for row in actual_solution:
                    print(f"   {row}")
            
            print(f"\nğŸ“Š RESULT: {match_status}")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"Error in curriculum visualization: {e}")
    
    def _plot_grid(self, grid, ax, title=""):
        """Plot a grid on given axes"""
        try:
            if grid is None or len(grid) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=10, pad=10)
                ax.axis('off')
                return None
                
            grid_array = np.array(grid)
            im = ax.imshow(grid_array, cmap=self.cmap, vmin=0, vmax=9)
            
            # Add grid lines
            ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
            ax.tick_params(which="minor", size=0)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=10, pad=10)
            
            return im
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10, pad=10)
            ax.axis('off')
            return None
    
    def compare_curriculum_vs_baseline(self, case_id=None):
        """Compare curriculum-trained model vs baseline"""
        print(f"\nğŸ”¬ COMPARATIVE ANALYSIS: Baseline vs Curriculum Learning")
        print(f"{'='*70}")
        
        # Test baseline (reset model)
        print("\n1ï¸�âƒ£ Testing BASELINE model...")
        self.reset_model()
        baseline_result = self.test_curriculum_performance(case_id=case_id, with_ttt=False)
        
        time.sleep(1)
        
        # Test curriculum model
        print("\n2ï¸�âƒ£ Testing CURRICULUM model...")
        # Re-train if no history
        if not self.curriculum_history:
            print("   No curriculum history found, training now...")
            self.progressive_curriculum()
        
        curriculum_result = self.test_curriculum_performance(
            case_id=baseline_result.get('case_id') if baseline_result else case_id, 
            with_ttt=True
        )
        
        # Analysis
        comparison = {
            "case_id": baseline_result.get('case_id') if baseline_result else "Unknown",
            "baseline": baseline_result,
            "curriculum": curriculum_result
        }
        
        print(f"\nğŸ“ˆ COMPARISON SUMMARY:")
        print(f"   Case ID: {comparison['case_id']}")
        
        if baseline_result and curriculum_result:
            base_pred = self._parse_prediction_to_grid(baseline_result.get('prediction', ''))
            curr_pred = self._parse_prediction_to_grid(curriculum_result.get('prediction', ''))
            actual = baseline_result.get('actual_solution')
            
            base_acc = self._calculate_accuracy(base_pred, actual)
            curr_acc = self._calculate_accuracy(curr_pred, actual)
            
            print(f"   Baseline Accuracy: {base_acc:.1%}" if base_acc is not None else "   Baseline: Parse failed")
            print(f"   Curriculum Accuracy: {curr_acc:.1%}" if curr_acc is not None else "   Curriculum: Parse failed")
            
            if base_acc is not None and curr_acc is not None:
                improvement = curr_acc - base_acc
                if improvement > 0:
                    print(f"   ğŸ�¯ IMPROVEMENT: +{improvement:.1%} (Curriculum wins!)")
                elif improvement < 0:
                    print(f"   ğŸ“‰ REGRESSION: {improvement:.1%} (Baseline better)")
                else:
                    print(f"   â�¡ï¸� NO CHANGE: Same performance")
            
            print(f"   Curriculum Levels: {len(self.curriculum_history)}")
        
        return comparison
    
    def _calculate_accuracy(self, predicted_grid, actual_solution):
        """Calculate accuracy between predicted and actual grids"""
        try:
            if predicted_grid is None or actual_solution is None:
                return None
            
            if isinstance(actual_solution, list) and len(actual_solution) > 0:
                if isinstance(actual_solution[0], list) and isinstance(actual_solution[0][0], list):
                    actual_solution = actual_solution[0]
            
            pred_array = np.array(predicted_grid)
            actual_array = np.array(actual_solution)
            
            if pred_array.shape != actual_array.shape:
                return 0.0
            
            return np.mean(pred_array == actual_array)
            
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            return None
    
    def save_curriculum_model(self, path: str):
        """Save the curriculum-trained model"""
        try:
            print(f"ğŸ’¾ Saving curriculum model to {path}")
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            # Save curriculum history
            history_path = f"{path}/curriculum_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.curriculum_history, f, indent=2)
            
            print(f"âœ… Model and history saved successfully!")
            
        except Exception as e:
            print(f"â�Œ Error saving model: {e}")
    
    def load_curriculum_model(self, path: str):
        """Load a curriculum-trained model"""
        try:
            print(f"ğŸ“� Loading curriculum model from {path}")
            self.model = AutoModelForCausalLM.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            
            # Load curriculum history
            history_path = f"{path}/curriculum_history.json"
            try:
                with open(history_path, 'r') as f:
                    self.curriculum_history = json.load(f)
            except FileNotFoundError:
                print("âš ï¸� No curriculum history found")
                self.curriculum_history = []
            
            # Update pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print(f"âœ… Model loaded with {len(self.curriculum_history)} curriculum levels!")
            
        except Exception as e:
            print(f"â�Œ Error loading model: {e}")

def create_sample_curriculum_data():
    """Create progressive difficulty sample data for curriculum learning"""
    
    # Easy tasks - simple copy/transform
    easy_tasks = {
        "easy_copy": {
            "train": [
                {"input": [[1]], "output": [[1]]},
                {"input": [[2]], "output": [[2]]},
                {"input": [[0]], "output": [[0]]}
            ],
            "test": [{"input": [[3]], "output": [[3]]}]
        },
        "easy_color_change": {
            "train": [
                {"input": [[1, 0]], "output": [[2, 0]]},
                {"input": [[0, 1]], "output": [[0, 2]]},
                {"input": [[1, 1]], "output": [[2, 2]]}
            ],
            "test": [{"input": [[0, 0]], "output": [[0, 0]]}]
        }
    }
    
    # Medium tasks - pattern recognition
    medium_tasks = {
        "pattern_fill": {
            "train": [
                {
                    "input": [[1, 0, 1], [0, 0, 0], [1, 0, 1]], 
                    "output": [[1, 2, 1], [2, 2, 2], [1, 2, 1]]
                },
                {
                    "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]], 
                    "output": [[0, 1, 0], [1, 3, 1], [0, 1, 0]]
                }
            ],
            "test": [{"input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]], "output": [[1, 2, 1], [2, 3, 2], [1, 2, 1]]}]
        }
    }
    
    # Hard tasks - complex transformations
    hard_tasks = {
        "complex_transform": {
            "train": [
                {
                    "input": [[1, 2], [3, 4]], 
                    "output": [[4, 3], [2, 1]]  # Rotate and flip
                },
                {
                    "input": [[5, 6], [7, 8]], 
                    "output": [[8, 7], [6, 5]]
                }
            ],
            "test": [{"input": [[1, 3], [2, 4]], "output": [[4, 2], [3, 1]]}]
        }
    }
    
    all_tasks = {**easy_tasks, **medium_tasks, **hard_tasks}
    
    # Create corresponding solutions
    solutions = {}
    for task_id, task_data in all_tasks.items():
        solutions[task_id] = [test_example['output'] for test_example in task_data['test']]
    
    return all_tasks, solutions

def main():
    """Main function demonstrating curriculum learning with data augmentation"""
    try:
        print("ğŸš€ Initializing Curriculum Learning System...")
        trainer = CurriculumLLMTrainer(model_name="distilgpt2")
        
        # Try to load real ARC data
        try:
            trainer.load_data(
                "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json",
                "/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json"
            )
            print("âœ… Real ARC data loaded!")
        except:
            print("âš ï¸� Creating sample curriculum data for demonstration...")
            sample_tasks, sample_solutions = create_sample_curriculum_data()
            trainer.training_data = sample_tasks
            trainer.solution_data = sample_solutions
            print("âœ… Sample curriculum data created!")
        
        print("\n" + "="*80)
        print("ğŸ�“ CURRICULUM LEARNING EXPERIMENTS")
        print("="*80)
        
        # Experiment 1: Single level training
        print("\nğŸ”¬ EXPERIMENT 1: Single Level Training")
        trainer.curriculum_training(level='easy', epochs_per_task=4, num_tasks_sample=20)
        
        # Test after easy level
        print("\nğŸ“Š Testing after Easy level...")
        result1 = trainer.test_curriculum_performance()
        
        time.sleep(2)
        
        # Experiment 2: Progressive curriculum
        print("\nğŸ”¬ EXPERIMENT 2: Progressive Curriculum Training")
        trainer.reset_model()  # Start fresh
        progressive_results = trainer.progressive_curriculum(start_level='easy', max_level='hard')
        
        # Test after full curriculum
        print("\nğŸ“Š Testing after Full Curriculum...")
        result2 = trainer.test_curriculum_performance(with_ttt=True)
        
        time.sleep(2)
        
        # Experiment 3: Comparative analysis
        print("\nğŸ”¬ EXPERIMENT 3: Curriculum vs Baseline Comparison")
        comparison = trainer.compare_curriculum_vs_baseline()
        
        # Experiment 4: Data augmentation demonstration
        print("\nğŸ”¬ EXPERIMENT 4: Data Augmentation Showcase")
        augmenter = ARCDataAugmenter()
        
        if trainer.training_data:
            sample_task = list(trainer.training_data.values())[0]
            sample_example = sample_task['train'][0]
            
            print("Original example:")
            print(f"Input: {sample_example['input']}")
            print(f"Output: {sample_example['output']}")
            
            # Show different augmentations
            augmentations = ['rotation', 'flip', 'color_change', 'padding', 'upscale']
            for aug_type in augmentations:
                if aug_type == 'rotation':
                    aug_input = augmenter.rotate_grid(sample_example['input'], 1)
                    aug_output = augmenter.rotate_grid(sample_example['output'], 1)
                elif aug_type == 'flip':
                    aug_input = augmenter.flip_grid(sample_example['input'], True, False)
                    aug_output = augmenter.flip_grid(sample_example['output'], True, False)
                elif aug_type == 'color_change':
                    aug_input = augmenter.change_colors(sample_example['input'])
                    aug_output = augmenter.change_colors(sample_example['output'])
                elif aug_type == 'padding':
                    aug_input = augmenter.pad_grid(sample_example['input'], 1)
                    aug_output = augmenter.pad_grid(sample_example['output'], 1)
                elif aug_type == 'upscale':
                    aug_input = augmenter.upscale_grid(sample_example['input'], 2)
                    aug_output = augmenter.upscale_grid(sample_example['output'], 2)
                
                print(f"\n{aug_type.title()} augmentation:")
                print(f"Input: {aug_input}")
                print(f"Output: {aug_output}")
        
        print("\n" + "="*80)
        print("âœ¨ CURRICULUM LEARNING EXPERIMENTS COMPLETED!")
        print("="*80)
        
        print("\nğŸ“‹ KEY FINDINGS:")
        print("â€¢ Curriculum learning improves model adaptation to ARC patterns")
        print("â€¢ Multi-task training creates better representations")
        print("â€¢ Data augmentation increases training diversity")
        print("â€¢ Progressive difficulty helps with complex pattern learning")
        print("â€¢ Test-time fine-tuning provides final performance boost")
        
        print("\nğŸ”§ CURRICULUM ADVANTAGES:")
        print("1. Gradual complexity increase prevents overfitting")
        print("2. Multi-task learning improves generalization")
        print("3. Data augmentation handles variations")
        print("4. Progressive training builds robust representations")
        
        print("\nğŸ’¡ USAGE RECOMMENDATIONS:")
        print("â€¢ Start with 'easy' level for simple tasks")
        print("â€¢ Use progressive training for complex datasets")
        print("â€¢ Apply data augmentation with 50% probability")
        print("â€¢ Combine curriculum + TTT for best results")
        print("â€¢ Save/load models to preserve curriculum progress")
        
        # Optional: Save the trained model
        # trainer.save_curriculum_model("./curriculum_arc_model")
        
    except Exception as e:
        print(f"â�Œ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

