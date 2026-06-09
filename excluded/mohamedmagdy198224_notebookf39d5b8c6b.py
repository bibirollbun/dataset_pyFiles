# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import json
import sympy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict, Any

class MathOlympiadSolver:
    def __init__(self):
        """Initialize the math problem solver with models and tools"""
        self.models = {
            'flan_t5': {
                'model': AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large"),
                'tokenizer': AutoTokenizer.from_pretrained("google/flan-t5-large")
            }
        }
        
        self.math_tools = {
            'simplify': sympy.simplify,
            'solve': sympy.solve,
            'expand': sympy.expand,
            'factor': sympy.factor
        }

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load problem data from JSONL file"""
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]

    def preprocess_problem(self, problem_text: str) -> str:
        """Format the problem for the model"""
        return f"Solve this math olympiad problem step by step: {problem_text}"

    def generate_solution(self, problem_text: str) -> str:
        """Generate a solution using the AI model"""
        inputs = self.models['flan_t5']['tokenizer'](
            self.preprocess_problem(problem_text),
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        outputs = self.models['flan_t5']['model'].generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        
        return self.models['flan_t5']['tokenizer'].decode(outputs[0], skip_special_tokens=True)

    def verify_solution(self, problem: str, solution: str) -> bool:
        """Verify the mathematical correctness of the solution"""
        try:
            # Basic verification - can be enhanced
            return bool(self.extract_math_expression(problem)) and bool(self.extract_math_expression(solution))
        except:
            return False

    def extract_math_expression(self, text: str) -> Any:
        """Extract mathematical expressions from text"""
        try:
            if '=' in text:
                parts = text.split('=')
                if len(parts) >= 2:
                    return sympy.sympify(parts[1].strip())
            return None
        except:
            return None

    def create_submission(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate submission solutions for all test problems"""
        solutions = []
        for problem in test_data:
            solution = self.generate_solution(problem['problem'])
            solutions.append({
                'id': problem['id'],
                'solution': solution,
                'verified': self.verify_solution(problem['problem'], solution)
            })
        return solutions

    def save_submission(self, solutions: List[Dict[str, Any]], filename: str = "submission.jsonl"):
        """Save solutions in competition format"""
        with open(filename, 'w') as f:
            for sol in solutions:
                f.write(json.dumps({'id': sol['id'], 'solution': sol['solution']}) + '\n')

def main():
    solver = MathOlympiadSolver()
    
    try:
        # Load test data (update path for Kaggle environment)
        test_data = solver.load_data('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.jsonl')
        
        # Generate solutions
        solutions = solver.create_submission(test_data)
        
        # Save submission
        solver.save_submission(solutions)
        print("Submission file created successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()

