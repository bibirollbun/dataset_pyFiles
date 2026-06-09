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


import os

import pandas as pd
import polars as pl

import kaggle_evaluation.aimo_2_inference_server


import pandas as pd

# Load the reference dataset
ref_df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv')
print("Reference Data Info:")
print(ref_df.info())
print("\nReference Data Head:")
print(ref_df.head())

# Load the sample submission file to inspect its structure
sample_sub = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/sample_submission.csv')
print("\nSample Submission Data:")
print(sample_sub.head())


import pandas as pd

# Load the test dataset (public test set)
test_df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv')
print("Test Data Info:")
print(test_df.info())
print("\nTest Data Head:")
print(test_df.head())

# Define a baseline prediction function
def baseline_predict(problem):
    """
    A baseline prediction function that takes a problem statement (LaTeX string)
    and returns a constant value. This is just a starting point.
    
    For this example, we'll simply return 0 for every problem.
    In future steps, you might use natural language processing, feature extraction,
    or more sophisticated heuristics to generate predictions.
    """
    # Here you could process the 'problem' text if needed
    # For now, we'll return a baseline answer of 0 (and ensure it follows modulo 1000 rule)
    prediction = 0  
    return prediction % 1000  # Ensuring answer lies between 0 and 999

# Create the submission DataFrame using the test set ids and our baseline predictions
submission_df = test_df[['id']].copy()
submission_df['answer'] = test_df['problem'].apply(baseline_predict)

print("Baseline Submission Data:")
print(submission_df.head())


import re
import sympy as sp

# Define some common symbols for simple equations
x, y, z = sp.symbols('x y z')

def enhanced_predict(problem):
    """
    Enhanced prediction function that attempts to solve simple arithmetic or algebraic problems.
    
    It works for arithmetic expressions like $1-1$ or $0\times10$, and also interprets
    basic equations such as $4+x=4$.
    """
    # Extract the first math expression enclosed in dollar signs ($...$)
    expressions = re.findall(r'\$(.*?)\$', problem)
    
    if expressions:
        expr = expressions[0]
        # Replace common LaTeX operators with Python equivalents.
        expr = expr.replace(r'\times', '*')
        try:
            # Check if the expression represents an equation
            if '=' in expr:
                # Split at the first '=' into left and right sides
                left_side, right_side = expr.split('=', 1)
                # Use sympy with some pre-defined symbols
                left_expr = sp.sympify(left_side, locals={'x': x, 'y': y, 'z': z})
                right_expr = sp.sympify(right_side, locals={'x': x, 'y': y, 'z': z})
                
                # Determine the free symbols present in the equation
                free_symbols = left_expr.free_symbols.union(right_expr.free_symbols)
                if free_symbols:
                    # Solve the equation for the free symbols.
                    # For simplicity, we solve for the first symbol we encounter.
                    symbol_to_solve = list(free_symbols)[0]
                    sol = sp.solve(sp.Eq(left_expr, right_expr), symbol_to_solve)
                    
                    if sol:
                        # If the solution is a list, take the first solution.
                        sol_val = sol[0]
                        # Try to convert to an integer (if not an integer, it might be a fraction; we force int)
                        sol_int = int(sol_val)
                        return sol_int % 1000
                else:
                    # If no free symbols exist, evaluate the expression normally
                    val = sp.sympify(expr)
                    return int(val) % 1000
            else:
                # No equality in the expression: evaluate as a simple arithmetic expression.
                val = sp.sympify(expr, locals={'x': x, 'y': y, 'z': z})
                return int(val) % 1000
        except Exception as e:
            # If parsing/evaluation fails, gracefully default to 0.
            return 0
    # If there's no mathematical expression found, return 0.
    return 0

# Apply the enhanced_predict function to our test DataFrame
submission_df['answer'] = test_df['problem'].apply(enhanced_predict)

print("Enhanced Baseline Submission Data:")
print(submission_df)


import re
import sympy as sp

# Predefine some symbols for solving simple equations
x, y, z = sp.symbols('x y z')

def enhanced_predict(problem):
    """
    Attempts to evaluate a mathematical expression enclosed in '$...$' in the problem text.
    Handles both pure arithmetic expressions and simple equations.
    """
    # Extract content between dollar signs
    expressions = re.findall(r'\$(.*?)\$', problem)
    if expressions:
        expr = expressions[0]
        # Replace common LaTeX operators
        expr = expr.replace(r'\times', '*')
        try:
            # When there's an equality, solve for the free variable
            if '=' in expr:
                left_side, right_side = expr.split('=', 1)
                left_expr = sp.sympify(left_side, locals={'x': x, 'y': y, 'z': z})
                right_expr = sp.sympify(right_side, locals={'x': x, 'y': y, 'z': z})
                
                # Identify free symbols in the equation
                free_symbols = left_expr.free_symbols.union(right_expr.free_symbols)
                if free_symbols:
                    # Solve for one of the variables (the first one found)
                    symbol_to_solve = list(free_symbols)[0]
                    solution = sp.solve(sp.Eq(left_expr, right_expr), symbol_to_solve)
                    if solution:
                        # Convert the solution to an integer and apply modulo 1000
                        sol_val = solution[0]
                        sol_int = int(sol_val)
                        return sol_int % 1000
                else:
                    # If no variable is present, simply evaluate the expression
                    val = sp.sympify(expr)
                    return int(val) % 1000
            else:
                # Evaluate simple arithmetic expressions
                val = sp.sympify(expr, locals={'x': x, 'y': y, 'z': z})
                return int(val) % 1000
        except Exception:
            # If evaluation fails, fall back to 0
            return 0
    # No dollar-sign expression found: fallback to 0
    return 0

def robust_predict(problem):
    """
    A robust predictor that integrates multiple strategies:
      1. Use enhanced_predict() for math expressions in '$...$'
      2. Pattern matching for specific phrases (e.g., "sum of digits")
      3. A fallback heuristic if numbers are present in the text.
    """
    # First, try to solve using enhanced_predict
    result = enhanced_predict(problem)
    # If enhanced_predict returns a nonzero value or if the problem explicitly has math formatting, use it.
    if result != 0 or '$' in problem:
        return result
    
    # Lowercase the problem text for matching common patterns
    problem_lower = problem.lower()
    
    # Example: handle problems asking for "sum of digits"
    if "sum of digits" in problem_lower:
        match = re.search(r'(\d+)', problem_lower)
        if match:
            number_str = match.group(1)
            digits_sum = sum(int(digit) for digit in number_str)
            return digits_sum % 1000

    # Fallback strategy: if there are any numbers in the text, sum them (as a simple heuristic)
    numbers = re.findall(r'\d+', problem_lower)
    if numbers:
        total = sum(int(n) for n in numbers)
        return total % 1000

    # Default fallback
    return 0

# Apply the robust_predict function on our test data
submission_df['answer'] = test_df['problem'].apply(robust_predict)
print("Robust Prediction Submission Data:")
print(submission_df)


# Define additional sample problems to further test our predictor
additional_examples = pd.DataFrame({
    'id': ['ex1', 'ex2', 'ex3', 'ex4'],
    'problem': [
        "What is $3+7$?",                      # Expects 10
        "Find the sum of digits of 12345.",      # Should return 1+2+3+4+5 = 15
        "Solve $2x+3=7$ for $x$.",               # Expects x = 2
        "What is $5\\times6$?"                   # Expects 30
    ]
})

additional_examples['predicted'] = additional_examples['problem'].apply(robust_predict)
print(additional_examples)


import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# Include transformation for implicit multiplication: e.g., "2x" becomes "2*x"
transformations = standard_transformations + (implicit_multiplication_application,)

# Define symbols
x, y, z = sp.symbols('x y z')

def parse_math_expression(expr_string):
    """
    Parse a math expression string using Sympy's parse_expr with implicit multiplication.
    Returns the parsed expression or None if parsing fails.
    """
    try:
        return parse_expr(expr_string, transformations=transformations, local_dict={'x': x, 'y': y, 'z': z})
    except Exception:
        return None

def enhanced_predict(problem):
    """
    Attempts to extract and evaluate a math expression from within dollar signs ($...$).
    Handles an arithmetic expression or an equation (e.g., "2x+3=7").
    """
    # Extract all expressions enclosed in $...$
    expressions = re.findall(r'\$(.*?)\$', problem)
    if expressions:
        expr = expressions[0]
        # Replace common LaTeX operators
        expr = expr.replace(r'\times', '*')
        try:
            if '=' in expr:
                # For equation problems, split at the first '='
                left_side, right_side = expr.split('=', 1)
                left_expr = parse_math_expression(left_side)
                right_expr = parse_math_expression(right_side)
                if left_expr is None or right_expr is None:
                    return 0
                # Identify free symbols (variables) in the equation
                free_symbols = left_expr.free_symbols.union(right_expr.free_symbols)
                if free_symbols:
                    # Solve for one of the free symbols – typically the desired unknown.
                    symbol_to_solve = list(free_symbols)[0]
                    solution = sp.solve(sp.Eq(left_expr, right_expr), symbol_to_solve)
                    if solution:
                        sol_val = solution[0]
                        sol_int = int(sol_val)
                        return sol_int % 1000
                else:
                    # If there are no free symbols, just evaluate the whole expression.
                    val = parse_math_expression(expr)
                    if val is not None:
                        return int(val) % 1000
            else:
                # Evaluate a simple arithmetic expression without the equality sign.
                val = parse_math_expression(expr)
                if val is not None:
                    return int(val) % 1000
        except Exception:
            return 0
    # No detectable math expression in dollar signs; fallback to 0.
    return 0

def robust_predict(problem):
    """
    A more robust predictor that first attempts to evaluate math inside '$...$' and,
    if that doesn't yield a nonzero result (or no math expression is found), uses
    natural language cues as a fallback.
    """
    # Try the math evaluator first.
    result = enhanced_predict(problem)
    if result != 0 or '$' in problem:
        return result
    
    # Convert the problem to lower case for pattern matching.
    problem_lower = problem.lower()
    
    # Pattern handling: for example, "sum of digits" problems.
    if "sum of digits" in problem_lower:
        match = re.search(r'(\d+)', problem_lower)
        if match:
            number_str = match.group(1)
            digits_sum = sum(int(digit) for digit in number_str)
            return digits_sum % 1000

    # Fallback: if there are any numbers, sum them.
    numbers = re.findall(r'\d+', problem_lower)
    if numbers:
        total = sum(int(n) for n in numbers)
        return total % 1000

    return 0

# Test on the additional examples
import pandas as pd

additional_examples = pd.DataFrame({
    'id': ['ex1', 'ex2', 'ex3', 'ex4'],
    'problem': [
        "What is $3+7$?",                      # Expected 10
        "Find the sum of digits of 12345.",      # Expected 15
        "Solve $2x+3=7$ for $x$.",               # Expected: 2, since 2x+3=7 -> x=2
        "What is $5\\times6$?"                   # Expected 30
    ]
})

additional_examples['predicted'] = additional_examples['problem'].apply(robust_predict)
print(additional_examples)


import pandas as pd

# Load test data (the file with all competition problems)
test_df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv')

# Use the robust_predict function on the 'problem' column
test_df['answer'] = test_df['problem'].apply(robust_predict)

# Ensure the final dataframe has two columns: 'id' and 'answer'
submission_df = test_df[['id', 'answer']]

# Save the submission file; this is the file you will submit
submission_df.to_csv('submission.parquet', index=False)
print("Final submission file 'submission.parquet' created successfully.")



import os
import pandas as pd
import polars as pl

# Import the evaluation module (this module is provided by the competition)
import kaggle_evaluation.aimo_2_inference_server

# Inference code: the predict function
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """
    Make a prediction.

    This function is called for each inference request.
    It extracts the problem id and question string from the provided Polars DataFrames,
    computes a prediction (an integer between 0 and 999), and returns a DataFrame 
    with the id and the prediction.
    """
    # Unpack values from the DataFrames.
    # (Note: This sample uses .item(0), but you might adjust this if needed.)
    id_value = id_.item(0)
    question_value = question.item(0)
    
    # Here you insert your model's prediction logic. For this sample, we simply return 0.
    prediction = 0  # Replace this with your actual model prediction code.
    
    # Return the result as a DataFrame with columns 'id' and 'answer'
    return pl.DataFrame({'id': [id_value], 'answer': [prediction]})

# Initialize the inference server with the predict function.
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

# When in a competition (hidden test set) environment, serve inference requests.
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # Otherwise, run the local gateway for testing using the public test file.
    inference_server.run_local_gateway(('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',))


