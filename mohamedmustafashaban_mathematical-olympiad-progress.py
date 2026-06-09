# Import necessary libraries
# - pandas: for data manipulation and analysis.
# - numpy: for numerical operations.
# - torch: for working with PyTorch models.
# - os: for operating system functionalities.
# - logging: for logging information and errors.
# - heapq: for heap queue algorithms.
# - warnings: to suppress warnings during execution


# Import necessary libraries
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging
import matplotlib.pyplot as plt
from warnings import filterwarnings
import sympy as sp  # For LaTeX and mathematical processing
filterwarnings('ignore')



# Load a specialized math model (e.g., GPT-3 or a fine-tuned model)
def load_math_model():
    """Load a specialized math model and tokenizer."""
    try:
        print("Loading specialized math model...")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")  # Example of a more powerful model
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None


# Process a math problem using the specialized model
def process_problem_with_math_model(problem, tokenizer, model):
    """Process a math problem using the specialized math model."""
    try:
        inputs = tokenizer.encode(problem, return_tensors="pt")
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
        solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return solution
    except Exception as e:
        logging.error(f"Error processing problem with math model: {e}")
        return None


# Convert LaTeX expressions to computable format using SymPy
def latex_to_solution(latex_expression):
    """Convert LaTeX expressions to computable format using SymPy."""
    try:
        # Remove LaTeX formatting (e.g., $, \[, \])
        cleaned_expression = latex_expression.replace("$", "").replace("\\[", "").replace("\\]", "")
        # Use SymPy to parse and evaluate the expression
        expr = sp.sympify(cleaned_expression)
        return expr
    except Exception as e:
        logging.error(f"Error converting LaTeX to solution: {e}")
        return None


# Solve multiple problems using the specialized model
def solve_problems_with_math_model(data, tokenizer, model):
    """Solve multiple problems using the specialized math model."""
    results = []
    for idx, row in data.iterrows():
        problem = row.get('problem', None)
        if problem:
            # Step 1: Process the problem using the math model
            solution = process_problem_with_math_model(problem, tokenizer, model)
            # Step 2: Convert LaTeX to computable format
            computable_solution = latex_to_solution(solution)
            # Step 3: Evaluate the solution and apply modulo 1000
            try:
                final_solution = int(computable_solution.evalf()) % 1000
            except:
                final_solution = -1
        else:
            final_solution = -1
        results.append(final_solution)
    
    logging.info(f"Processed {len(results)} problems with the math model.")
    return results


# Save results to a CSV file
def save_results(results, output_file):
    """Save results to a CSV file."""
    try:
        submission = pd.DataFrame({'id': range(len(results)), 'solution': results})
        submission.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file} successfully!")
    except Exception as e:
        logging.error(f"Error saving results: {e}")


# Plot the results
def plot_results(results):
    """Plot the results for visualization."""
    plt.figure(figsize=(10, 5))
    plt.hist(results, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Solutions")
    plt.xlabel("Solutions")
    plt.ylabel("Frequency")
    plt.axvline(np.mean(results), color='red', linestyle='dashed', linewidth=1)
    plt.text(np.mean(results) + 1, 1, f'Mean: {np.mean(results):.2f}', color='red')
    plt.show()


# Main function to execute the workflow
def main(reference_file, test_file, output_file):
    """Main function to execute the workflow."""
    if not os.path.exists(reference_file) or not os.path.exists(test_file):
        logging.error("Error: Reference or test file not found!")
        return

    # Load data files
    reference_data = pd.read_csv(reference_file)
    test_data = pd.read_csv(test_file)

    # Load the specialized math model
    tokenizer, model = load_math_model()

    # Solve problems using the specialized math model
    results = solve_problems_with_math_model(test_data, tokenizer, model)

    # Save results to a file
    save_results(results, output_file)

    # Plot results
    plot_results(results)


# Execution block to run the main function
if __name__ == "__main__":
    reference_file = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"
    test_file = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv"
    output_file = "/kaggle/working/submission.parquet"  

    main(reference_file, test_file, output_file)

