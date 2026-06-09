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





!pip install latex2sympy
import os
import pandas as pd
import polars as pl
import latex2sympy
import sympy
import re
import numpy as np  # For numerical calculations and NaN handling

# Enhanced Helper Functions

def extract_numbers(problem_text):
    # More robust regex (handles decimals, scientific notation, etc.)
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", problem_text)
    try:
        return [float(n) for n in numbers]
    except ValueError: # Handles cases where conversion to float may fail
        return [] # Returns an empty list if no numbers are extracted


def parse_latex(latex_expression):
    try:
        sympy_expression = latex2sympy.latex2sympy(latex_expression)
        return sympy_expression
    except (latex2sympy.utils.LatexSyntaxError, TypeError, AttributeError) as e: # More specific error catching
        print(f"LaTeX Syntax Error: {e}")
        return None
    except Exception as e:
        print(f"Error parsing LaTeX: {e}")
        return None


def evaluate_sympy_expression(sympy_expression):
    """
    YOUR MATHEMATICAL PROBLEM-SOLVING LOGIC HERE.
    """
    try:
        if sympy_expression is None:
            return None

        # Example (replace with your logic based on problem type):
        if isinstance(sympy_expression, sympy.Number):  # If it's already a number
            return sympy_expression
        elif isinstance(sympy_expression, sympy.Add) or isinstance(sympy_expression, sympy.Mul) or isinstance(sympy_expression, sympy.Pow):  # Check for simple arithmetic operations
            try: # Try to evaluate basic operations
              return sympy.N(sympy_expression) # Convert to a number
            except TypeError: # If the evaluation fails
              return sympy_expression # Return the symbolic expression
        elif isinstance(sympy_expression, sympy.Symbol): # If it is a symbol
            return sympy_expression # Return the symbol
        elif isinstance(sympy_expression, sympy.Eq): # If it is an equation
            return sympy_expression # Return the equation
        else:
            return None # Handle more complex cases as needed

    except (TypeError, NameError, AttributeError, SyntaxError) as e:  # More specific error catching
        print(f"Error evaluating SymPy expression: {e}")
        return None
    except Exception as e:
        print(f"General Error evaluating SymPy expression: {e}")
        return None


def solve_problem(problem_latex):
    try:
      sympy_expression = parse_latex(problem_latex)
      if sympy_expression is not None:
        solution = evaluate_sympy_expression(sympy_expression)
        return solution
      else:
        numbers = extract_numbers(problem_latex)
        if numbers:
          # If parsing fails, try simple arithmetic on extracted numbers (if applicable)
          # Example: if problem is "What is 1 + 2?"
          try:
            return eval(problem_latex.replace('What is ','').replace('?','').replace('\\times','*').replace(' ','')) # Placeholder, do not use eval!
          except:
            return None
        else:
          return None
    except Exception as e:
        print(f"Error solving problem: {e}")
        return None


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    question_latex = question.item(0)

    solution = solve_problem(question_latex)

    if solution is not None:
        try:
            answer = int(solution) % 1000  # Apply modulo 1000
        except (TypeError, ValueError): # Handle if solution is a sympy expression
            answer = 0
    else:
        answer = 0  # Default answer

    return pl.DataFrame({'id': id_, 'answer': answer})






import os
import pandas as pd
import polars as pl
import latex2sympy
import sympy
import re
import numpy as np

# Enhanced Helper Functions

def extract_numbers(problem_text):
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", problem_text)
    try:
        return [float(n) for n in numbers]
    except ValueError:
        return []

def parse_latex(latex_expression):
    try:
        sympy_expression = latex2sympy.latex2sympy(latex_expression)
        return sympy_expression
    except (latex2sympy.utils.LatexSyntaxError, TypeError, AttributeError) as e:
        print(f"LaTeX Syntax Error: {e}")
        return None
    except Exception as e:
        print(f"Error parsing LaTeX: {e}")
        return None

def evaluate_sympy_expression(sympy_expression):
    """
    YOUR MATHEMATICAL PROBLEM-SOLVING LOGIC HERE.  THIS IS THE MOST 
    IMPORTANT PART THAT YOU MUST IMPLEMENT.

    Example (replace with your logic based on problem type):
    """
    try:
        if sympy_expression is None:
            return None

        # Example: Solving simple equations (replace with your logic)
        if isinstance(sympy_expression, sympy.Eq):
            try:
                x = sympy.Symbol('x')  # Define x as a symbol (or other relevant symbol)
                solution = sympy.solve(sympy_expression, x)
                if solution:
                    return solution[0]  # Return the first solution (handle multiple solutions if needed)
                else:
                    return None # No solution found
            except Exception as e:
                print(f"Error solving equation: {e}")
                return None
        elif isinstance(sympy_expression, sympy.Number):
            return sympy_expression
        elif isinstance(sympy_expression, sympy.Add) or isinstance(sympy_expression, sympy.Mul) or isinstance(sympy_expression, sympy.Pow):
            try:
              return sympy.N(sympy_expression)
            except TypeError:
              return sympy_expression
        elif isinstance(sympy_expression, sympy.Symbol):
            return sympy_expression
        elif isinstance(sympy_expression, sympy.Rel): # Handle inequalities
            return sympy_expression # Placeholder - add inequality solving logic
        # ... (Add more cases for different problem types)

        return None  # Default: could not solve

    except (TypeError, NameError, AttributeError, SyntaxError) as e:
        print(f"Error evaluating SymPy expression: {e}")
        return None
    except Exception as e:
        print(f"General Error evaluating SymPy expression: {e}")
        return None

def solve_problem(problem_latex):
    try:
      sympy_expression = parse_latex(problem_latex)
      if sympy_expression is not None:
        solution = evaluate_sympy_expression(sympy_expression)
        return solution
      else:
        numbers = extract_numbers(problem_latex)
        if numbers:
          # If parsing fails, try simple arithmetic on extracted numbers (if applicable)
          # Example: if problem is "What is 1 + 2?"
          # Note: This is a TEMPORARY fallback. Use SymPy for real math!
          try:
            return eval(problem_latex.replace('What is ','').replace('?','').replace('\\times','*').replace(' ','')) # Placeholder, do not use eval!
          except:
            return None
        else:
          return None
    except Exception as e:
        print(f"Error solving problem: {e}")
        return None


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    question_latex = question.item(0)

    solution = solve_problem(question_latex)

    if solution is not None:
        try:
            answer = int(solution) % 1000
        except (TypeError, ValueError):
            answer = 0  # Or a more sophisticated default value
    else:
        answer = 0  # Default answer if problem can't be solved

    return pl.DataFrame({'id': id_, 'answer': answer})

# ... (Inference server initialization remains the same)


import os
import pandas as pd
import polars as pl
import latex2sympy
import sympy
import re
import pandas as pd
import numpy as np  # For numerical calculations and NaN handling

# Enhanced Helper Functions

def extract_numbers(problem_text):
    # More robust regex (handles decimals, scientific notation, etc.)
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", problem_text)
    try:
        return [float(n) for n in numbers]
    except ValueError: # Handles cases where conversion to float may fail
        return [] # Returns an empty list if no numbers are extracted


def parse_latex(latex_expression):
    try:
        sympy_expression = latex2sympy.latex2sympy(latex_expression)
        return sympy_expression
    except (latex2sympy.utils.LatexSyntaxError, TypeError, AttributeError) as e: # More specific error catching
        print(f"LaTeX Syntax Error: {e}")
        return None
    except Exception as e:
        print(f"Error parsing LaTeX: {e}")
        return None


def evaluate_sympy_expression(sympy_expression):
    """
    YOUR MATHEMATICAL PROBLEM-SOLVING LOGIC HERE.
    """
    try:
        if sympy_expression is None:
            return None

        # Example (replace with your logic based on problem type):
        if isinstance(sympy_expression, sympy.Number):  # If it's already a number
            return sympy_expression
        elif isinstance(sympy_expression, sympy.Add) or isinstance(sympy_expression, sympy.Mul) or isinstance(sympy_expression, sympy.Pow):  # Check for simple arithmetic operations
            try: # Try to evaluate basic operations
              return sympy.N(sympy_expression) # Convert to a number
            except TypeError: # If the evaluation fails
              return sympy_expression # Return the symbolic expression
        elif isinstance(sympy_expression, sympy.Symbol): # If it is a symbol
            return sympy_expression # Return the symbol
        elif isinstance(sympy_expression, sympy.Eq): # If it is an equation
            return sympy_expression # Return the equation
        else:
            return None # Handle more complex cases as needed

    except (TypeError, NameError, AttributeError, SyntaxError) as e:  # More specific error catching
        print(f"Error evaluating SymPy expression: {e}")
        return None
    except Exception as e:
        print(f"General Error evaluating SymPy expression: {e}")
        return None


def solve_problem(problem_latex):
    try:
      sympy_expression = parse_latex(problem_latex)
      if sympy_expression is not None:
        solution = evaluate_sympy_expression(sympy_expression)
        return solution
      else:
        numbers = extract_numbers(problem_latex)
        if numbers:
          # If parsing fails, try simple arithmetic on extracted numbers (if applicable)
          # Example: if problem is "What is 1 + 2?"
          try:
            return eval(problem_latex.replace('What is ','').replace('?','').replace('\\times','*').replace(' ','')) # Placeholder, do not use eval!
          except:
            return None
        else:
          return None
    except Exception as e:
        print(f"Error solving problem: {e}")
        return None


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    question_latex = question.item(0)

    solution = solve_problem(question_latex)

    if solution is not None:
        try:
            answer = int(solution) % 1000  # Apply modulo 1000
        except (TypeError, ValueError): # Handle if solution is a sympy expression
            answer = 0
    else:
        answer = 0  # Default answer

    return pl.DataFrame({'id': id_, 'answer': answer})




import os
import pandas as pd
import polars as pl
import latex2sympy  # If you're using it
import sympy
import re
import numpy as np
import kaggle  # For Kaggle API interaction

# --- Helper Functions (Your Problem-Solving Logic) ---

def extract_numbers(problem_text):
    #... (Your implementation)

def parse_latex(latex_expression):
    #... (Your implementation)

def evaluate_sympy_expression(sympy_expression):
    """
    YOUR MATHEMATICAL PROBLEM-SOLVING LOGIC HERE. THIS IS THE MOST CRUCIAL PART.
    """
    try:
        if sympy_expression is None:
            return None

        # Example: Solving simple equations (replace with your actual logic)
        if isinstance(sympy_expression, sympy.Eq):
            try:
                x = sympy.Symbol('x')
                solution = sympy.solve(sympy_expression, x)
                if solution:
                    return solution
                else:
                    return None
            except Exception as e:
                print(f"Error solving equation: {e}")
                return None
        #... (Add more cases for different problem types)

        return None  # Default: could not solve

    except Exception as e:
        print(f"Error evaluating SymPy expression: {e}")
        return None

def solve_problem(problem_latex):
    try:
        sympy_expression = parse_latex(problem_latex)
        if sympy_expression is not None:
            solution = evaluate_sympy_expression(sympy_expression)
            return solution
        else:
            numbers = extract_numbers(problem_latex)
            if numbers:
                try:
                    return eval(problem_latex.replace('What is ', '').replace('?', '').replace('\\times', '*').replace(' ', ''))  # Placeholder, do not use eval in your final solution!
                except:
                    return None
            else:
                return None
    except Exception as e:
        print(f"Error solving problem: {e}")
        return None

# --- Kaggle Submission Code ---

# Set environment variable (important - do this at the VERY beginning of your notebook)
os.environ['KAGGLE_CONFIG_DIR'] = '/kaggle/input'  # Or the correct path

# Install latex2sympy (if needed - do this in a separate cell at the top of the notebook)
#!pip install latex2sympy

def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    question_latex = question.item(0)

    solution = solve_problem(question_latex)

    if solution is not None:
        try:
            answer = int(solution) % 1000
        except (TypeError, ValueError):
            answer = 0
    else:
        answer = 0


import os
import pandas as pd
import polars as pl
import latex2sympy  # If you're using it
import sympy
import re
import numpy as np
import kaggle  # For Kaggle API interaction

# --- Helper Functions (Your Problem-Solving Logic) ---

def extract_numbers(problem_text):  # Correct indentation
    # ... (Your implementation - must be indented)
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", problem_text)
    try:
        return [float(n) for n in numbers]
    except ValueError:
        return []


def parse_latex(latex_expression):  # Correct indentation
    # ... (Your implementation - must be indented)
    try:
        sympy_expression = latex2sympy.latex2sympy(latex_expression)
        return sympy_expression
    except (latex2sympy.utils.LatexSyntaxError, TypeError, AttributeError) as e:
        print(f"LaTeX Syntax Error: {e}")
        return None
    except Exception as e:
        print(f"Error parsing LaTeX: {e}")
        return None

def evaluate_sympy_expression(sympy_expression):  # Correct indentation
    # ... (Your implementation - must be indented)
    try:
        if sympy_expression is None:
            return None

        # Example: Solving simple equations (replace with your actual logic)
        if isinstance(sympy_expression, sympy.Eq):
            try:
                x = sympy.Symbol('x')
                solution = sympy.solve(sympy_expression, x)
                if solution:
                    return solution[0]
                else:
                    return None
            except Exception as e:
                print(f"Error solving equation: {e}")
                return None
        # ... (Add more cases for different problem types)

        return None  # Default: could not solve

    except Exception as e:
        print(f"Error evaluating SymPy expression: {e}")
        return None

def solve_problem(problem_latex):  # Correct indentation
    try:
        sympy_expression = parse_latex(problem_latex)
        if sympy_expression is not None:
            solution = evaluate_sympy_expression(sympy_expression)
            return solution
        else:
            numbers = extract_numbers(problem_latex)
            if numbers:
                try:
                    return eval(problem_latex.replace('What is ', '').replace('?', '').replace('\\times', '*').replace(' ', ''))  # Placeholder, do not use eval in your final solution!
                except:
                    return None
            else:
                return None
    except Exception as e:
        print(f"Error solving problem: {e}")
        return None

# ... (Rest of your code: predict function, Kaggle API interaction)


import os
import pandas as pd
import polars as pl
import latex2sympy  # If you're using it
import sympy
import re
import numpy as np
import kaggle  # For Kaggle API interaction

# --- Helper Functions (Your Problem-Solving Logic) ---

def extract_numbers(problem_text):
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", problem_text)
    try:
        return [float(n) for n in numbers]
    except ValueError:
        return

def parse_latex(latex_expression):
    try:
        sympy_expression = latex2sympy.latex2sympy(latex_expression)
        return sympy_expression
    except (latex2sympy.utils.LatexSyntaxError, TypeError, AttributeError) as e:
        print(f"LaTeX Syntax Error: {e}")
        return None
    except Exception as e:
        print(f"Error parsing LaTeX: {e}")
        return None

def evaluate_sympy_expression(sympy_expression):
    """
    YOUR MATHEMATICAL PROBLEM-SOLVING LOGIC HERE. THIS IS THE MOST CRUCIAL PART.
    """
    try:
        if sympy_expression is None:
            return None

        # Example: Solving simple equations (replace with your actual logic)
        if isinstance(sympy_expression, sympy.Eq):
            try:
                x = sympy.Symbol('x')
                solution = sympy.solve(sympy_expression, x)
                if solution:
                    return solution
                else:
                    return None
            except Exception as e:
                print(f"Error solving equation: {e}")
                return None
        #... (Add more cases for different problem types)

        return None  # Default: could not solve

    except Exception as e:
        print(f"Error evaluating SymPy expression: {e}")
        return None

def solve_problem(problem_latex):
    try:
        sympy_expression = parse_latex(problem_latex)
        if sympy_expression is not None:
            solution = evaluate_sympy_expression(sympy_expression)
            return solution
        else:
            numbers = extract_numbers(problem_latex)
            if numbers:
                try:
                    return eval(problem_latex.replace('What is ', '').replace('?', '').replace('\\times', '*').replace(' ', ''))  # Placeholder, do not use eval in your final solution!
                except:
                    return None
            else:
                return None
    except Exception as e:
        print(f"Error solving problem: {e}")
        return None

# --- Kaggle Submission Code ---

# Set environment variable (important - do this at the VERY beginning of your notebook)
os.environ['KAGGLE_CONFIG_DIR'] = '/kaggle/input'  # Or the correct path

# Install latex2sympy (if needed - do this in a separate cell at the top of the notebook)
#!pip install latex2sympy

def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    question_latex = question.item(0)

    solution = solve_problem(question_latex)

    if solution is not None:
        try:
            answer = int(solution) % 1000
        except (TypeError, ValueError):
            answer = 0
    else:
        answer = 0

    return pl.DataFrame({'id': id_, 'answer': answer})

# Initialize Kaggle API (after setting the environment variable)
api = kaggle.KaggleApi()
api.authenticate()

competition_slug = "ai-mathematical-olympiad-progress-prize-2"  # Replace if needed

# --- Generate Submission File and Submit ---

predictions =  # Make sure this list is populated correctly!

# Example (replace with your actual data loading and prediction loop)
# for id_, question in zip(test_ids, test_questions): # Replace test_ids and test_questions
#     prediction = predict(pl.DataFrame({'id': id_}), pl.DataFrame({'question': question}))
#     predictions.append(prediction.to_dict()) # Convert to dict and append

#... your code to generate predictions and populate the 'predictions' list...

df = pd.DataFrame(predictions)  # Create the pandas DataFrame

submission_file_path = os.path.join(os.getcwd(), "submission.csv")
df.to_csv(submission_file_path, index=False)

# Submit the competition
api.competition_submit(submission_file_path, "Your submission description", competition_slug)

print("Submission Complete!")


import os
import pandas as pd
import polars as pl
import latex2sympy  # If you're using it
import sympy
import re
import numpy as np
import kaggle  # For Kaggle API interaction

#... (Your helper functions: extract_numbers, parse_latex, evaluate_sympy_expression, solve_problem)

# --- Kaggle Submission Code ---

os.environ['KAGGLE_CONFIG_DIR'] = '/kaggle/input'  # Or the correct path

#... (Your predict function)

# Initialize Kaggle API
api = kaggle.KaggleApi()
api.authenticate()

competition_slug = "ai-mathematical-olympiad-progress-prize-2"  # Replace if needed

# --- Generate Submission File and Submit ---

predictions =  #Initialize the list - THIS IS THE FIX

# Example (replace with your actual data loading and prediction loop)
# Assuming you have a way to get the test data (e.g., from a CSV or the Kaggle API)
# and it's in a variable called 'test_data' (list of dictionaries)

# Example using test_data (replace with your actual data source):

# for item in test_data:
#     problem_latex = item['problem']
#     solution = solve_problem(problem_latex)

#     if solution is not None:
#         try:
#             answer = int(solution) % 1000
#             predictions.append({'id': item['id'], 'answer': answer})
#         except (TypeError, ValueError):
#             answer = 0
#             predictions.append({'id': item['id'], 'answer': answer})
#     else:
#         answer = 0
#         predictions.append({'id': item['id'], 'answer': answer})

#... your code to generate predictions and populate the 'predictions' list...

df = pd.DataFrame(predictions)  # Now 'predictions' should be a valid list

submission_file_path = os.path.join(os.getcwd(), "submission.csv")
df.to_csv(submission_file_path, index=False)

# Submit the competition
api.competition_submit(submission_file_path, "Your submission description", competition_slug)

print("Submission Complete!")


import os
import pandas as pd
import polars as pl
import latex2sympy
import sympy
import re
import numpy as np  # For numerical calculations and NaN handling

# Enhanced Helper Functions

def extract_numbers(problem_text):
    # More robust regex (handles decimals, scientific notation, etc.)
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", problem_text)
    try:
        return [float(n) for n in numbers]
    except ValueError: # Handles cases where conversion to float may fail
        return [] # Returns an empty list if no numbers are extracted


def parse_latex(latex_expression):
    try:
        sympy_expression = latex2sympy.latex2sympy(latex_expression)
        return sympy_expression
    except (latex2sympy.utils.LatexSyntaxError, TypeError, AttributeError) as e: # More specific error catching
        print(f"LaTeX Syntax Error: {e}")
        return None
    except Exception as e:
        print(f"Error parsing LaTeX: {e}")
        return None


def evaluate_sympy_expression(sympy_expression):
    """
    YOUR MATHEMATICAL PROBLEM-SOLVING LOGIC HERE.
    """
    try:
        if sympy_expression is None:
            return None

        # Example (replace with your logic based on problem type):
        if isinstance(sympy_expression, sympy.Number):  # If it's already a number
            return sympy_expression
        elif isinstance(sympy_expression, sympy.Add) or isinstance(sympy_expression, sympy.Mul) or isinstance(sympy_expression, sympy.Pow):  # Check for simple arithmetic operations
            try: # Try to evaluate basic operations
              return sympy.N(sympy_expression) # Convert to a number
            except TypeError: # If the evaluation fails
              return sympy_expression # Return the symbolic expression
        elif isinstance(sympy_expression, sympy.Symbol): # If it is a symbol
            return sympy_expression # Return the symbol
        elif isinstance(sympy_expression, sympy.Eq): # If it is an equation
            return sympy_expression # Return the equation
        else:
            return None # Handle more complex cases as needed

    except (TypeError, NameError, AttributeError, SyntaxError) as e:  # More specific error catching
        print(f"Error evaluating SymPy expression: {e}")
        return None
    except Exception as e:
        print(f"General Error evaluating SymPy expression: {e}")
        return None


def solve_problem(problem_latex):
    try:
      sympy_expression = parse_latex(problem_latex)
      if sympy_expression is not None:
        solution = evaluate_sympy_expression(sympy_expression)
        return solution
      else:
        numbers = extract_numbers(problem_latex)
        if numbers:
          # If parsing fails, try simple arithmetic on extracted numbers (if applicable)
          # Example: if problem is "What is 1 + 2?"
          try:
            return eval(problem_latex.replace('What is ','').replace('?','').replace('\\times','*').replace(' ','')) # Placeholder, do not use eval!
          except:
            return None
        else:
          return None
    except Exception as e:
        print(f"Error solving problem: {e}")
        return None


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    question_latex = question.item(0)

    solution = solve_problem(question_latex)

    if solution is not None:
        try:
            answer = int(solution) % 1000  
        except (TypeError, ValueError): 
            answer = 0
    else:
        answer = 0  # Default answer

    return pl.DataFrame({'id': id_, 'answer': answer})


import pandas as pd
# ... other imports

# Example: If test data is in a CSV file (replace with actual file path)
test_df = pd.read_csv("/kaggle/input/test.csv")  # Replace with actual path

predictions = []
for index, row in test_df.iterrows(): # Or use .itertuples()
    problem_latex = row['question'] # Adapt this to your data
    id_ = row['id']
    solution = solve_problem(problem_latex)

    if solution is not None:
        try:
            answer = int(solution) % 1000
            predictions.append({'id': id_, 'answer': answer})
        except (TypeError, ValueError):
            answer = 0
            predictions.append({'id': id_, 'answer': answer})
    else:
        answer = 0
        predictions.append({'id': id_, 'answer': answer})


print(predictions) # Check the contents of the predictions list!

df = pd.DataFrame(predictions)
# ... rest of the code (saving to CSV and submitting)

