pip install sympy


import polars as pl
import pandas as pd
from sympy import symbols, solve, sympify
import re

# Define the predict function
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction for each math problem."""
    id_value = id_['id'][0] if isinstance(id_, pl.DataFrame) else id_.iloc[0, 0]
    question_text = question['problem'][0] if isinstance(question, pl.DataFrame) else question.iloc[0, 0]

    # Step 1: Identify and solve simple algebraic problems
    prediction = 0
    try:
        # Look for simple equations like "Solve for x: x^2 - 3x + 2 = 0"
        match = re.search(r"solve\s+for\s+(\w)[^:]*:\s*(.*)=0", question_text, re.IGNORECASE)
        if match:
            var = match.group(1)
            equation = match.group(2).strip()
            x = symbols(var)

            # Convert the equation string to a symbolic expression
            equation_sympy = sympify(equation)
            solutions = solve(equation_sympy, x)

            # Take the first solution if available and apply modulo 1000
            if solutions:
                prediction = int(solutions[0]) % 1000
        else:
            # Placeholder for more advanced problem-solving logic
            prediction = 0  # Default if no solution found
    except Exception as e:
        print(f"Error processing problem: {e}")
        prediction = 0

    # Step 2: Return the prediction
    return pl.DataFrame({'id': id_value, 'answer': prediction}) if isinstance(id_, pl.DataFrame) else pd.DataFrame({'id': [id_value], 'answer': [prediction]})



test_data = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv')
sample_question = test_data.loc[0, 'problem']

# Prepare input for prediction
id_input = pl.DataFrame({'id': [1]})
question_input = pl.DataFrame({'problem': [sample_question]})

# Get prediction
predicted_answer = predict(id_input, question_input)
print(predicted_answer)


predictions = []
for _, row in test_data.iterrows():
    id_input = pl.DataFrame({'id': [row['id']]})
    question_input = pl.DataFrame({'problem': [row['problem']]})
    prediction = predict(id_input, question_input)
    predictions.append({'id': row['id'], 'answer': prediction['answer'][0]})

# Save predictions to a CSV file for submission
submission_df = pd.DataFrame(predictions)
submission_df.to_csv('submission.csv', index=False)


import pandas as pd
import polars as pl

# Initialize predictions list
predictions = []

# Iterate through the test data
test_data = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv')
for _, row in test_data.iterrows():
    id_input = pl.DataFrame({'id': [row['id']]})
    question_input = pl.DataFrame({'problem': [row['problem']]})
    prediction = predict(id_input, question_input)
    predictions.append({'id': row['id'], 'answer': prediction['answer'][0]})

# Create a DataFrame from predictions
submission_df = pd.DataFrame(predictions)

# Save the DataFrame to a `.parquet` file
submission_df.to_parquet('submission.parquet', index=False)


