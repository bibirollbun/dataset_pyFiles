# Explainable Reasoning Tutor Agent â€“ Kaggle Capstone Submission
# Track: Agents for Good | Author: DHRUBA SHIL

from sympy import symbols, Eq, solve, simplify, expand
from sympy.parsing.sympy_parser import parse_expr

# Session memory
session_history = []

# Tool: Math parser + solver
def parse_and_solve(problem_text):
    try:
        # Define common symbols
        x, y, z = symbols('x y z')

        # Split by commas to detect multiple equations
        equations = [eq.strip() for eq in problem_text.lower().replace('^', '**').split(',')]
        parsed_eqs = []

        for eq in equations:
            if '=' in eq:
                left, right = eq.split('=')
                parsed_eqs.append(Eq(parse_expr(left), parse_expr(right)))
            else:
                return [{"step": "âš ï¸� Please include '=' in each equation."}]

        # Solve system of equations
        solution = solve(parsed_eqs, dict=True)
        steps = [{"step": f"Parsed equations: {parsed_eqs}"}]

        if solution:
            for var in solution[0]:
                steps.append({"step": f"Solved: {var} = {solution[0][var]}"})
        else:
            steps.append({"step": "âš ï¸� No solution found or system is inconsistent."})

        return steps

    except Exception as e:
        return [{"step": f"âš ï¸� Error parsing or solving: {str(e)}"}]

# Tool: Visualizer
def visualize_steps(steps):
    print("\nğŸ§  Step-by-Step Explanation:")
    for i, step in enumerate(steps):
        print(f"Step {i+1}: {step['step']}")
    print()

# Tool: Feedback collector
def collect_feedback():
    feedback = input("Was this explanation helpful? (yes/no): ")
    print("âœ… Thanks for your feedback!" if feedback.lower() == "yes" else "âš ï¸� We'll try to improve.")

# Agent logic
def reasoning_agent(problem_text):
    steps = parse_and_solve(problem_text)
    visualize_steps(steps)
    collect_feedback()
    session_history.append({"problem": problem_text, "steps": steps})

# Main loop
print("ğŸ“š Explainable Reasoning Tutor Agent")
print("Helping students understand math problems through step-by-step reasoning.\n")

while True:
    user_input = input("Enter a math problem (e.g. '2*x + 3 = 7' or 'x + y = 4, x - y = 0') or type 'exit': ")
    if user_input.lower() == "exit":
        break
    reasoning_agent(user_input)

# Show session history
print("\nğŸ“œ Session History:")
for entry in session_history:
    print(f"- Problem: {entry['problem']}")

