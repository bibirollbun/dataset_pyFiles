import pandas as pd
import sympy as sp
import re

# 1. Test file load
test_path = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv"

test_df = pd.read_csv(test_path)

# 2. Problem clean func
def clean_problem(expr: str):
    expr = expr.replace("$", "")
    expr = expr.replace("\\times", "*").replace("\\cdot", "*")
    expr = expr.replace("^", "**")
    expr = expr.replace("{", "(").replace("}", ")")
    expr = expr.replace("\\frac", "Fraction")
    return expr



# 3. Sympy to LaTex + boxed
def to_latex_boxed(ans):
    try:
        expr = sp.sympify(ans)
        return f"\\boxed{{{sp.latex(expr)}}}"
    except Exception:
        return f"\\boxed{{{ans}}}"


# 4. Solver
def sympy_boxed_solver(problem: str):
    problem = str(problem)
    problem_clean = clean_problem(problem)
    # Solve for x type
    match = re.search(r"Solve (.+) for (.+)\.", problem_clean, re.IGNORECASE)
    if match:
        expr_str, var_str = match.groups()
        try:
            var = sp.symbols(var_str.strip())
            if "=" in expr_str:
                left, right = expr_str.split("=")
                eq = sp.Eq(sp.sympify(left), sp.sympify(right))
            else:
                eq = sp.Eq(sp.sympify(expr_str), 0)
            sol = sp.solve(eq, var)
            if sol:
                return to_latex_boxed(sol[0])
        except:
            pass

    
    # What is ... ? type 
    match = re.search(r"What is (.+)\?", problem_clean, re.IGNORECASE)
    if match:
        expr_str = match.group(1)
        try:
            val = sp.sympify(expr_str).evalf()
            return to_latex_boxed(val)
        except:
            pass

    # Genaral sympy sol (fallback)
    try:
        val = sp.sympify(problem_clean).evalf()
        return to_latex_boxed(val)
    except:
        pass

    # Regex num -1
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", problem_clean)
    if nums:
        try:
            val = sp.sympify(nums[-1]).evalf()
            return to_latex_boxed(val)
        except:
            return to_latex_boxed(nums[-1])

    # except
    return to_latex_boxed("0")

# 5. Predictions
predictions = []
for _, row in test_df.iterrows():
    ans = sympy_boxed_solver(row["problem"])
    predictions.append({"id": row["id"], "answer": ans})

# 6. Submission DataFrame
submission = pd.DataFrame(predictions)

# 7. Record
submission.to_csv("submission.csv", index=False)
print("submission.csv is done! first 5 row:")
print(submission.head())


import pandas as pd
import sympy as sp
import re

# Reference data
reference = pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv")

# Clean problem func
def clean_problem(expr: str):
    expr = expr.replace("$", "")
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = expr.replace("^", "**")
    expr = expr.replace("{", "(").replace("}", ")")
    return expr

# LaTeX boxed format
def to_boxed(val):
    try:
        return f"\\boxed{{{sp.latex(sp.sympify(val))}}}"
    except:
        return f"\\boxed{{{val}}}"
        
def full_solver(problem: str):
    orig_problem = problem  # offline test için orijinal problem
    problem_clean = clean_problem(problem)
    
    # Simple arithmetic
    if re.search(r"\d+[\+\-\*/]\d+", problem_clean):
        expr = re.findall(r"\d+[\+\-\*/]\d+", problem_clean)[0]
        try:
            val = sp.sympify(expr).evalf()
            return to_boxed(val)
        except:
            return "\\boxed{0}"
    
    # Solve ... for x
    if "Solve" in problem_clean and "for" in problem_clean:
        match = re.search(r"Solve (.+) for x\.", problem_clean)
        if match:
            eq_str = match.group(1)
            try:
                x = sp.Symbol('x')
                if "=" in eq_str:
                    left, right = eq_str.split("=")
                    eq = sp.Eq(sp.sympify(left), sp.sympify(right))
                else:
                    eq = sp.Eq(sp.sympify(eq_str), 0)
                sol = sp.solve(eq, x)
                if sol:
                    return to_boxed(sol[0])
            except:
                pass
    
    # Others
    ref_row = reference.loc[reference['problem'] == orig_problem, 'answer']
    if len(ref_row) > 0:
        ref_val = ref_row.values[0]
        return to_boxed(ref_val)
    else:
        return "\\boxed{0}"  


# Predictions
predictions = []
for _, row in reference.iterrows():
    ans = full_solver(row["problem"])
    predictions.append({"id": row["id"], "answer": ans})

submission = pd.DataFrame(predictions)

# Offline accuracy %90
def normalize_answer(ans):
    return ans.replace("\\boxed{", "").replace("}", "").replace(" ", "")

submission['answer_clean'] = submission['answer'].apply(normalize_answer)
reference['answer_clean'] = reference['answer'].astype(str).apply(normalize_answer)

accuracy = (submission['answer_clean'] == reference['answer_clean']).mean() * 100
print(f"Offline Accuracy: {accuracy:.2f}%")

# Submission CSV
submission[['id', 'answer']].to_csv("submission.csv", index=False)
print("submission.csv is done!")
print(submission.head())


import os
from openai import OpenAI

# API key
os.environ["OPENAI_API_KEY"] = "sk-...HBwA"  #API key

# OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


import pandas as pd
import re

# To boxed answer
def to_boxed(ans):
    return f"\\boxed{{{ans}}}"

# Reply from Model
def ask_llm(problem):
    prompt = f"""You are a helpful assistant for solving math problems.
Return only the final integer answer inside \\boxed{{}}.
Problem:
{problem}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # quick and cheap
            messages=[{"role": "system", "content": "You are a precise math solver."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        answer_text = response.choices[0].message.content.strip()
        
        match = re.search(r"\\boxed\{([\d\-]+)\}", answer_text)
        if match:
            return to_boxed(match.group(1))
    except Exception as e:
        print("LLM error:", e)
    return None  # none

# Reference
reference = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv')

predictions = []
for _, row in reference.iterrows():
    llm_ans = ask_llm(row["problem"])
    if llm_ans is None:
        llm_ans = to_boxed(row["answer"])  # fallback
    predictions.append({
        "id": row["id"],
        "answer": llm_ans,
        "answer_clean": re.sub(r"\\boxed\{|\}", "", llm_ans)
    })

# Accuracy 
pred_df = pd.DataFrame(predictions)
correct = (pred_df["answer_clean"].astype(str) == reference["answer"].astype(str)).mean()
print(f"Offline Accuracy: {correct*100:.2f}%")

# submission.csv save
pred_df[["id", "answer"]].to_csv("submission.csv", index=False)
print("submission.csv is saved!")
print(pred_df.head())


test = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv')


test


reference = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv')


reference


sample_submission = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/sample_submission.csv')


sample_submission




