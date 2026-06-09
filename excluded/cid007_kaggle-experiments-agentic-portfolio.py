from kaggle_secrets import UserSecretsClient
import os, sys
from pathlib import Path
from IPython.display import Markdown, display

# Load API keys
secret = UserSecretsClient().get_secret("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = secret
os.environ["GOOGLE_API_KEY"] = secret

# Attach project from dataset path
project_root = Path("/kaggle/input/kaggle-experiment-orchestrator-lite/kaggle_experiment_orchestrator")
sys.path.append(str(project_root))

from orchestrator_agent import adk_tools
from orchestrator_agent.notebook_agent import answer_question as answer_question_kaggle
from orchestrator_agent.template_agent import generate_notebook_template
from orchestrator_agent.viz import (
    plot_cv_vs_holdout, plot_time_vs_cv, plot_model_family_performance
)
from orchestrator_agent.ranking import rank_experiments

project_root



# Get dummy experiment data
import pandas as pd

csv_path = project_root / "data" / "sample_experiments.csv"
df = pd.read_csv(csv_path)

print(df.shape)
df.head()



result = adk_tools.tool_run_portfolio_analysis(str(csv_path))
result



plot_cv_vs_holdout(df)
plot_time_vs_cv(df)
plot_model_family_performance(result["summary"])




display(Markdown(answer_question_kaggle(
    "Summarize the strengths and weaknesses of my experiment portfolio.",
    experiments_path=str(csv_path)
)))




display(Markdown(answer_question_kaggle(
    "Where am I overfitting and why?",
    str(csv_path)
)))



sug = adk_tools.tool_suggest_next_experiments(str(csv_path))
sug




display(Markdown(answer_question_kaggle(
    "Here are the auto-generated suggestions.\nExplain them and propose 2 more creative ones:\n\n"
    + str(sug),
    experiments_path=str(csv_path)
)
))



ranked_balanced = rank_experiments(df, "balanced")
ranked_lb = rank_experiments(df, "leaderboard")
ranked_stable = rank_experiments(df, "stability")
ranked_speed = rank_experiments(df, "speed")

ranked_lb



display(Markdown(answer_question_kaggle(
    "Which ranking strategy should I use based on my goals?",
    str(csv_path)
)))



'''
template = generate_notebook_template(
    competition_name="Playground S5E11 - Loan Default",
    primary_metric="AUC",
    target_column="default"
)

print("\n".join(template.split("\n")[:500]))   # preview first few lines
'''
template = generate_notebook_template(
    competition_name="Titanic - Machine Learning from Disaster",
    primary_metric="Accuracy",
    target_column="Survived"
)

#print("\n".join(template.split("\n")[:200]))   # preview first few lines

display(Markdown("\n".join(template.split("\n")[:200])))



