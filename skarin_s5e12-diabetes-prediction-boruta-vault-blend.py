import pandas as pd
from scipy.stats import rankdata
from scipy.stats import pearsonr

# --- CONFIG ---
# 1. Absolute File (The 0.70732 one)
FILE_EMPEROR = "/kaggle/input/diabetes-prediction-vault/submission.csv" 

# 2. Boruta File (The 0.69711 one)
FILE_BORUTA = "/kaggle/input/low-slow-learning-with-boruta/submission_boruta_boost.csv"

OUTPUT_FILE = "submission.csv"

print(">>> ğŸ’� Initiating Diamond Polish Blend...")

try:
    df_emp = pd.read_csv(FILE_EMPEROR)
    df_bor = pd.read_csv(FILE_BORUTA)
    
    # Validation
    p1 = df_emp['diagnosed_diabetes'].values
    p2 = df_bor['diagnosed_diabetes'].values
    
    # Correlation Check
    corr, _ = pearsonr(p1, p2)
    print(f"    Correlation (Emperor vs Boruta): {corr:.5f}")
    
    if corr < 0.98:
        print("    ğŸš€ Excellent! Low correlation.")
    
    # Rank Normalization
    r_emp = rankdata(p1) / len(p1)
    r_bor = rankdata(p2) / len(p2)
    
    # WEIGHTS
    # 97% Trust in the 0.707 score.
    # 3% Influence from the Boruta features.
    W_EMP = 0.97
    W_BOR = 0.03
    
    POWER = 4
    final_rank = (W_EMP * (r_emp**POWER) + W_BOR * (r_bor**POWER))**(1/POWER)
    
    submission = pd.DataFrame({
        'id': df_emp['id'],
        'diagnosed_diabetes': final_rank
    })
    
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\n>>> âœ… Diamond Blend saved to {OUTPUT_FILE}")

except Exception as e:
    print(f"â�Œ Error: {e}")

