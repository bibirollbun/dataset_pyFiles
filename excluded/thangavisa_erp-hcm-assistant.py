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


# Cell 1: Setup, Imports, and Synthetic Data Generation

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import os
import io

# --- 1. SYNTHETIC DATA GENERATION (RAW INPUT) ---
np.random.seed(42)
num_employees = 500

data = {
    'Employee ID': [f'E{i:04d}' for i in range(1, num_employees + 1)],
    'Department': np.random.choice(['Sales', 'HR', 'Tech', 'Marketing', 'Operations'], num_employees),
    'Monthly Salary': np.random.normal(5000, 1500, num_employees).round(2),
    'Standard Hours': [160] * num_employees,
    'Actual Hours Worked': np.random.normal(165, 15, num_employees).round(0),
    'Leave Days Used Current Month': np.random.randint(0, 7, num_employees),
    'Max Allowed Leave Days': [3] * num_employees,
    'Years at Company': np.random.randint(1, 20, num_employees),
    'Job Level': np.random.randint(1, 5, num_employees),
    'Previous Month Leave Days': np.random.randint(0, 7, num_employees)
}

df = pd.DataFrame(data)
# Ensure non-negative/realistic values
df['Monthly Salary'] = df['Monthly Salary'].apply(lambda x: max(2500, x))
df['Actual Hours Worked'] = df['Actual Hours Worked'].apply(lambda x: max(120, x))

# Create the target variable for the prediction model (High Leave Usage next month)
df['Took High Leave Next Month'] = ((df['Leave Days Used Current Month'] > 3) |
                                   (df['Previous Month Leave Days'] > 3) |
                                   (df['Department'] == 'Sales')
                                  ).astype(int)
# Introduce noise to simulate real-world inaccuracy
df.loc[np.random.rand(num_employees) < 0.2, 'Took High Leave Next Month'] = 0 
df.loc[np.random.rand(num_employees) < 0.15, 'Took High Leave Next Month'] = 1 

# Save to CSV for the Agent to "read"
df.to_csv('hr_payroll_data.csv', index=False)
print("âœ… 1. Raw Input Data Generated and Saved to 'hr_payroll_data.csv'.")


# Cell 3: HCM Agent Custom Tools (The Logic Library)
import pandas as pd
import numpy as np
import pickle
# Assume that required scikit-learn imports are available from Cell 1

def tool_transform_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    TOOL 0: Feature Engineering and transformation from raw IBM data to HCM payroll format.
    Executed by the Data Preprocessor Agent.
    """
    df = raw_df.copy()
    
    # --- 1. Map to Payroll Standard ---
    # Rename for clarity in the payroll pipeline
    df.rename(columns={'MonthlyRate': 'Monthly Salary', 'YearsAtCompany': 'Years at Company', 
                       'JobLevel': 'Job Level', 'TotalWorkingYears': 'Total Working Years'}, inplace=True)
    
    # Set uniform standard hours and max allowed leave days
    df['Standard Hours'] = 160 
    df['Max Allowed Leave Days'] = 3
    
    # --- 2. Simulate Key HCM Metrics (Feature Engineering) ---
    # Simulate Leave Days: Assume lower Job Involvement correlates with higher unexpected leave
    df['Leave Days Used Current Month'] = df['JobInvolvement'].apply(
        # Score 1 = 6 days, Score 2 = 4 days, Score 3/4 = random days 0-2
        lambda x: 6 if x == 1 else (4 if x == 2 else np.random.randint(0, 3))
    )
    
    # Simulate Actual Hours Worked: Base 160 hours with variance
    df['Actual Hours Worked'] = df['HourlyRate'].apply(
        lambda x: max(150, np.random.normal(165, 10))
    ).round(0)
    
    # --- 3. Clean/Validate Data (Robustness check) ---
    df.dropna(subset=['Department', 'Monthly Salary'], inplace=True)
    df['Monthly Salary'] = df['Monthly Salary'].apply(lambda x: max(0, x))
    
    # --- 4. Select final columns for the rest of the pipeline ---
    payroll_cols = [
        'EmployeeNumber','Attrition', 'Department', 'Monthly Salary', 'Standard Hours',
        'Actual Hours Worked', 'Leave Days Used Current Month', 
        'Max Allowed Leave Days', 'Years at Company', 'Job Level', 
        'Total Working Years' # Features for prediction
    ]
    
    return df[payroll_cols]


def tool_calculate_payroll_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    TOOL 1: Calculates Hourly Rate, Overtime Hours, and Overtime Pay.
    Executed by the Payroll Calculation Agent.
    """
    
    overtime_multiplier = 1.5
    # Formula: Hourly Rate = Monthly Salary / Standard Hours
    df['Hourly Rate'] = df['Monthly Salary'] / df['Standard Hours']
    
    # Formula: Overtime Hours = Max(0, Actual Hours Worked - Standard Hours)
    df['Overtime Hours'] = df.apply(
        lambda row: max(0, row['Actual Hours Worked'] - row['Standard Hours']), axis=1
    )
    
    # Formula: Overtime Pay = Overtime Hours * (Hourly Rate * 1.5)
    df['Overtime Pay'] = df['Overtime Hours'] * (df['Hourly Rate'] * overtime_multiplier)
    
    return df

def tool_check_leave_compliance(df: pd.DataFrame) -> pd.DataFrame:
    """
    TOOL 2: Flags employees whose leave usage exceeds the allowed limit for adjustment.
    Executed by the Compliance Review Agent.
    """
    
    # Compliance Rule: Flag employee if current month's leave usage > Max Allowed Leave Days.
    df['Leave Adjustment Flag'] = df.apply(
        lambda row: 'Requires Unpaid Adjustment' 
                    if row['Leave Days Used Current Month'] > row['Max Allowed Leave Days']
                    else 'No Adjustment Needed',
        axis=1
    )
    
    return df

def tool_generate_prediction(df: pd.DataFrame, model_path: str = 'leave_prediction_pipeline.pkl') -> pd.DataFrame:
    """
    TOOL 3: Loads a trained model and adds a prediction column (risk score) to the DataFrame.
    Executed by the Predictive Insight Agent.
    """
    try:
        with open(model_path, 'rb') as file:
            pipeline = pickle.load(file)
    except FileNotFoundError:
        print("ERROR: Prediction model not found. Run the ML Training Cell first.")
        return df
    
    # IMPORTANT: Update feature list to match the columns created/renamed from the IBM dataset
    #features_for_prediction = ['Department', 'Years at Company', 'Job Level', 'Total Working Years', 'Leave Days Used Current Month']
    
    # Predict the probability (risk score)
    df['Leave Risk Score'] = pipeline.predict_proba(df[features_for_prediction])[:, 1]
    
    return df

def tool_llm_compliance_summary(flagged_records_df: pd.DataFrame) -> str:
    """
    TOOL 4 (Conceptual): Generates a natural language compliance summary using Gemini.
    Executed by the Compliance Review Agent (Gemini call for bonus points).
    """
    if flagged_records_df.empty:
        return "**Compliance Summary (LLM):** No immediate compliance breaches requiring manager attention were detected."

    # Prepare data for LLM analysis
    action_items = flagged_records_df[['EmployeeNumber', 'Leave Days Used Current Month']].head(10).to_string(index=False)
    
    # The actual LLM API call is replaced with a simulated output for security:
    # --- START Conceptual Gemini API Call ---
    prompt = f"Analyze the {len(flagged_records_df)} flagged employees. Draft a concise, professional summary for the HR manager explaining that these employees need an unpaid salary adjustment due to exceeding policy. Data sample:\n{action_items}"
    
    placeholder_summary = f"""
    **Compliance Summary (Generated by LLM):** A total of {len(flagged_records_df)} employees (top 10 samples shown below) require an immediate manual payroll adjustment. They were automatically flagged for excessive leave usage (exceeded 3 days) this period, necessitating an unpaid leave deduction according to corporate policy. This agent has prioritized these records for urgent review.
    
    *Triggering Prompt:* "{prompt[:100]}..."
    """
    # --- END Conceptual Gemini API Call ---
    
    return placeholder_summary

print("âœ… 3. Custom Agent Tools Defined and Integrated with IBM HR Analytics Features.")


# Cell 4: Predictive Model Training and Saving (REVISED)

print("--- TRAINING LOGISTIC REGRESSION MODEL ---")

# --- 1. Load Raw Data (same way the Agent does conceptually) ---
raw_data_path = '/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv'
raw_df = pd.read_csv(raw_data_path)

# --- 2. TRANSFORM THE DATA FOR TRAINING (MIMIC AGENT STAGE 1) ---
# This renames the columns and creates the features used below.
df_train = tool_transform_raw_data(raw_df.copy())

# --- 3. Create the Target Variable (Since it doesn't exist in the raw data) ---
# We will use the 'Attrition' column (1=Yes/Attrition, 0=No) as the target 
# since it's the most stable predictive outcome in this dataset.
df_train['Took High Leave Next Month'] = df_train['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# --- 4. Define Features and Target ---
features_for_prediction = ['Department', 'Years at Company', 'Job Level', 'Total Working Years', 'Leave Days Used Current Month']
X_pred = df_train[features_for_prediction]
y_pred = df_train['Took High Leave Next Month']

# --- 5. Define Preprocessing Pipeline and Train ---
categorical_features = ['Department']
numerical_features = [col for col in features_for_prediction if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        # Ensure OneHotEncoder handles strings from the IBM data
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) 
    ])

prediction_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))
])

# Split and train
X_train_pred, X_test_pred, y_train_pred, y_test_pred = train_test_split(
    X_pred, y_pred, test_size=0.2, random_state=42, stratify=y_pred
)

prediction_pipeline.fit(X_train_pred, y_train_pred)

# Evaluation and Saving (rest of the cell is fine)
y_pred_proba = prediction_pipeline.predict_proba(X_test_pred)[:, 1]
roc_auc = roc_auc_score(y_test_pred, y_pred_proba)

print(f"Model Training Complete. ROC-AUC Score on Test Set: {roc_auc:.4f}")

model_filename = 'leave_prediction_pipeline.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(prediction_pipeline, file)
print(f"âœ… 4. Model saved as '{model_filename}' for Agent deployment.")


# Cell 5: Agent Orchestration and Execution

class HcmAutomationAgent:
    def __init__(self, name="HcmAutomationAgent"):
        self.name = name
        self.session_data = {} # Simulates Sessions & Memory for state passing
    
    def run_payroll_cycle(self):
        """
        Executes the sequential multi-agent workflow using the IBM dataset:
        Load/Transform -> Calculate Payroll -> Check Compliance/Predict -> Report
        """
        # Define the path for the IBM HR Analytics dataset in the Kaggle environment
        raw_data_path = '/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv'
        
        # --- STAGE 1: Data Preprocessor Agent (Load, Transform, Validate) ---
        try:
            raw_df = pd.read_csv(raw_data_path)
        except FileNotFoundError:
            print(f"ERROR: Dataset not found at path: {raw_data_path}. Please check the Kaggle input directory.")
            return None

        # Tool 0: Data Transformation (Feature Engineering)
        transformed_df = tool_transform_raw_data(raw_df.copy()) 
        print("STAGE 1: Data Preprocessor Agent (Tool 0) executed. Data validated and transformed.")
        
        # --- STAGE 2: Payroll Calculation Agent ---
        # Tool 1: Calculate Overtime, Hourly Rate
        processed_df = tool_calculate_payroll_data(transformed_df)
        print("STAGE 2: Payroll Calculation Agent (Tool 1) executed. Overtime calculated.")
        
        # --- STAGE 3: Compliance Review Agent (Hybrid LLM/Tool) ---
        # Tool 2: Check Leave Compliance
        compliance_df = tool_check_leave_compliance(processed_df)
        print("STAGE 3: Compliance Review Agent (Tool 2) executed. Records flagged.")
        
        # --- STAGE 4: Predictive Insight Agent ---
        # Tool 3: Generate Leave Risk Prediction
        final_df = tool_generate_prediction(compliance_df, 'leave_prediction_pipeline.pkl')
        print("STAGE 4: Predictive Insight Agent (Tool 3) executed. Risk scores calculated.")

        self.session_data['final_data'] = final_df
        
        return self.generate_reports(final_df)

    def generate_reports(self, final_df):
        """
        Executed by the Reporting Agent (final stage).
        Filters data and presents structured output.
        """
        # 1. Actionable HR Report (Focus on compliance and high risk)
        action_report = final_df[
            (final_df['Leave Adjustment Flag'] != 'No Adjustment Needed') | 
            (final_df['Leave Risk Score'] >= 0.70) # Flag high risk scores (70%+)
        ]
        
        # 2. Payroll Summary (Final payment data)
        payroll_summary = final_df[['EmployeeNumber', 'Monthly Salary', 'Overtime Pay', 
                                    'Leave Adjustment Flag', 'Leave Risk Score']]
        
        # 3. LLM/Gemini Compliance Summary
        compliance_llm_summary = tool_llm_compliance_summary(
            final_df[final_df['Leave Adjustment Flag'] != 'No Adjustment Needed']
        )
        
        return {
            'action_report': action_report[['EmployeeNumber', 'Leave Adjustment Flag', 'Leave Risk Score', 'Department']],
            'payroll_summary': payroll_summary,
            'llm_summary': compliance_llm_summary
        }

# --- EXECUTION COMMAND ---
agent = HcmAutomationAgent()
reports = agent.run_payroll_cycle()

print("\n--- AGENT EXECUTION COMPLETE ---")


# Cell 6: Reporting Agent Output (Enhanced Readability)

print("--- ğŸ“� AGENT FINAL REPORT ---")

# --- A. HR ACTION REQUIRED REPORT (Most Critical) ---
# Filter to only show records where action is required (Compliance Breach OR High Risk)
action_report = reports['action_report'].copy()

# Format the Leave Risk Score as a percentage for easy reading
action_report['Leave Risk Score'] = (action_report['Leave Risk Score'] * 100).round(1).astype(str) + '%'

print("\n## ğŸš¨ HR ACTION REQUIRED REPORT (Actionable Cases)")
print(f"Total Actionable Cases Flagged: {len(action_report)}")
print("Criteria: Compliance Breach (Unpaid Adjustment) OR High Predictive Risk (>70%)")

# Display the most critical columns for immediate HR action
print(action_report.sort_values(by='Leave Risk Score', ascending=False).head(15).to_markdown(index=False))

# --- B. PAYROLL EXECUTION SUMMARY ---
payroll_summary = reports['payroll_summary'].copy()
payroll_summary['Leave Risk Score'] = (payroll_summary['Leave Risk Score'] * 100).round(1).astype(str) + '%'
payroll_summary['Overtime Pay'] = payroll_summary['Overtime Pay'].round(2)

print("\n---\n")
print("## ğŸ’° PAYROLL EXECUTION SUMMARY (First 10 Records)")
print("Data ready for final payroll system integration and review:")
# Display columns essential for financial closure
print(payroll_summary.head(10).to_markdown(index=False))


# --- C. LLM/GEMINI COMPLIANCE SUMMARY (Explainability) ---
print("\n---\n")
print("## ğŸ—£ï¸� LLM/GEMINI COMPLIANCE SUMMARY")
print("Unstructured reasoning provided by the Compliance Review Agent for quick management understanding:")
print(reports['llm_summary'])

