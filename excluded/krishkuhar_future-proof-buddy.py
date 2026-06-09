import numpy as np # linear algebra
import pandas as pd # data processing, CSV I/O (e.g., pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- Agent Configuration & Core Data ---
# Prometheus Style Metrics (Internal Tracing/Observability)
METRICS = {
    "sleep_debt_total": 0.0,
    "distraction_minutes": 0,
    "money_outflow_rate": 0.0,
    "deadline_collision_risk": 0.0,
}

# --- 4. A2A Protocol Implementation (Simulated) ---

class AnalysisPayload:
    """A structured data object representing the Agent-to-Agent (A2A) protocol."""
    def __init__(self, risk_map, crash_point, trend_alert, metrics):
        self.risk_map = risk_map
        self.crash_point = crash_point
        self.trend_alert = trend_alert
        self.metrics = metrics

def A2A_Transfer(payload: AnalysisPayload, sender: str, target: str):
    """Simulates the secure serialization and transfer of a payload."""
    print(f" Â  [A2A Protocol] {sender} sending structured AnalysisPayload to {target}...")
    return payload

print("Environment Setup and A2A Protocol Defined successfully.")


# --- Synthetic Data Generation ---
def generate_synthetic_data(days=14):
    """Generates 14 days of simulated user behavioral data."""
    dates = [datetime.now() - timedelta(days=d) for d in range(days - 1, -1, -1)]
    
    data = {
        'date': dates,
        'actual_sleep_hrs': np.clip(np.random.normal(6.5, 0.8, days) + np.linspace(0.5, 0, days), 4.5, 8.5),
        'screen_time_hrs': np.clip(np.random.normal(4.5, 1.2, days), 3, 8),
        'study_output_score': np.clip(np.random.normal(60, 15, days), 30, 95),
        'daily_spending': np.clip(np.random.normal(40, 15, days), 10, 80),
        'mood_score': np.clip(np.random.normal(7, 1.5, days), 4, 10),
        'deadlines_in_7d': [2, 1, 0, 3, 2, 1, 0, 1, 2, 0, 1, 2, 3, 1] 
    }
    df = pd.DataFrame(data)
    df['day_of_week'] = df['date'].dt.day_name()
    return df

# --- 1. DataCollector Agent ---
class DataCollector:
    """Sequential Agent responsible for cleaning, transforming, and calculating base metrics."""
    def __init__(self, data):
        self.df = data
        self.ideal_sleep = 8.0 # User defined ideal

    def run_collection(self):
        print(">> DataCollector: Processing raw behavioral streams...")
        
        # Calculate Sleep Debt
        self.df['sleep_debt_day'] = self.ideal_sleep - self.df['actual_sleep_hrs']
        METRICS['sleep_debt_total'] = self.df['sleep_debt_day'].tail(7).sum() # 7-day rolling debt
        
        # Calculate Distraction Minutes
        self.df['distraction_minutes'] = (self.df['screen_time_hrs'] * 60 * 0.35).round(0)
        METRICS['distraction_minutes'] = int(self.df['distraction_minutes'].tail(3).mean()) # 3-day average
        
        # Calculate Financial Outflow Rate
        METRICS['money_outflow_rate'] = self.df['daily_spending'].tail(7).mean().round(2)
        
        print(f" Â  Calculated 7-day Sleep Debt: {METRICS['sleep_debt_total']:.1f} hrs")
        return self.df

# --- RUN BLOCK 2 ---
df = generate_synthetic_data(days=14)
collector = DataCollector(df)
processed_df = collector.run_collection()

print("\n--- DataCollector Output Sample (Last 5 days) ---")
print(processed_df[['date', 'actual_sleep_hrs', 'sleep_debt_day']].tail())


# --- 2. BehaviorPredictor Agent ---
class BehaviorPredictor:
    """Sequential Agent responsible for risk assessment and pattern detection."""
    def __init__(self, processed_data):
        self.df = processed_data
        self.RISK_MAP = {}
        self.weekly_risk = 0

    def calculate_risk_scores(self):
        print(">> BehaviorPredictor: Calculating current risk profile...")

        latest_data = self.df.iloc[-1]
        
        # Risk Score Calculations depend on the global METRICS populated by DataCollector
        self.RISK_MAP['Sleep Risk'] = np.clip((METRICS['sleep_debt_total'] * 5) + (10 - latest_data['mood_score']) * 5, 0, 100).round(0)
        self.RISK_MAP['Finance Risk'] = np.clip((METRICS['money_outflow_rate'] / 50) * 100, 0, 100).round(0)
        deadline_risk = (latest_data['deadlines_in_7d'] / 3) * 50
        self.RISK_MAP['Burnout Risk'] = np.clip((self.RISK_MAP['Sleep Risk'] * 0.5) + deadline_risk, 0, 100).round(0)
        self.RISK_MAP['Productivity Risk'] = np.clip((METRICS['distraction_minutes'] / 180) * 50 + (100 - latest_data['study_output_score']) * 0.5, 0, 100).round(0)
        
        self.weekly_risk = np.mean(list(self.RISK_MAP.values())).round(1)
        print(f" Â  Calculated Overall Weekly Risk Score: {self.weekly_risk}")
        
    def detect_trend_alert(self):
        """Simulates 'TrendAlertsâ„¢' by finding repeating low-output days."""
        low_output_days = self.df[self.df['study_output_score'] < 50]['day_of_week'].value_counts()
        
        if not low_output_days.empty and low_output_days.iloc[0] >= 2:
            most_common_day = low_output_days.index[0]
            count = low_output_days.iloc[0]
            return f"New TrendAlertsâ„¢ detected: You hit a **Productivity Wall** every **{most_common_day}** ({count} times recently). This is a fixable curse."
        return None

    def future_crashpoint_detector(self):
        """Tool (Code Execution): Uses rolling regression to simulate predicting a crash."""
        X = self.df[['sleep_debt_day', 'distraction_minutes', 'deadlines_in_7d']].tail(7)
        y = self.df['study_output_score'].tail(7)

        if len(X) < 5: return None

        model = LinearRegression()
        model.fit(X, y)
        
        latest_inputs = X.iloc[-1].values.reshape(1, -1)
        future_input_2 = latest_inputs * 1.15 # Simulate a worsening state
        future_outputs = model.predict(future_input_2.reshape(1, -1))
        
        if future_outputs[0] < 40:
            crash_time = datetime.now() + timedelta(hours=48)
            return {
                "time": crash_time.strftime('%I:%M %p, %a'),
                "score": future_outputs[0].round(0),
            }
        return None

    def run_prediction(self):
        self.calculate_risk_scores()
        trend_alert = self.detect_trend_alert()
        crash_point = self.future_crashpoint_detector()
        
        # Use A2A Protocol to send analysis to the Adviser Agent
        payload = AnalysisPayload(self.RISK_MAP, crash_point, trend_alert, METRICS)
        return A2A_Transfer(payload, "BehaviorPredictor", "Adviser Agent")

# --- RUN BLOCK 3 ---
predictor = BehaviorPredictor(processed_df)
analysis_payload = predictor.run_prediction()

print("\n--- BehaviorPredictor Output (Analysis Payload Snapshot) ---")
print(f"Risk Map: {analysis_payload.risk_map}")
print(f"Trend Alert: {analysis_payload.trend_alert}")
print(f"Crash Point: {analysis_payload.crash_point}")


# --- 3. Adviser Agent (LLM Simulation) ---
def LLM_Advisor_Reasoning(payload: AnalysisPayload, df_full):
    """Agent powered by an LLM: Decides persona, generates warning, and provides the micro-plan."""
    risk_map = payload.risk_map
    metrics = payload.metrics
    crash_point = payload.crash_point
    trend_alert = payload.trend_alert
    
    # Determine Style (Mood-based advice style)
    # df_full is passed here as a simulated way to access necessary context (mood score)
    mood_score = df_full['mood_score'].iloc[-1]
    
    highest_risk_category = max(risk_map, key=risk_map.get)
    highest_risk_score = risk_map[highest_risk_category]
    
    style = "Chill" if mood_score > 6 and highest_risk_score < 70 else "Stressed"

    print(f">> Adviser Agent: Reasoning with detected style: '{style}'")

    if highest_risk_score >= 75:
        if style == "Chill":
            warning = f"Heads up, chief. Your **{highest_risk_category}** is spiking at {highest_risk_score}/100. You've got {metrics['sleep_debt_total']:.1f} hours of sleep debt. That's a disaster in slow motion."
            plan = "Micro-Plan: Finish whatever you're doing in the next 30 mins, and force a 7-hour bedtime. Tomorrow's work depends on it."
        else:
            warning = f"**EMERGENCY WARNING: {highest_risk_category} is critical ({highest_risk_score}/100).** This isn't sustainable. You are approaching a **Burnout Crash**."
            plan = "Micro-Plan: Immediate 15-minute complete disconnect (no screen, no work). Go drink water, look outside, and reschedule your lowest priority task for tomorrow."
    
    elif crash_point:
        warning = f"Future CrashPoint Detector flag: You're on track for a productivity score of **{crash_point['score']}** around **{crash_point['time']}**. Your current habits are predicting a wipeout."
        plan = f"Micro-Plan: The fix is simple: 20 minutes of focused effort on {highest_risk_category.replace(' Risk', '')} now, followed by a guaranteed {int(metrics['distraction_minutes'] * 0.5)} minutes of low-cost, screen-free rest."

    elif trend_alert:
        warning = trend_alert
        plan = f"Micro-Plan: Want to fix the curse? Proactively block off 1 hour of 'deep work' on {trend_alert.split('**')[2].strip()} morning *before* that wall hits."
        
    else:
        warning = f"Everything looks steady, but I see you spent {metrics['distraction_minutes']} minutes scrolling today. That's like setting money on fire. Projected waste: {metrics['distraction_minutes'] * 0.1} minutes of productive time tomorrow."
        plan = "Micro-Plan: Close the app you used the most right now. Reward yourself with a 15-minute intentional break later."

    return {"warning": warning, "plan": plan, "risk_map": risk_map}

# --- Visualization ---
def plot_risk_map(risk_map):
    """Generates the Personal Risk Map Bar Chart."""
    risk_df = pd.DataFrame(list(risk_map.items()), columns=['Category', 'Risk Score'])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Risk Score', y='Category', data=risk_df, palette='viridis')
    plt.title('Personal Risk Map (0-100)', fontsize=16, color='white')
    plt.xlabel('Risk Score', fontsize=14, color='white')
    plt.ylabel('Risk Category', fontsize=14, color='white')
    
    # Customizing axes for dark mode readability
    plt.gca().set_facecolor('#333333')
    plt.gcf().set_facecolor('#333333')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.grid(axis='x', alpha=0.2, color='white')
    plt.xlim(0, 100)
    
    # Add risk scores to bars
    for index, row in risk_df.iterrows():
        plt.text(row['Risk Score'] + 1, index, f"{row['Risk Score']}", color='white', ha="left", va="center")
        
    plt.tight_layout()
    plt.show()

# --- RUN BLOCK 4 ---
advice = LLM_Advisor_Reasoning(analysis_payload, df)

print("\n" + "="*80)
print(" Â  Â  Â  Â  Â  âš¡ï¸� TrendAlertsâ„¢ & Future CrashPoint Detector Warning âš¡ï¸�")
print("="*80)
print(f"**WARNING:** {advice['warning']}")
print(f"\n**Micro-Plan:** {advice['plan']}")
print("="*80)

# Visualization
plot_risk_map(advice['risk_map'])

print("--- Agent System Execution Complete ---")


# --- 5. Agent Deployment Simulation ---


def DeploymentOrchestrator(df):
    """
    Manages the sequential execution of all agents, handling instantiation,
    sequencing, and final output display.
    """
    print("--- 5. Agent Deployment: Orchestrator Initializing Services ---")
    print(" Â  [Deployment] Authentication and Resource Allocation successful.")
    
    # 1. Initialize and Run DataCollector
    collector = DataCollector(df)
    processed_df = collector.run_collection()
    
    # 2. Initialize and Run BehaviorPredictor (sends payload via A2A)
    predictor = BehaviorPredictor(processed_df)
    analysis_payload = predictor.run_prediction() 
    
    # 3. Run LLM Adviser (Receives A2A Payload)
    advice = LLM_Advisor_Reasoning(analysis_payload, df)
    
    print("\n--- Agent Trace and Metrics (Observability) ---")
    print(f"Prometheus Metrics Snapshot: {METRICS}")
    print("---------------------------------------------")
    
    print("\n" + "="*80)
    print(" Â  Â  Â  Â  Â  âš¡ï¸� TrendAlertsâ„¢ & Future CrashPoint Detector Warning âš¡ï¸�")
    print("="*80)
    print(f"**WARNING:** {advice['warning']}")
    print(f"\n**Micro-Plan:** {advice['plan']}")
    print("="*80)
    
    # 4. Visualization
    print("\n--- Generating Visualization: Personal Risk Map ---")
    plot_risk_map(advice['risk_map'])
    
    print("--- Deployment: Agent System Shutdown ---")


# --- RUN BLOCK 5 (Main Execution) ---
# NOTE: The helper functions (generate_synthetic_data, LLM_Advisor_Reasoning, DataCollector, BehaviorPredictor)
# must be defined in the preceding cells for this block to run successfully.

# Global DF for mood lookups in LLM_Advisor_Reasoning (simulated shared state)
df = generate_synthetic_data(days=14) 
DeploymentOrchestrator(df)

