import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP: Synthetic Data Generator
# ==========================================
def generate_supply_chain_data(days=365):
    """
    Generates synthetic daily sales data with seasonality and injected anomalies.
    """
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days)
    
    # Create base trend + seasonality
    trend = np.linspace(100, 150, days)
    seasonality = 10 * np.sin(np.linspace(0, 3.14 * 8, days))
    noise = np.random.normal(0, 5, days)
    
    values = trend + seasonality + noise
    
    # Inject Anomalies (e.g., Viral trend or Supply Shock)
    # Spike at index 300
    values[300] = values[300] * 2.5 
    # Drop at index 350
    values[350] = values[350] * 0.2
    
    df = pd.DataFrame({'ds': dates, 'y': values})
    return df

# ==========================================
# 2. THE TOOLKIT (The "Hands" of the Agent)
# ==========================================
class SupplyChainTools:
    
    @staticmethod
    def detect_anomalies(data):
        """
        Tool: Uses Isolation Forest to identify outliers in the data.
        """
        print(f"   [Tool Log] Running IsolationForest on {len(data)} data points...")
        df = data.copy()
        # Simple feature engineering: using value 'y'
        model = IsolationForest(contamination=0.03, random_state=42)
        df['anomaly'] = model.fit_predict(df[['y']])
        
        # -1 indicates anomaly, 1 indicates normal
        anomalies = df[df['anomaly'] == -1]
        
        # Check if the very last data point is an anomaly
        latest_is_anomaly = df.iloc[-1]['anomaly'] == -1
        
        return {
            "total_anomalies": len(anomalies),
            "latest_is_anomaly": latest_is_anomaly,
            "anomaly_dates": anomalies['ds'].dt.date.tolist(),
            "df_result": df # Return df for plotting later
        }

    @staticmethod
    def forecast_demand(data, horizon=7):
        """
        Tool: Uses Exponential Smoothing (Holt-Winters) to predict future demand.
        """
        print(f"   [Tool Log] Training Forecasting Model (Holt-Winters)...")
        series = data['y']
        
        # Fit model
        model = ExponentialSmoothing(series, seasonal_periods=30, trend='add', seasonal='add', use_boxcox=True).fit()
        forecast = model.forecast(horizon)
        
        avg_forecast = forecast.mean()
        
        return {
            "forecast_values": forecast,
            "average_predicted_demand": round(avg_forecast, 2),
            "horizon": horizon
        }

    @staticmethod
    def inventory_strategy(current_stock, predicted_demand):
        """
        Tool: Rule-based logic to determine inventory action.
        """
        print(f"   [Tool Log] Calculating inventory strategy...")
        safety_stock_buffer = 1.2 # Maintain 20% buffer
        
        required_stock = predicted_demand * safety_stock_buffer
        
        if current_stock < required_stock:
            shortage = required_stock - current_stock
            return f"ACTION: REORDER. Projected shortage of {int(shortage)} units."
        elif current_stock > (required_stock * 2):
            return "ACTION: LIQUIDATE. Overstock detected. Recommend discount."
        else:
            return "ACTION: HOLD. Inventory levels are healthy."

# ==========================================
# 3. THE AGENT (The "Brain")
# ==========================================
class AgentOrchestrator:
    def __init__(self):
        self.memory = []
        self.tools = SupplyChainTools()
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.memory.append(entry)
        print(entry)

    def run_analysis_cycle(self, data, current_stock_level):
        self.log("AGENT STARTED: Initiating Supply Chain Review Cycle.")
        
        # --- Step 1: Anomaly Detection ---
        self.log("STEP 1: Checking for data anomalies...")
        anomaly_report = self.tools.detect_anomalies(data)
        
        if anomaly_report['latest_is_anomaly']:
            self.log("!!! ALERT: Anomaly detected in most recent data point.")
            context = "URGENT: Market volatility detected."
        else:
            self.log("Status: Data looks normal. Proceeding with standard forecast.")
            context = "STANDARD: Routine check."

        # --- Step 2: Forecasting ---
        self.log("STEP 2: Forecasting demand for next 7 days...")
        forecast_report = self.tools.forecast_demand(data)
        pred_demand = forecast_report['average_predicted_demand']
        self.log(f"Insight: Predicted average daily demand is {pred_demand} units.")

        # --- Step 3: Decision Making ---
        self.log("STEP 3: Formulating Inventory Strategy...")
        decision = self.tools.inventory_strategy(current_stock_level, pred_demand)
        
        # --- Step 4: Final Output Generation ---
        final_report = {
            "context": context,
            "decision": decision,
            "details": f"Based on a forecast of {pred_demand} and current stock of {current_stock_level}.",
            "visualization_data": anomaly_report['df_result']
        }
        
        self.log(f"CONCLUSION: {decision}")
        self.log("AGENT FINISHED: Cycle Complete.")
        
        return final_report

# ==========================================
# 4. EXECUTION (The "Kaggle" Workflow)
# ==========================================

# A. Generate Data
df = generate_supply_chain_data()

# B. Initialize Agent
agent = AgentOrchestrator()

# C. Run Agent (Scenario: We have 120 units in stock)
# The agent will analyze the history, predict demand, and check if 120 is enough.
results = agent.run_analysis_cycle(df, current_stock_level=120)

# ==========================================
# 5. VISUALIZATION (Proof of Work)
# ==========================================
print("\n--- Generating Dashboard ---")
plt.figure(figsize=(15, 6))

# Plot Actual Sales
plt.plot(df['ds'], df['y'], label='Actual Sales', color='blue', alpha=0.6)

# Highlight Anomalies
anomalies = results['visualization_data'][results['visualization_data']['anomaly'] == -1]
plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Detected Anomalies', s=50, zorder=5)

plt.title(f"SupplyChain-Zero Analysis\nAgent Decision: {results['decision']}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print Agent Memory Log
print("\n--- Agent Memory Dump (Observability) ---")
for log in agent.memory:
    print(log)

