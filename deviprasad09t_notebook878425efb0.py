import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Load your Walmart sales data
file_path = '/kaggle/input/walmart-sales/Walmart_Sales.xlsx'
df_raw = pd.read_excel(file_path)

# Preview the first few rows to verify
print(df_raw.head())



# Cleaning tool customized for Walmart dataset
def tool_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Store', 'Weekly_Sales'])
    df['Weekly_Sales'] = df['Weekly_Sales'].clip(lower=0)
    return df

# KPI calculation adapted to Walmart sales data
def tool_compute_overall_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    total_sales = df['Weekly_Sales'].sum()
    avg_sales = df['Weekly_Sales'].mean()
    total_stores = df['Store'].nunique()
    sales_last_7_weeks = df.groupby('Date')['Weekly_Sales'].sum().tail(7).to_dict()
    return {
        'total_sales': float(round(total_sales, 2)),
        'avg_weekly_sales': float(round(avg_sales, 2)),
        'total_stores': total_stores,
        'last_7_weeks_sales': sales_last_7_weeks
    }



def tool_top_stores(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).head(n).reset_index()

def tool_plot_sales_trend(df: pd.DataFrame):
    weekly_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=weekly_sales, x='Date', y='Weekly_Sales', marker='o')
    plt.title('Weekly Sales Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



@dataclass
class AgentMemory:
    kpi_history: List[Dict[str, Any]] = field(default_factory=list)
    conversations: List[Dict[str, str]] = field(default_factory=list)

    def add_kpis(self, kpis: Dict[str, Any]):
        self.kpi_history.append(kpis)

    def add_message(self, role: str, content: str):
        self.conversations.append({'role': role, 'content': content})

@dataclass
class EnterpriseAnalyticsAgent:
    df_raw: pd.DataFrame
    memory: AgentMemory = field(default_factory=AgentMemory)
    df_clean: Optional[pd.DataFrame] = None

    def initialize(self):
        self.df_clean = tool_clean_data(self.df_raw)
        kpis = tool_compute_overall_kpis(self.df_clean)
        self.memory.add_kpis(kpis)
        return kpis

    def route_question(self, question: str) -> str:
        question = question.lower()
        if 'summary' in question or 'kpi' in question:
            return 'overall_kpis'
        if 'store' in question:
            return 'top_stores'
        if 'trend' in question or 'time' in question or 'weekly' in question:
            return 'sales_trend'
        return 'overall_kpis'

    def answer(self, question: str) -> str:
        self.memory.add_message('user', question)
        if self.df_clean is None:
            self.initialize()

        action = self.route_question(question)

        if action == 'overall_kpis':
            kpis = tool_compute_overall_kpis(self.df_clean)
            self.memory.add_kpis(kpis)
            answer = (
                f"Total sales: {kpis['total_sales']:.2f}, "
                f"Average weekly sales: {kpis['avg_weekly_sales']:.2f}, "
                f"Number of stores: {kpis['total_stores']}."
            )
        elif action == 'top_stores':
            stores = tool_top_stores(self.df_clean)
            answer = 'Top stores by weekly sales:\n' + stores.to_string(index=False)
        elif action == 'sales_trend':
            answer = 'Here is the weekly sales trend plot.'
            tool_plot_sales_trend(self.df_clean)
        else:
            kpis = tool_compute_overall_kpis(self.df_clean)
            answer = "Here's the summary:\n" + str(kpis)

        self.memory.add_message('assistant', answer)
        return answer



# Create and use the agent
agent = EnterpriseAnalyticsAgent(df_raw=df_raw)
print("Initializing Agent and computing base KPIs...")
base_kpis = agent.initialize()
print("Base KPIs:", base_kpis)

# Example Questions:
print("\nUser: Provide a summary of sales performance.")
print("Agent:", agent.answer("Provide a summary of sales performance."))

print("\nUser: Which stores have the highest sales?")
print("Agent:", agent.answer("Which stores have the highest sales?"))

print("\nUser: Show me the sales trend over the last months.")
print("Agent:", agent.answer("Show me the sales trend over the last months."))


