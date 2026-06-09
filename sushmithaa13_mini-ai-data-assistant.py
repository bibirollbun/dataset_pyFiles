import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("âœ… Gemini configured successfully.")


data = {
    "Employee": ["Asha", "Ravi", "John", "Priya", "Sam"],
    "Sales": [12000, 18000, 9000, 22000, 15000],
    "Region": ["South", "North", "East", "West", "South"],
    "Month": ["Jan", "Jan", "Jan", "Jan", "Jan"]
}

df = pd.DataFrame(data)
df.to_csv("sample_sales.csv", index=False)

print("âœ… Dummy CSV created: sample_sales.csv")
df


# SimpleAgent to use gemini-pro
class SimpleAgent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

    def run(self, user_input):
        import google.generativeai as genai
        
        # Use a model compatible with Kaggle
        model = genai.GenerativeModel("gemini-pro")

        prompt = f"""
        You are {self.name}.
        {self.system_prompt}

        User request:
        {user_input}
        """

        response = model.generate_content(prompt)
        return response.text

print("ğŸ”„ Agent updated to use gemini-pro model.")


class CSVAgent(SimpleAgent):
    def load_csv(self, file_path):
        df = pd.read_csv(file_path)
        return df

    def clean_data(self, df):
        # Simple cleaning: drop duplicates, fill missing values if needed
        df = df.drop_duplicates()
        df = df.fillna(0)
        return df

    def summarize(self, df):
        return df.describe(include="all")

# Instantiate agent
csv_agent = CSVAgent(
    name="CSV Loader & Cleaner Agent",
    system_prompt="You load CSV files, clean them, and provide dataset summaries."
)

print("âœ… CSV Agent created.")


# Load the dummy CSV
df = csv_agent.load_csv("sample_sales.csv")
print("ğŸ“„ Loaded CSV:")
display(df)

# Clean
clean_df = csv_agent.clean_data(df)
print("\nğŸ§¹ Cleaned Data:")
display(clean_df)

# Summary
summary = csv_agent.summarize(clean_df)
print("\nğŸ“Š Summary Statistics:")
display(summary)


class InsightsAgent:
    def __init__(self, name="Insights Agent"):
        self.name = name

    def generate_insights(self, df):
        insights = []

        # Top performer
        top_employee = df.loc[df["Sales"].idxmax()]["Employee"]
        insights.append(f"ğŸ�† Top performer: {top_employee}")

        # Lowest performer
        low_employee = df.loc[df["Sales"].idxmin()]["Employee"]
        insights.append(f"ğŸ“‰ Lowest performer: {low_employee}")

        # Average sales
        avg_sales = df["Sales"].mean()
        insights.append(f"ğŸ’° Average sales: {avg_sales}")

        # Regional performance
        region_sales = df.groupby("Region")["Sales"].sum().to_dict()
        insights.append(f"ğŸŒ� Regional sales: {region_sales}")

        # Any anomalies
        high_sales_threshold = avg_sales * 1.5
        high_sales = df[df["Sales"] > high_sales_threshold]["Employee"].tolist()
        insights.append(f"âš ï¸� Employees exceeding 1.5x average sales: {high_sales}")

        return "\n".join(insights)

# Instantiate Insights Agent
insights_agent = InsightsAgent()
print("âœ… Insights Agent ready (Kaggle-safe).")


insights = insights_agent.generate_insights(clean_df)
print(insights)


class DashboardAgent:
    def __init__(self, name="Dashboard Agent"):
        self.name = name

    def create_dashboard(self, df):
        print("ğŸ“Š Dashboard Summary:\n")
        print(df.describe(include="all"))

        # Sales by Employee
        plt.figure(figsize=(8,5))
        sns.barplot(x="Employee", y="Sales", data=df, palette="viridis")
        plt.title("Sales by Employee")
        plt.show()

        # Sales by Region
        plt.figure(figsize=(8,5))
        region_sales = df.groupby("Region")["Sales"].sum().reset_index()
        sns.barplot(x="Region", y="Sales", data=region_sales, palette="magma")
        plt.title("Total Sales by Region")
        plt.show()

# Instantiate
dashboard_agent = DashboardAgent()
print("âœ… Dashboard Agent ready.")


# 1ï¸�âƒ£ Load & Clean CSV
df = csv_agent.load_csv("sample_sales.csv")
clean_df = csv_agent.clean_data(df)

print("ğŸ“„ Loaded & Cleaned Data:")
display(clean_df)

# 2ï¸�âƒ£ Generate Insights
insights = insights_agent.generate_insights(clean_df)
print("\nğŸ“˜ Insights:\n")
print(insights)

# 3ï¸�âƒ£ Show Dashboard
dashboard_agent.create_dashboard(clean_df)


class LocalQnAAgent:
    def __init__(self, df):
        self.df = df

    def answer(self, question):
        q = question.lower()

        # 1. Top seller
        if "top seller" in q or "highest" in q:
            row = self.df.loc[self.df["Sales"].idxmax()]
            return f"ğŸ�† Top seller: {row['Employee']} with {row['Sales']} sales."

        # 2. Lowest performer
        if "lowest" in q:
            row = self.df.loc[self.df["Sales"].idxmin()]
            return f"ğŸ“‰ Lowest performer: {row['Employee']} with {row['Sales']} sales."

        # 3. Average sales
        if "average" in q:
            avg = self.df["Sales"].mean()
            return f"ğŸ“Š Average sales: {avg:.2f}"

        # 4. Region sales
        if "region" in q:
            region_sales = self.df.groupby("Region")["Sales"].sum().to_dict()
            return f"ğŸŒ� Regional sales: {region_sales}"

        # 5. Employee-specific question
        for name in self.df["Employee"].str.lower():
            if name in q:
                row = self.df[self.df["Employee"].str.lower() == name].iloc[0]
                return (
                    f"{row['Employee']} sold {row['Sales']} units "
                    f"in the {row['Region']} region ({row['Month']})."
                )

        return "â�“ I don't know this yet. Try asking about top seller, average sales, region performance, or employee details."


class LocalQnAAgentWithCharts(LocalQnAAgent):
    def answer(self, question):
        q = question.lower()

        # Chart requests
        if "plot" in q or "chart" in q or "graph" in q:
            self.df.plot(x="Employee", y="Sales", kind="bar")
            plt.title("Sales by Employee")
            plt.xlabel("Employee")
            plt.ylabel("Sales")
            plt.tight_layout()
            plt.show()
            return "ğŸ“Š Chart generated."

        # Otherwise use the parent class logic
        return super().answer(question)


print("âœ… Local QnA Agent Ready (No API calls used!)")


qa = LocalQnAAgent(df)
qa.answer("Who is the top seller?")


qa = LocalQnAAgentWithCharts(df)
qa.answer("plot sales")
qa.answer("generate a chart of employee performance")
qa.answer("show sales graph")

