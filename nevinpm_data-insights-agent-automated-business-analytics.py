import os
import pandas as pd
import numpy as np



data_path ="/kaggle/input/supermarket-sales-data/annex2.csv"  


df = pd.read_csv(data_path)
df.head()




df["TotalSales"] = df["Quantity Sold (kilo)"] * df["Unit Selling Price (RMB/kg)"]

df.head()




df["Date"] = pd.to_datetime(df["Date"])

df[["Date"]].head()



DATE_COL = "Date"         
VALUE_COL = "TotalSales"  
GROUP_COL = "Item Code"   



DATE_COL = "Date"        
VALUE_COL = "TotalSales"  
GROUP_COL = "Item Code"   



def compute_kpis(df):
    return {
        "row_count": int(len(df)),
        "total_sales": float(df[VALUE_COL].sum()),
        "avg_sale": float(df[VALUE_COL].mean()),
        "max_sale": float(df[VALUE_COL].max()),
        "min_sale": float(df[VALUE_COL].min()),
        "unique_products": int(df[GROUP_COL].nunique()),
    }

def daily_sales(df):
    daily = (
        df.groupby(df[DATE_COL].dt.date)[VALUE_COL]
          .sum()
          .rename("daily_sales")
          .reset_index()
    )
    return daily

def top_products(df, n=5):
    tp = (
        df.groupby(GROUP_COL)[VALUE_COL]
          .sum()
          .sort_values(ascending=False)
          .head(n)
          .reset_index()
    )
    return tp

def detect_anomalies(df, z_threshold=3.0):
    daily = daily_sales(df)
    mean = daily["daily_sales"].mean()
    std = daily["daily_sales"].std(ddof=0)
    if std == 0 or np.isnan(std):
        daily["z"] = 0.0
        return daily.iloc[0:0]
    daily["z"] = (daily["daily_sales"] - mean) / std
    return daily[abs(daily["z"]) >= z_threshold]

print(compute_kpis(df))
print(top_products(df))
print(detect_anomalies(df))



print("KPIs:")
print(compute_kpis(df))

print("\nTop Products:")
print(top_products(df))

print("\nAnomalies:")
print(detect_anomalies(df))



from kaggle_secrets import UserSecretsClient

import google.generativeai as gen

def llm_chat(system_prompt: str, user_prompt: str) -> str:
    try:
        api_key = UserSecretsClient().get_secret("GEMINI_API_KEY")
    except Exception:
        return "[Simulated LLM: GEMINI_API_KEY secret not accessible in this environment.]"
    try:
        gen.configure(api_key=api_key)
        model = gen.GenerativeModel("gemini-2.0-flash")
        prompt = system_prompt + "\n\n" + user_prompt
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Simulated LLM: Gemini unreachable → {e}]"



class DataInsightsAgent:
    def __init__(self, df):
        self.df = df

    def summarize_dataset(self):
        kpis = compute_kpis(self.df)
        tp = top_products(self.df, n=5)
        anomalies = detect_anomalies(self.df)

        context = {
            "kpis": kpis,
            "top_products": tp.to_dict(orient="records"),
            "anomalies": anomalies.to_dict(orient="records"),
        }

        system_prompt = (
            "You are a business analytics assistant. "
            "Given KPIs, top products, and anomaly days, "
            "write a clear executive summary for a small business owner. "
            "Use bullet points and simple language."
        )

        user_prompt = f"Analytics:\n{context}"

        return llm_chat(system_prompt, user_prompt)

    def answer_question(self, question: str):
        sample = self.df.head(50).to_dict(orient="records")

        system_prompt = (
            "You are a data analyst. Use the sample of sales data "
            "and business reasoning to answer the question."
        )

        user_prompt = f"Sample data: {sample}\n\nQuestion: {question}"

        return llm_chat(system_prompt, user_prompt)

    def whatsapp_style_summary(self):
        kpis = compute_kpis(self.df)
        tp = top_products(self.df, n=5)

        lines = [
            "Daily Sales Summary:",
            f"- Total sales: {kpis['total_sales']:.2f}",
            f"- Avg sale: {kpis['avg_sale']:.2f}",
            f"- Unique products: {kpis['unique_products']}",
            "",
            "Top products:",
        ]

        for _, row in tp.iterrows():
            lines.append(f"- {row[GROUP_COL]}: {row[VALUE_COL]:.2f}")

        return "\n".join(lines)



agent = DataInsightsAgent(df)

print("=== WhatsApp Style Summary ===")
print(agent.whatsapp_style_summary())



print("=== Executive Summary (LLM) ===")
print(agent.summarize_dataset())

print("\n=== Q&A Example ===")
print(agent.answer_question("Which products performed best and why?"))



print("=== WhatsApp Style Summary ===")
print(agent.whatsapp_style_summary())

print("\n\n=== Executive Summary (LLM) ===")
print(agent.summarize_dataset())

print("\n\n=== Q&A Example ===")
print(agent.answer_question("Which products should the store owner focus on in the next month and why?"))



output_text = agent.summarize_dataset()

with open("/kaggle/working/submission.txt", "w") as f:
    f.write(output_text)

output_text


