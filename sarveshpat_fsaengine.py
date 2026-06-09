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


import pandas as pd
from dataclasses import dataclass
from typing import Optional

# Data containers

@dataclass
class CompanyData:
    name: str
    df: pd.DataFrame

@dataclass
class RatioResults:
    table: pd.DataFrame

@dataclass
class ForecastResults:
    table: pd.DataFrame

@dataclass
class AnalysisReport:
    narrative: str
    ratios: RatioResults
    forecasts: ForecastResults
    valuation: dict
    beneish: dict
    altman: dict



class BaseAgent:
    """
    Simple base class for all agents with a name and a log method.
    """

    def __init__(self, name: str):
        self.name = name

    def log(self, msg: str) -> None:
        print(f"[{self.name}] {msg}")


class CompanyLookupAgent(BaseAgent):
    """
    Maps company names in user text to local CSV file paths.
    This keeps the system offline and deterministic.
    """

    def __init__(self, mapping: dict):
        super().__init__("CompanyLookupAgent")
        # Normalize keys to lowercase
        self.mapping = {k.lower(): v for k, v in mapping.items()}

    def get_file_for_company(self, user_text: str) -> str:
        self.log(f"User input: {user_text}")
        words = user_text.lower().split()
        for w in words:
            if w in self.mapping:
                self.log(f"Matched company: {w}")
                return self.mapping[w]
        raise ValueError("No matching company found in the mapping.")



class DataIntakeAgent(BaseAgent):
    """
    Loads CSV data for a given company and returns a structured CompanyData object.
    """

    def __init__(self):
        super().__init__("DataIntakeAgent")

    def load_from_csv(self, name: str, path: str) -> CompanyData:
        self.log(f"Loading {name} from {path}")
        df = pd.read_csv(path)
        df = df.sort_values("Year").reset_index(drop=True)
        return CompanyData(name=name, df=df)


class RatioAnalysisAgent(BaseAgent):
    """
    Computes key financial ratios:
    - Profitability (margins)
    - Liquidity (current ratio)
    - Leverage (debt-to-equity)
    - Efficiency (asset turnover)
    - Simple valuation (EPS and P/E)
    """

    def __init__(self):
        super().__init__("RatioAnalysisAgent")

    def compute_ratios(self, data: CompanyData) -> RatioResults:
        self.log(f"Computing ratios for {data.name}")
        df = data.df.copy()

        df["GrossMargin"] = (df["Revenue"] - df["COGS"]) / df["Revenue"]
        df["OperatingMargin"] = df["OperatingIncome"] / df["Revenue"]
        df["NetMargin"] = df["NetIncome"] / df["Revenue"]

        df["CurrentRatio"] = df["CurrentAssets"] / df["CurrentLiabilities"]
        df["DebtToEquity"] = df["TotalDebt"] / df["TotalEquity"]
        df["AssetTurnover"] = df["Revenue"] / df["TotalAssets"]

        df["EarningsPerShare"] = df["NetIncome"] / df["SharesOutstanding"]
        df["PE"] = df["Price"] / df["EarningsPerShare"]

        cols = [
            "Year",
            "GrossMargin",
            "OperatingMargin",
            "NetMargin",
            "CurrentRatio",
            "DebtToEquity",
            "AssetTurnover",
            "EarningsPerShare",
            "PE"
        ]
        ratio_table = df[cols].copy()
        return RatioResults(table=ratio_table)



class ForecastingAgent(BaseAgent):
    """
    Generates simple revenue forecasts using historical CAGR.
    """

    def __init__(self, horizon: int = 3):
        super().__init__("ForecastingAgent")
        self.horizon = horizon

    def _cagr(self, series: pd.Series) -> Optional[float]:
        if len(series) < 2:
            return None
        start = series.iloc[0]
        end = series.iloc[-1]
        if start <= 0:
            return None
        periods = len(series) - 1
        return (end / start) ** (1 / periods) - 1

    def forecast_revenue(self, data: CompanyData) -> ForecastResults:
        self.log(f"Forecasting revenue for {data.name}")
        rev = data.df["Revenue"]
        cagr = self._cagr(rev)
        if cagr is None:
            cagr = 0.0

        last_year = int(data.df["Year"].iloc[-1])
        last_rev = float(rev.iloc[-1])

        years = []
        values = []
        current = last_rev

        for i in range(1, self.horizon + 1):
            current *= (1 + cagr)
            years.append(last_year + i)
            values.append(current)

        fdf = pd.DataFrame({
            "Year": years,
            "ForecastRevenue": values,
            "AssumedCAGR": [cagr] * len(years)
        })
        return ForecastResults(table=fdf)



class DCFValuationAgent(BaseAgent):
    """
    Light DCF valuation:
    - Uses last net margin
    - Applies it to 3-year revenue forecasts
    - Uses 10% discount rate and 2% terminal growth
    This is a simplified model intended for demonstration.
    """

    def __init__(self):
        super().__init__("DCFValuationAgent")
        self.discount_rate = 0.10
        self.terminal_growth = 0.02

    def compute_dcf(self, data: CompanyData, forecasts: ForecastResults) -> dict:
        self.log(f"Running DCF for {data.name}")
        latest = data.df.iloc[-1]
        net_margin = latest["NetIncome"] / latest["Revenue"]

        f = forecasts.table
        if len(f) < 3:
            return {
                "DCFValue": None,
                "TerminalValue": None,
                "NetMarginUsed": net_margin
            }

        f1 = f["ForecastRevenue"].iloc[0]
        f2 = f["ForecastRevenue"].iloc[1]
        f3 = f["ForecastRevenue"].iloc[2]

        fcff1 = f1 * net_margin
        fcff2 = f2 * net_margin
        fcff3 = f3 * net_margin

        terminal_fcff = fcff3 * (1 + self.terminal_growth)
        terminal_value = terminal_fcff / (self.discount_rate - self.terminal_growth)

        d1 = 1 / (1 + self.discount_rate)
        d2 = 1 / ((1 + self.discount_rate) ** 2)
        d3 = 1 / ((1 + self.discount_rate) ** 3)

        present_value = (
            fcff1 * d1 +
            fcff2 * d2 +
            fcff3 * d3 +
            terminal_value * d3
        )

        return {
            "DCFValue": present_value,
            "TerminalValue": terminal_value,
            "NetMarginUsed": net_margin
        }



class BeneishAgent(BaseAgent):
    """
    Lean Beneish M-Score based on a simplified 4-ratio model:
    DSRI, GMI, AQI, SGI.
    Uses approximations due to limited data fields.
    """

    def __init__(self):
        super().__init__("BeneishAgent")

    def compute_beneish(self, data: CompanyData) -> dict:
        df = data.df.copy()
        if len(df) < 2:
            return {"MScore": None, "Interpretation": "Not enough data"}

        t1 = df.iloc[-2]
        t2 = df.iloc[-1]

# Receivables are not available in the dataset, so a consistent proxy is used:
# Receivables ≈ 30% of current assets. This assumption keeps the Beneish model operational.

        rec1 = t1["CurrentAssets"] * 0.3
        rec2 = t2["CurrentAssets"] * 0.3
        dsri = (rec2 / t2["Revenue"]) / (rec1 / t1["Revenue"])

        gm1 = (t1["Revenue"] - t1["COGS"]) / t1["Revenue"]
        gm2 = (t2["Revenue"] - t2["COGS"]) / t2["Revenue"]
        gmi = gm1 / gm2

        aqi = (1 - (t2["CurrentAssets"] / t2["TotalAssets"])) / (1 - (t1["CurrentAssets"] / t1["TotalAssets"]))
        sgi = t2["Revenue"] / t1["Revenue"]

        m_score = -6.0 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
        interpretation = "Possible manipulation risk" if m_score > -2.22 else "No indication of manipulation"

        return {"MScore": m_score, "Interpretation": interpretation}


class AltmanZAgent(BaseAgent):
    """
    Lean Altman Z-Score model using available fields.
    Uses:
    - Working capital
    - Retained earnings (approx = equity - net income)
    - EBIT
    - Market value of equity
    - Book liabilities
    """

    def __init__(self):
        super().__init__("AltmanZAgent")

    def compute_z(self, data: CompanyData) -> dict:
        df = data.df
        t = df.iloc[-1]

        wc = t["CurrentAssets"] - t["CurrentLiabilities"]
        retained = t["TotalEquity"] - t["NetIncome"]
        ebit = t["OperatingIncome"]
        mve = t["SharesOutstanding"] * t["Price"]
        liabilities = t["TotalAssets"] - t["TotalEquity"]
        if liabilities <= 0:
            liabilities = 1e-6

        z = (
            1.2 * (wc / t["TotalAssets"]) +
            1.4 * (retained / t["TotalAssets"]) +
            3.3 * (ebit / t["TotalAssets"]) +
            0.6 * (mve / liabilities) +
            1.0 * (t["Revenue"] / t["TotalAssets"])
        )

        if z > 2.99:
            risk = "Safe Zone"
        elif z > 1.81:
            risk = "Grey Zone"
        else:
            risk = "Distress Zone"

        return {"ZScore": z, "RiskZone": risk}



class NarrativeSummaryAgent(BaseAgent):
    """
    Generates a plain-language narrative combining:
    - Ratios
    - Forecasts
    - DCF valuation
    - Beneish M-Score
    - Altman Z-Score
    """

    def __init__(self):
        super().__init__("NarrativeSummaryAgent")

    def generate_summary(
        self,
        data: CompanyData,
        ratios: RatioResults,
        forecasts: ForecastResults,
        valuation: dict,
        beneish: dict,
        altman: dict
    ) -> AnalysisReport:
        self.log(f"Generating narrative for {data.name}")
        r = ratios.table
        latest = r.iloc[-1]

        y = int(latest["Year"])
        gm = float(latest["GrossMargin"])
        opm = float(latest["OperatingMargin"])
        nm = float(latest["NetMargin"])
        cr = float(latest["CurrentRatio"])
        de = float(latest["DebtToEquity"])
        at = float(latest["AssetTurnover"])
        pe = float(latest["PE"])

        lines = []
        lines.append(f"Company: {data.name}")
        lines.append("")
        lines.append(f"As of {y}, gross margin is {gm:.1%}, operating margin is {opm:.1%}, and net margin is {nm:.1%}.")
        lines.append(f"The current ratio is about {cr:.2f}, with a debt-to-equity ratio of {de:.2f} and asset turnover of {at:.2f}.")
        lines.append(f"The price-to-earnings ratio is approximately {pe:.1f}x.")

        f = forecasts.table
        if len(f) > 0:
            fy = int(f["Year"].iloc[-1])
            fv = float(f["ForecastRevenue"].iloc[-1])
            cg = float(f["AssumedCAGR"].iloc[0])
            lines.append("")
            lines.append(
                f"Based on historical growth, revenue is projected to increase at roughly {cg:.1%} per year, "
                f"reaching about {fv:,.0f} by {fy}."
            )

        lines.append("")
        if valuation["DCFValue"] is not None:
            lines.append("DCF valuation (simplified):")
            lines.append(f"Estimated intrinsic value of future cash flows: {valuation['DCFValue']:,.0f}")
            lines.append(f"Terminal value used in the model: {valuation['TerminalValue']:,.0f}")
        else:
            lines.append("DCF valuation could not be computed due to insufficient forecast data.")

        lines.append("")
        if beneish["MScore"] is not None:
            lines.append("Fraud risk (Beneish M-Score, simplified):")
            lines.append(f"M-Score: {beneish['MScore']:.2f}")
            lines.append(f"Interpretation: {beneish['Interpretation']}")
        else:
            lines.append("Beneish M-Score not available (not enough historical data).")

        lines.append("")
        lines.append("Bankruptcy risk (Altman Z-Score, simplified):")
        lines.append(f"Z-Score: {altman['ZScore']:.2f}")
        lines.append(f"Risk Zone: {altman['RiskZone']}")

        lines.append("")
        lines.append(
            "Overall, this summary combines profitability, liquidity, leverage, efficiency, valuation, "
            "fraud indicators, and bankruptcy risk into a single view."
        )

        narrative = "\n".join(lines)

        return AnalysisReport(
            narrative=narrative,
            ratios=ratios,
            forecasts=forecasts,
            valuation=valuation,
            beneish=beneish,
            altman=altman
        )



class DirectorAgent(BaseAgent):
    """
    Orchestrates the full workflow:
    - Look up company CSV
    - Load data
    - Compute ratios
    - Forecast revenue
    - Run DCF, Beneish, Altman
    - Generate narrative summary
    """

    def __init__(self):
        super().__init__("DirectorAgent")

        self.lookup_agent = CompanyLookupAgent({
            "apple": "/kaggle/input/afa-financial-statements-v1/apple.csv",
            "tesla": "/kaggle/input/afa-financial-statements-v1/tesla.csv",
            "nvidia": "/kaggle/input/afa-financial-statements-v1/nvidia.csv"
        })

        self.data_agent = DataIntakeAgent()
        self.ratio_agent = RatioAnalysisAgent()
        self.forecast_agent = ForecastingAgent()
        self.dcf_agent = DCFValuationAgent()
        self.beneish_agent = BeneishAgent()
        self.altman_agent = AltmanZAgent()
        self.narrative_agent = NarrativeSummaryAgent()

    def run_full_analysis(self, name: str, csv_path: str) -> AnalysisReport:
        self.log(f"Running full analysis for {name}")
        data = self.data_agent.load_from_csv(name, csv_path)
        ratios = self.ratio_agent.compute_ratios(data)
        forecasts = self.forecast_agent.forecast_revenue(data)
        valuation = self.dcf_agent.compute_dcf(data, forecasts)
        beneish = self.beneish_agent.compute_beneish(data)
        altman = self.altman_agent.compute_z(data)
        report = self.narrative_agent.generate_summary(
            data, ratios, forecasts, valuation, beneish, altman
        )
        return report

    def analyze_company_by_name(self, user_text: str) -> AnalysisReport:
        csv_path = self.lookup_agent.get_file_for_company(user_text)
        name = csv_path.split("/")[-1].replace(".csv", "").title()
        return self.run_full_analysis(name, csv_path)



director = DirectorAgent()

# Example: analyze Tesla
report = director.analyze_company_by_name("analyze tesla")

print("=== RATIOS TABLE ===")
print(report.ratios.table)

print("\n=== FORECAST TABLE ===")
print(report.forecasts.table)

print("\n=== VALUATION (DCF) ===")
print(report.valuation)

print("\n=== BENEISH M-SCORE ===")
print(report.beneish)

print("\n=== ALTMAN Z-SCORE ===")
print(report.altman)

print("\n=== NARRATIVE REPORT ===")
print(report.narrative)



director = DirectorAgent()

for company in ["apple", "tesla", "nvidia"]:
    print("=" * 60)
    print(f"ANALYSIS FOR: {company.upper()}")
    print("=" * 60)
    req = f"analyze {company}"
    rep = director.analyze_company_by_name(req)
    print(rep.narrative)
    print("\n")



# Safety check before generating output file
try:
    _ = DirectorAgent
except NameError:
    raise RuntimeError("DirectorAgent is not defined. Please run the entire notebook from the top (Run All) before running this cell.")



director = DirectorAgent()
sample_report = director.analyze_company_by_name("nvidia")

output_text = sample_report.narrative

output_path = "/kaggle/working/afa_output.txt"

with open(output_path, "w") as f:
    f.write("Autonomous Financial Analyst (AFA Agent) Output\n")
    f.write("----------------------------------------------\n\n")
    f.write(output_text)

print("Output file saved to:", output_path)


