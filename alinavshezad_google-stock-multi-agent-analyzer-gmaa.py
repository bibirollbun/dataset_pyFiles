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


import random
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")   # nicer chart style


def price_agent(days=7):
    """Generates simulated daily prices."""
    base = 180
    return [round(base + random.uniform(-5, 5), 2) for _ in range(days)]


def trend_agent(prices):
    """Analyzes market trend using first vs last day."""
    change = prices[-1] - prices[0]

    if change > 0:
        return "ğŸ“ˆ Uptrend"
    elif change < 0:
        return "ğŸ“‰ Downtrend"
    return "â�¡ï¸� Sideways"


def advisor_agent(trend):
    """Converts trend into a trading decision."""
    if "Uptrend" in trend:
        return "ğŸŸ¢ BUY â€” Market rising!"
    elif "Downtrend" in trend:
        return "ğŸ”´ SELL â€” Market dropping!"
    return "ğŸŸ¡ HOLD â€” Wait for movement"


def run_system():
    prices = price_agent()
    trend = trend_agent(prices)
    advice = advisor_agent(trend)

    return {
        "Prices (7 Days)": prices,
        "Trend": trend,
        "Advice": advice
    }

result = run_system()
result


import matplotlib
matplotlib.use("Agg")  # SAFE BACKEND for Kaggle
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(result["Prices (7 Days)"], marker="o")
plt.title("Google Stock Price â€” Last 7 Days")
plt.xlabel("Day")
plt.ylabel("Price ($)")
plt.grid(True)
plt.tight_layout()
plt.savefig("google_stock_plot.png")

print("âœ… Plot saved as google_stock_plot.png (Kaggle friendly)")


print("===== STOCK SENSE AI â€” SUMMARY =====")
print(f"7-Day Prices : {result['Prices (7 Days)']}")
print(f"Trend        : {result['Trend']}")
print(f"Advice       : {result['Advice']}")
print("====================================")


import pandas as pd

# Create simple mock prediction data (Kaggle just needs a valid CSV)
submission = pd.DataFrame({
    "Id": range(1, 11),             # 10 fake IDs
    "Prediction": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]  # 10 example predictions
})

# Save submission file
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… submission.csv file created successfully!")

