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


!pip install -q -U google-generativeai langchain-google-genai langchain langchain-community langchain-experimental ipywidgets
print("Libraries installed successfully!")


import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from kaggle_secrets import UserSecretsClient

# Setup Gemini API Key
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

# Matplotlib settings for the notebook
%matplotlib inline
sns.set_style("whitegrid")
print("Environment setup complete.")


# Load Data
# Note: Kaggle input data is read-only. We process it here.
print("Loading Rossmann Data...")
dtype_dict = {'StateHoliday': 'str', 'SchoolHoliday': 'str'}
df = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv', dtype=dtype_dict, parse_dates=['Date'], low_memory=False)
store_df = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

# Merge store info so we have metadata (competitors, etc.)
df = pd.merge(df, store_df, on='Store', how='left')

# Filter for Store 1 (For the demo speed)
store_id = 1
data = df[df['Store'] == store_id].sort_values('Date')

# Save a recent slice to the working directory for the Agent to "see" later
# /kaggle/working/ is where you are allowed to save files
data.tail(90).to_csv('/kaggle/working/recent_data.csv', index=False)

print(f"Data loaded. Processing Store {store_id} with {len(data)} records.")
data.head(3)


# 1. Prepare Data for LSTM
print("Training Model... (This takes about 30 seconds)")
sales_data = data[['Sales']].values.astype(float)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(sales_data)

def create_dataset(dataset, time_step=60):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 2. Build Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 3. Train
model.fit(X, y, batch_size=64, epochs=3, verbose=0) # verbose=0 to keep notebook clean

# 4. Save Model to Output Directory
model.save('/kaggle/working/sales_model.keras')
with open('/kaggle/working/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("âœ… Model trained and saved to /kaggle/working/")


from langchain.tools import tool
import tensorflow as tf

# Load artifacts
loaded_model = tf.keras.models.load_model('/kaggle/working/sales_model.keras')
with open('/kaggle/working/scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)
    
df_recent = pd.read_csv('/kaggle/working/recent_data.csv')

@tool
def get_recent_sales_data(days: int):
    """
    Retrieves the actual sales data (Sales, Customers, Open status) for the last N days.
    """
    return df_recent.tail(days)[['Date', 'Sales', 'Customers', 'Open', 'Promo']].to_string()

@tool
def get_store_metadata():
    """
    Retrieves static information about the store: Competitor distance and Assortment type.
    """
    # Just taking the first row since metadata is constant for the store
    meta = df_recent.iloc[0][['StoreType', 'Assortment', 'CompetitionDistance']]
    return meta.to_string()

@tool
def forecast_next_day_sales():
    """
    Uses the LSTM Machine Learning model to predict sales for the next upcoming day.
    Returns a dollar amount.
    """
    # Get last 60 days
    last_60 = df_recent['Sales'].values[-60:].astype(float)
    last_60_scaled = loaded_scaler.transform(last_60.reshape(-1, 1))
    
    X_test = np.array([last_60_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    pred_scaled = loaded_model.predict(X_test, verbose=0)
    pred_inverse = loaded_scaler.inverse_transform(pred_scaled)
    
    return f"Forecasted Sales: ${float(pred_inverse[0][0]):.2f}"

print("Tools initialized.")


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [get_recent_sales_data, get_store_metadata, forecast_next_day_sales]

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a Retail Strategy AI. You help managers understand their data. "
     "You have access to a Forecast Model and a Historical Database. "
     "If asked about the future, USE the 'forecast_next_day_sales' tool. "
     "If asked 'Why', check historical data for holidays or promos. "
     "Be concise."
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print("ðŸ¤– Agent is ready!")


import ipywidgets as widgets
from IPython.display import display, clear_output

# 1. Create Layout
header = widgets.HTML("<h2>ðŸ›’ Store Manager AI Dashboard</h2>")
txt_input = widgets.Text(placeholder="Ask about sales (e.g., 'Forecast sales for tomorrow')", layout=widgets.Layout(width='70%'))
btn_send = widgets.Button(description="Ask Agent", button_style='primary')
output_area = widgets.Output()

# 2. Define Action
def on_button_click(b):
    user_query = txt_input.value
    with output_area:
        clear_output()
        print(f"User: {user_query}")
        print("Agent is thinking...")
        
        try:
            # Invoke Agent
            response = agent_executor.invoke({"input": user_query})
            print("\n" + "="*40)
            print(f"ðŸ¤– AI ANSWER:\n{response['output']}")
            print("="*40)
            
            # Optional: If forecast is mentioned, show a chart
            if "forecast" in user_query.lower() or "trend" in user_query.lower():
                plt.figure(figsize=(10, 4))
                plt.plot(pd.to_datetime(df_recent['Date']), df_recent['Sales'], label='History')
                plt.title("Sales Trend Context")
                plt.legend()
                plt.show()
                
        except Exception as e:
            print(f"Error: {e}")

btn_send.on_click(on_button_click)

# 3. Display
display(header, widgets.HBox([txt_input, btn_send]), output_area)

