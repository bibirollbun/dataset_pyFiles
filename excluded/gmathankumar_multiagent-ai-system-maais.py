

!pip install -q google-generativeai

import google.generativeai as genai

# Put your API key here
GEMINI_API_KEY = "AIzaSyDUD7Yg78d8VRjEW58BGafNMBd68AiP-6w"

# Check if key is empty
if GEMINI_API_KEY.strip() == "":
    raise RuntimeError("â�Œ API key is empty. Add your Gemini API key.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load model
model = genai.GenerativeModel("gemini-1.5-flash")

print("âœ… MultiAgent AI System (MAAIS) ready to start!")    



# ------------------------
# IMPORTS
# ------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
import google.generativeai as genai

# ------------------------
# GEMINI SETUP
# ------------------------
GEMINI_API_KEY = "AIzaSyDUD7Yg78d8VRjEW58BGafNMBd68AiP-6w"  # Replace key
if GEMINI_API_KEY.strip() == "":
    raise RuntimeError("â�Œ Please set your Gemini API key")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
print("âœ… MultiAgent AI System (MAAIS)!")

# ------------------------
# GLOBALS
# ------------------------
df_data = None
chat_history = []

# ------------------------
# HELPERS
# ------------------------
def create_big_button(text):
    return widgets.Button(description=text, layout=widgets.Layout(width='200px', height='50px'))

def run_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"â�Œ Error: {e}"

# ------------------------
# HOME PAGE
# ------------------------
def build_home_page():
    content = """
    <h2>Multiâ€‘Agent AI System (MAAIS)</h2>
    <p>Agents:</p>
    <ul>
      <li>Data Analytics</li>
      <li>Customer Support</li>
      <li>HR Agent</li>
      <li>Finance Agent</li>
    </ul>
    <p><b>How to use:</b></p>
    <ol>
      <li>Go to a page</li>
      <li>Upload a CSV or use sample data</li>
      <li>Click Run / Ask</li>
      <li>Click Clear if needed</li>
    </ol>
    """
    return widgets.HTML(content)

# ------------------------
# DATA ANALYTICS AGENT
# ------------------------
def build_data_agent_page():
    global df_data
    out = widgets.Output()

    upload = widgets.FileUpload(accept='.csv', multiple=False, description="Upload CSV")
    button_sample = create_big_button("Use Sample Data")
    button_run = create_big_button("Run Analysis")
    button_clear = create_big_button("Clear Output")

    sample_df = pd.DataFrame({
        "CustomerID":[1,2,3,4,5],
        "Age":[25,30,45,50,29],
        "Balance":[5000,7500,12000,20000,7000],
        "PurchaseHistory":[5,7,12,15,8]
    })

    def load_sample(b):
        nonlocal sample_df
        global df_data
        df_data = sample_df.copy()
        with out:
            clear_output()
            print("âœ… Loaded sample data")
            display(df_data.head())

    def upload_changed(change):
        global df_data
        if upload.value:
            try:
                content = list(upload.value.values())[0]['content']
                df_data = pd.read_csv(pd.io.common.BytesIO(content))
                with out:
                    clear_output()
                    print("âœ… CSV loaded")
                    display(df_data.head())
            except Exception as e:
                with out:
                    clear_output()
                    print("â�Œ Error:", e)

    def run_click(b):
        global df_data
        with out:
            clear_output()
            if df_data is None:
                print("âš ï¸� No data: Upload or use sample")
                return
            # stats
            num = df_data.select_dtypes(include='number')
            if not num.empty:
                print("ğŸ“Š Summary statistics:")
                display(num.describe())
            # AI insight
            prompt = "Here is some data:\n" + df_data.head().to_csv(index=False)
            res = run_gemini(prompt)
            print("\nğŸ¤– AI Insights:")
            print(res)

    def clear_click(b):
        global df_data
        df_data = None
        upload.value.clear()
        with out:
            clear_output()
            print("âš ï¸� Output cleared")

    button_sample.on_click(load_sample)
    upload.observe(upload_changed, names='value')
    button_run.on_click(run_click)
    button_clear.on_click(clear_click)

    box = widgets.VBox([
        widgets.HTML("<h3>Data Analytics Agent</h3>"),
        widgets.HBox([upload, button_sample]),
        widgets.HBox([button_run, button_clear]),
        out
    ])
    return box

# ------------------------
# CUSTOMER SUPPORT AGENT
# ------------------------
def build_cs_agent_page():
    global chat_history
    out = widgets.Output()
    input_box = widgets.Text(placeholder="Type your question hereâ€¦")
    btn_send = create_big_button("Send")
    btn_clear = create_big_button("Clear Chat")

    chat_history = []

    def ask_ai(q):
        return run_gemini(f"Answer this question professionally: {q}")

    def send(b):
        global chat_history
        q = input_box.value.strip()
        if not q:
            return
        input_box.value = ""
        chat_history.append("User: " + q)
        ans = ask_ai(q)
        chat_history.append("AI: " + ans)
        with out:
            clear_output()
            for m in chat_history:
                print(m)

    def clear_chat(b):
        global chat_history
        chat_history = []
        with out:
            clear_output()
            print("âš ï¸� Chat cleared")

    btn_send.on_click(send)
    btn_clear.on_click(clear_chat)

    box = widgets.VBox([
        widgets.HTML("<h3>Customer Support Agent</h3>"),
        input_box,
        widgets.HBox([btn_send, btn_clear]),
        out
    ])
    return box

# ------------------------
# HR AGENT
# ------------------------
def build_hr_agent_page():
    global df_data
    out = widgets.Output()
    upload = widgets.FileUpload(accept='.csv', multiple=False, description="Upload HR CSV")
    button_sample = create_big_button("Use Sample HR Data")
    button_run = create_big_button("Run HR Analysis")
    button_clear = create_big_button("Clear Output")

    sample_df = pd.DataFrame({
        "EmployeeID":[101,102,103,104,105],
        "Name":["Alice","Bob","Charlie","David","Eva"],
        "Department":["HR","Finance","IT","Marketing","HR"],
        "Age":[25,30,35,28,40],
        "Salary":[50000,60000,55000,65000,58000],
        "PerformanceScore":[8,7,9,6,7]
    })

    def load_sample(b):
        global df_data
        df_data = sample_df.copy()
        with out:
            clear_output()
            print("âœ… Loaded sample HR data")
            display(df_data.head())

    def upload_changed(change):
        global df_data
        if upload.value:
            try:
                content = list(upload.value.values())[0]['content']
                df_data = pd.read_csv(pd.io.common.BytesIO(content))
                with out:
                    clear_output()
                    print("âœ… HR CSV loaded")
                    display(df_data.head())
            except Exception as e:
                with out:
                    clear_output()
                    print("â�Œ Error:", e)

    def run_click(b):
        global df_data
        with out:
            clear_output()
            if df_data is None:
                print("âš ï¸� No data: Upload or use sample")
                return
            # AI insight
            prompt = "Here is HR employee data:\n" + df_data.head().to_csv(index=False)
            res = run_gemini(prompt)
            print("\nğŸ¤– AI Insights:")
            print(res)

    def clear_click(b):
        global df_data
        df_data = None
        upload.value.clear()
        with out:
            clear_output()
            print("âš ï¸� Output cleared")

    button_sample.on_click(load_sample)
    upload.observe(upload_changed, names='value')
    button_run.on_click(run_click)
    button_clear.on_click(clear_click)

    box = widgets.VBox([
        widgets.HTML("<h3>HR Agent</h3>"),
        widgets.HBox([upload, button_sample]),
        widgets.HBox([button_run, button_clear]),
        out
    ])
    return box

# ------------------------
# FINANCE AGENT
# ------------------------
def build_finance_agent_page():
    global df_data
    out = widgets.Output()
    upload = widgets.FileUpload(accept='.csv', multiple=False, description="Upload Finance CSV")
    button_sample = create_big_button("Use Sample Finance Data")
    button_run = create_big_button("Run Finance Analysis")
    button_clear = create_big_button("Clear Output")

    sample_df = pd.DataFrame({
        "Month":["Jan","Feb","Mar","Apr","May"],
        "Revenue":[50000,60000,55000,65000,70000],
        "Expenses":[30000,35000,32000,40000,38000],
        "Profit":[20000,25000,23000,25000,32000]
    })

    def load_sample(b):
        global df_data
        df_data = sample_df.copy()
        with out:
            clear_output()
            print("âœ… Loaded sample Finance data")
            display(df_data.head())

    def upload_changed(change):
        global df_data
        if upload.value:
            try:
                content = list(upload.value.values())[0]['content']
                df_data = pd.read_csv(pd.io.common.BytesIO(content))
                with out:
                    clear_output()
                    print("âœ… Finance CSV loaded")
                    display(df_data.head())
            except Exception as e:
                with out:
                    clear_output()
                    print("â�Œ Error:", e)

    def run_click(b):
        global df_data
        with out:
            clear_output()
            if df_data is None:
                print("âš ï¸� No data: Upload or use sample")
                return
            # AI insight
            prompt = "Here is Finance data:\n" + df_data.head().to_csv(index=False)
            res = run_gemini(prompt)
            print("\nğŸ¤– AI Insights:")
            print(res)

    def clear_click(b):
        global df_data
        df_data = None
        upload.value.clear()
        with out:
            clear_output()
            print("âš ï¸� Output cleared")

    button_sample.on_click(load_sample)
    upload.observe(upload_changed, names='value')
    button_run.on_click(run_click)
    button_clear.on_click(clear_click)

    box = widgets.VBox([
        widgets.HTML("<h3>Finance Agent</h3>"),
        widgets.HBox([upload, button_sample]),
        widgets.HBox([button_run, button_clear]),
        out
    ])
    return box

# ------------------------
# MAIN UI
# ------------------------
pages = {
    "Home": build_home_page(),
    "Data Analytics": build_data_agent_page(),
    "Customer Support": build_cs_agent_page(),
    "HR": build_hr_agent_page(),
    "Finance": build_finance_agent_page()
}

buttons = [create_big_button(p) for p in pages]
main_out = widgets.Output()

def show(p):
    with main_out:
        clear_output()
        display(pages[p])

for b, name in zip(buttons, pages):
    b.on_click(lambda btn, n=name: show(n))

display(widgets.HBox(buttons), main_out)
show("Home")


