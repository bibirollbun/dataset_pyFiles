import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json


DATA_PATH = "/kaggle/input/lmsys-chatbot-arena"
TARGETS = ["winner_model_a", "winner_model_b", "winner_tie"]
train = pd.read_csv(DATA_PATH + "/train.csv")
test = pd.read_csv(DATA_PATH + "/test.csv")
sub = pd.read_csv(DATA_PATH + "/sample_submission.csv")


print(f"Data Shape | train {train.shape} | test {test.shape}")
print(f"-" * 50)
print(f">>> The First 3 Training Rows <<<")
display(train.head(3))
print(f"-" * 50)
print(f">>> The First 3 Test Rows <<<")
display(test.head(3))
assert train["id"].nunique() == len(train)


train = train.drop("id", axis=1)
print(f"After dropping 'id' column, shape of the training DataFrame becomes {train.shape}.")
print(f"-" * 50)
print(f">>> The First 3 Training Rows <<<")
train.head(3)


n_dups = train.duplicated(keep=False).sum()
print(f"There exist {n_dups} duplicated rows.")

n_rows1 = len(train)
train = train.drop_duplicates(keep="first", ignore_index=True)
n_rows2 = len(train)
print(f"After removing duplicates, #samples drops from {n_rows1} to {n_rows2}.")


train.isna().sum() #Data Quality pretty good


unknown_model_a = train["model_a"].isna().sum() + (train["model_a"] == "").sum()
unknown_model_b = train["model_b"].isna().sum() + (train["model_b"] == "").sum()
print(f"There exist {unknown_model_a} unknown model_a and {unknown_model_b} unknown model_b identities.")


demo_chat = train.iloc[0]
for col in ["prompt", "response_a", "response_b"]:
    assert isinstance(demo_chat[col], str)
    
print(f"=== Prompt ===")
print(demo_chat["prompt"])
print()
print(f"=== Response of {demo_chat['model_a']} ===") 
print(demo_chat["response_a"])
print()
print(f"=== Response of {demo_chat['model_b']} ===") 
print(demo_chat["response_b"])
print("-" * 50)
if demo_chat["winner_model_a"]:
    print(f">>> {demo_chat['model_a']} is the winner!!")
elif demo_chat["winner_model_b"]:
    print(f">>> {demo_chat['model_b']} is the winner!!")
else:
    print(f">>> Winner tie!!")


class ChatRenderer:
    
    CSS: str = """
        <style>
            lm-chat-body {
                display: flex;
                justify-content: center;
                align-items: flex-start;
                margin: 0;
                padding: 20px;
                font-family: Arial, sans-serif;
                background-color: #f5f5f5;
            }
            .lm-chat-container {
                display: flex;
                flex-direction: column;
                width: 100%;
                max-width: 1200px;
                border: 1px solid #ddd;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                background-color: #fff;
            }
            .lm-chat-panels {
                display: flex;
                width: 100%;
            }
            .lm-chat-panel {
                width: 50%;
                padding: 20px;
                box-sizing: border-box;
                border-right: 1px solid #ddd;
                position: relative;
            }
            .lm-chat-panel:last-child {
                border-right: none;
            }
            .lm-chat-model-header {
                font-weight: bold;
                margin-bottom: 10px;
                font-size: 14px;
                position: absolute;
                top: 10px;
                left: 20px;
                background-color: white;
                padding: 0 5px;
            }
            .lm-chat-prompt {
                background-color: #fdf5e6;
                padding: 10px;
                margin-top: 30px;
                margin-bottom: 20px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            .lm-chat-response {
                background-color: #fff;
                padding: 10px;
                margin-bottom: 20px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            .lm-chat-winner {
                text-align: center;
                padding: 10px;
                border-top: 1px solid #ddd;
                background-color: #f5f5f5;
                font-size: 16px;
            }
        </style>
    """

    def __init__(self, chat: pd.Series) -> None:
            self.model_a, self.model_b = chat["model_a"], chat["model_b"]
            self.prompt = chat["prompt"] if isinstance(chat["prompt"], list) else eval(chat["prompt"])
            self.res_a = chat["response_a"] if isinstance(chat["response_a"], list) else eval(chat["response_a"])
            self.res_b = chat["response_b"] if isinstance(chat["response_b"], list) else eval(chat["response_b"])
            self.targets = chat[TARGETS]
        
    def display(self, suppress_output: bool = False) -> None:
        if suppress_output:
            with io.capture_output() as captured:
                display(self._gen_html())
        else:
            display(self._gen_html())

    def _gen_html(self) -> HTML:
        html_content = f"""
            <script type="module" src="https://md-block.verou.me/md-block.js"></script>
            <html>
                <lm-chat-body>
                    <div class="lm-chat-container">
                        <div class="lm-chat-panels">
                            {self._gen_panel("a")}
                            {self._gen_panel("b")}
                        </div>
                        <div class="lm-chat-winner">
                            {self._get_winner()}
                        </div>
                    </div>
                </lm-chat-body>
            </html>
        """.encode("utf-16", "surrogatepass").decode("utf-16")
        
        html = HTML(self.CSS + html_content)
        
        return html

    def _gen_panel(self, model: str) -> str:
        res = self.res_a if model == "a" else self.res_b
        model_name = self.model_a if model == "a" else self.model_b
        panel = ""
        for p, r in zip(self.prompt, res):
            panel += f"""
                <div class="lm-chat-prompt">
                    <md-block>{p}</md-block>
                </div>
                <div class="lm-chat-response">
                    <md-block>{r}</md-block>
                </div>
            """
        panel = f"""
            <div class="lm-chat-panel">
                <div class="lm-chat-model-header">Model {model.upper()} - {model_name}</div>
                {panel}
            </div>
        """

        return panel

    
    def _get_winner(self) -> str:
        if self.targets["winner_model_a"] == 1:
            winner = f"Model A - <strong>{self.model_a}</strong> Wins!"
        elif self.targets["winner_model_b"] == 1:
            winner = f"Model B - <strong>{self.model_b}</strong> Wins!"
        else:
            winner = "Tie!"
        
        return winner


chat_renderer = ChatRenderer(train.iloc[5])
chat_renderer.display()


df_btl = (
    train.groupby(["model_a", "model_b"], as_index=False)
    .agg({"prompt": "count", **{winner: "sum" for winner in TARGETS}})
    .rename({"prompt": "battle_cnt"}, axis=1)
    .sort_values("battle_cnt", ascending=False)
    .reset_index(drop=True)
)
df_btl["model_a_win_rate"] = df_btl["winner_model_a"] / df_btl["battle_cnt"]
df_btl["model_b_win_rate"] = df_btl["winner_model_b"] / df_btl["battle_cnt"]
df_btl["tie_rate"] = df_btl["winner_tie"] / df_btl["battle_cnt"]
df_btl.head(3)


def check_turns(x):
    x = json.loads(x["prompt"])
    return len(x)
train["n_prompts"] = train.apply(check_turns, axis=1)

def check_turns_resa(x):
    x = json.loads(x["response_a"])
    return len(x)
train["n_res_a"] = train.apply(check_turns_resa, axis=1)

def check_turns_resb(x):
    x = json.loads(x["response_b"])
    return len(x)
train["n_res_b"] = train.apply(check_turns_resb, axis=1)
train.head()


colors = sns.color_palette("pastel")

n_turns_val_cnt = train["n_prompts"].value_counts().sort_index()
data, labels = n_turns_val_cnt.values, n_turns_val_cnt.index.tolist()
data_norm = data / np.sum(data) * 100
n_turns_max = np.max(labels)

fig, ax = plt.subplots(figsize=(20, 3))
sns.barplot(x=labels, y=data, palette=colors, ax=ax)
ax.text(4.5, 45000, "> 99.19% ←", c="r", fontsize=14, ha="right")
ax.axvline(4.5, c="g", linestyle="--")
ax.set_title(f"Distribution of #Turns | Max {n_turns_max}")
ax.set_xlabel("#Turns per Conversation")
ax.set_ylabel("Count")
for container in ax.containers:
    ax.bar_label(container)
plt.show()




