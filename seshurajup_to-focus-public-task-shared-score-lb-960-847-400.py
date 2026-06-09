best_submission_zip = "/kaggle/input/pipeline-new-way-for-task31-126-51/submission.zip" # Add your Submission to dataset and change path


import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import zipfile
df = pd.read_csv("/kaggle/input/code-golf-update-public-task-scores/public_task_score_lb.csv")
warnings.filterwarnings("ignore", category=RuntimeWarning)
df['task'] = df['task'].astype(str).str.zfill(3)
scores_df = df.set_index("task")
user_columns = scores_df.columns
scores_df = scores_df.apply(pd.to_numeric, errors="coerce")
lb_score_df = scores_df.apply(lambda x: x.dropna().min(), axis=1)
lb_score_df.columns = ["code_bytes"]
tasks_solved = lb_score_df.notna().sum()
tasks_code_bytes = lb_score_df.sum()
lb_score = 2500*tasks_solved - tasks_code_bytes
tasks_solved, tasks_code_bytes, lb_score

from IPython.display import display, HTML
display(HTML(f"""
<div style="font-family: monospace; font-size: 16px;">
    <p><strong>Total tasks:</strong> <span style="color: #1f77b4;">{df.shape[0]}</span></p>
    <p><strong>Total users:</strong> <span style="color: #ff7f0e;">{df.shape[1] - 1}</span></p>
    <p><strong style="color: green;">âœ… Total Tasks Solved:</strong> <span style="color: green; font-weight: bold;">{tasks_solved}</span></p>
    <p><strong>Total Avg Task Bytes:</strong> <span style="color: #9467bd;">{tasks_code_bytes/tasks_solved:.2f}</span></p>
    <p><strong style="color: red; font-size: 18px;">ğŸ�† Total LB Score:</strong> <span style="color: red; font-weight: bold; font-size: 20px;">{lb_score}</span></p>
</div>
"""))
top_score_rows = []
for task, row in scores_df.iterrows():
    min_val = row.min()
    top_users = row[row == min_val].dropna().index.tolist()
    top_score_rows.append({
        "task": task,
        "top_score": min_val,
        "top_users": '|'.join(top_users)
    })
top_scores_df = pd.DataFrame(top_score_rows)
top_scores_df = top_scores_df.sort_values(['top_score'])

task_info_list = []
with open(best_submission_zip, "rb") as f:
    with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
        for file_name in zf.namelist():
            if file_name.endswith(".py"):
                with zf.open(file_name) as file:
                    solution = file.read()
                    task_num = int(file_name.split("/")[-1].replace("task", "").replace(".py", ""))
                    task_info_list.append({
                        "task": task_num,
                        "score": max([0.1, 2500 - len(solution)]),
                        "size": min(2500, len(solution))
                    })

my_scores_df = pd.DataFrame(task_info_list)
top_scores_df["task"] = top_scores_df["task"].astype(int)
my_scores_df["task"] = my_scores_df["task"].astype(int)
focus_scores_df = top_scores_df[["task","top_score"]].merge(my_scores_df[["task", "score", "size"]], on="task", how="left")
focus_scores_df["focus"] = abs(focus_scores_df["top_score"] - focus_scores_df["size"])


from IPython.display import display, Markdown, Latex
splits = 10
def focus_leaderboard(df):
    df = df.sort_values('focus', ascending=False).reset_index(drop=True)
    nz = df[df.focus>0]['focus']
    bins = [0] + ([nz.min() + (nz.max()+1-nz.min())*i/(splits-1) for i in range(splits)] if len(nz)>0 else [0]*splits) + [nz.max()+1 if len(nz)>0 else 1]
    cols = ['#c6efce','#d9f0a3','#fff59d','#ffeb84','#ffd966','#ffc000','#ff9c66','#ff704d','#ff4d4d','#ff1a1a']
    labels = ['! Solved !'] + [f'{int(bins[i]):03d}-{int(bins[i+1]-1):03d}' for i in range(1,splits)]

    display(Markdown("<br>".join([
        f"<span style='background-color:{cols[i%len(cols)]}; padding:4px 30px; border-radius:5px; font-size:24px;'>{labels[i]}    -- {df[(df.focus>=bins[i])&(df.focus<bins[i+1])].shape[0]:03d} tasks - {int(df[(df.focus>=bins[i])&(df.focus<bins[i+1])]['focus'].sum()):05d} score</span>" 
        for i in range(splits)
    ])))
    
    def style_row(r):
        if r.focus==0: return [f'background-color:{cols[0]};']*len(r)
        for i in range(1,splits):
            if bins[i]<=r.focus<bins[i+1]: return [f'background-color:{cols[i%len(cols)]};']*len(r)
        return ['']*len(r)

    display(df.style.apply(style_row, axis=1)
            .format({'top_score':'{:.0f}','size':'{:.0f}','score':'{:.1f}','focus':'{:.0f}'})
            .bar(subset=['top_score','size'], color=['#a6cee3','#1f78b4']))
! cp $best_submission_zip /kaggle/working/


focus_leaderboard(focus_scores_df)


from IPython.display import display, Markdown, Latex
top_score_rows = []
for task, row in scores_df.iterrows():
    min_val = row.min()
    top_users = row[row == min_val].dropna().index.tolist()
    top_score_rows.append({
        "task": task,
        "top_score": min_val,
        "top_users": '|'.join(top_users)
    })
top_scores_df = pd.DataFrame(top_score_rows)
top_scores_df = top_scores_df.sort_values(['top_score'])
display(Markdown(top_scores_df.set_index("task").to_markdown()))


import pandas as pd
import matplotlib.pyplot as plt

bucket_edges = list(range(0, 401, 20))
bucket_labels = [f"{start}â€“{end - 1}" for start, end in zip(bucket_edges[:-1], bucket_edges[1:])]
bucket_labels.append("400+")

def bucket_fixed(x):
    if pd.isna(x):
        return None
    for i in range(len(bucket_edges) - 1):
        if bucket_edges[i] <= x < bucket_edges[i + 1]:
            return bucket_labels[i]
    return "400+"

bucketed_df = scores_df.apply(lambda col: col.map(bucket_fixed))

bucket_counts = bucketed_df.apply(lambda col: col.value_counts()).fillna(0)
bucket_counts = bucket_counts.reindex(bucket_labels).fillna(0).T

plt.figure(figsize=(18, 9))
bucket_counts.plot(kind="bar", stacked=True, figsize=(18, 7))
plt.xticks(rotation=45)
plt.legend(title="#Taks by Users", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()


solved_counts = scores_df.notna().sum(axis=1)
solved_df = pd.DataFrame({"task": solved_counts.index, "users_solved": solved_counts.values})
solved_df["task"] = pd.to_numeric(solved_df["task"], errors="coerce")


solved_df = solved_df.sort_values("task").reset_index(drop=True)


chunk_size = 100
num_chunks = len(solved_df) // chunk_size + int(len(solved_df) % chunk_size != 0)
chunks = [solved_df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]


for i, chunk in enumerate(chunks):
    plt.figure(figsize=(16, 4))
    sns.barplot(data=chunk, x="task", y="users_solved", palette="viridis")
    plt.title(f"Users Solving Tasks â€” Tasks {chunk['task'].iloc[0]} to {chunk['task'].iloc[-1]}")
    plt.xlabel("Task Number")
    plt.ylabel("Users Solved")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()




