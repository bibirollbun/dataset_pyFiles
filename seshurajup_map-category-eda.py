import pandas as pd
from IPython.display import Markdown, display
from rich.console import Console
console = Console()
def rprint(text, field):
    console.print(f"[bright_black bold]{field.upper()}[/bright_black bold]:", end=' ')
    if field == 'question':
        console.print(f"[green]{text}[/green]")
    elif field == 'answers':
        console.print(f"[pink]{text[0]} (correct)[/pink] "+" ".join([f"[blue]{x}[/blue]" for x in text[1:]]))
    else:
        tags = field.lower().split("_")
        answer_highlight = "green" if tags[0] == 'true' else 'red'
        answer = text[0].replace('\(','').replace('\)','').strip()
        misconception_highlight = 'green' if tags[1] == 'correct' else ('red' if tags[1] == 'neither' else 'blue')
        console.print(f"\t[{answer_highlight}]Answer: {answer}[/{answer_highlight}]")
        console.print(f"\t[{misconception_highlight}]Explanation: {text[1]}[/{misconception_highlight}]")
        if 'misconception' in tags[1]:
            console.print(f"\t[{misconception_highlight} bold]Misconception found: {text[2]}[/{misconception_highlight} bold]")
        
print = rprint
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
train = train.fillna("No Misconception")
train


train = train[train['QuestionText'].str.contains("yellow")].sort_values(['Category','Misconception']).reset_index(drop=True)
train


question = "A box contains 24 yellow and green balls.\n 3/8  of the balls are yellow.\n\n How many of the balls are green?"
print(question, 'question')


answers = [15, 3, 8, 9]
print(answers,'answers')


selected_exampels = train.groupby(['MC_Answer','Category','Misconception']).sample(1).sort_values(['Category'], ascending=True).reset_index(drop=True)
selected_exampels


for index, record in train.groupby(['MC_Answer','Category','Misconception']).sample(1).iterrows():
    category = record['Category'].lower()
    student_answer = record['MC_Answer']
    print([record['MC_Answer'],record['StudentExplanation'],record['Misconception']], category)




