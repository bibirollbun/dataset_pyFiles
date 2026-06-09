import pandas as pd
from IPython.display import display, HTML

train = pd.read_csv("/kaggle/input/drawing-with-llms/published/train.csv")
train


questions = pd.read_parquet("/kaggle/input/drawing-with-llms/published/questions.parquet")
questions


def display_description_with_questions(train_df, questions_df, index, image_id):
    description = train_df[train_df['id'] == image_id]['description'].values
    if len(description) == 0:
        display(HTML(f"<p style='color: red; font-weight: bold;'>No description found for ID: {image_id}</p>"))
        return
    
    description = description[0]
    questions_data = questions_df[questions_df['id'] == image_id][['question', 'choices', 'answer']]

    desc_html = f"""
    <div style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; 
                font-size: 18px; text-align: left; margin-bottom: 5px;">
        {index} &nbsp; | &nbsp;  {image_id} &nbsp; | &nbsp; <b> {description.upper()} </b>
    </div>
    """

    table_html = """
    <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; text-align: left;">
        <thead style="background-color: #f4a261; color: white;">
            <tr>
                <th style="padding: 8px;">Question</th>
                <th style="padding: 8px;">Choices</th>
                <th style="padding: 8px;">Answer</th>
            </tr>
        </thead>
        <tbody>
    """

    for _, row in questions_data.iterrows():
        table_html += f"""
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="padding: 8px;">{row['question']}</td>
            <td style="padding: 8px; color: #1565c0;">{row['choices']}</td>
            <td style="padding: 8px; background-color: #c8e6c9; font-weight: bold;">{row['answer']}</td>
        </tr>
        """

    table_html += "</tbody></table><hr style='border: 2px solid black; margin: 20px 0;'>"
    
    display(HTML(desc_html + table_html))


for i, row in train.iterrows():
    sample_id = row['id']
    display_description_with_questions(train, questions, i+1, sample_id)




