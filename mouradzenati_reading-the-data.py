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


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")



import pandas as pd

df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
df.head()
import pandas as pd

df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")




df.head()



df.columns



df['Category'].value_counts()




question_counts = train['QuestionId'].value_counts()
question_counts.head(10)




most_common_question_text = train[train["QuestionId"] == 31772]["QuestionText"].iloc[0]
print(most_common_question_text)
num_unique_questions = train["QuestionId"].nunique()
print(f"Number of unique questions: {num_unique_questions}")
# نختار فقط الإجابات المرتبطة بالسؤال رقم 31772
q_31772 = df[df['QuestionId'] == 31772]

# نطبع بعض الإحصائيات
print("عدد إجابات الطلاب لهذا السؤال:", len(q_31772))
print("\nأنواع الإجابات (MC_Answer):")
print(q_31772['MC_Answer'].value_counts())

print("\nأنواع سوء الفهم (Misconception):")
print(q_31772['Misconception'].value_counts())



q_31772 = df[df['QuestionId'] == 31772]
miscon = q_31772[q_31772['MC_Answer'] == 'False_Misconception']


# عرض أول 10 تفسيرات
miscon['StudentExplanation'].head(10)



q_31772 = df[df['QuestionId'] == 31772]
q_31772['Category'].value_counts()




miscon = q_31772[q_31772['Category'] == 'False_Misconception']
miscon['StudentExplanation'].head(10)




import matplotlib.pyplot as plt

# إنشاء توزيع التصنيفات لكل سؤال
category_distribution = df.groupby('QuestionId')['Category'].value_counts().unstack().fillna(0)

# رسم الرسم البياني المكدس
category_distribution.plot(kind='bar', stacked=True, figsize=(14, 6))
plt.title('Distribution of Answer Categories per Question')
plt.xlabel('Question ID')
plt.ylabel('Number of Student Answers')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# حساب عدد مرات تكرار كل تفسير
repeated_explanations = df['StudentExplanation'].value_counts()

# عرض التفسيرات المتكررة أكثر من مرة فقط
repeated_explanations[repeated_explanations > 1].head(10)



# استخراج نصوص الأسئلة الفريدة وعددها
unique_questions = df[['QuestionId', 'QuestionText']].drop_duplicates()
unique_questions.reset_index(drop=True, inplace=True)

# عرض جميع الأسئلة
unique_questions



miscon['StudentExplanation'].head(20)



# حساب عدد التفسيرات الخاطئة لكل سؤال
miscon_per_question = df[df['Category'] == 'False_Misconception']['QuestionId'].value_counts(normalize=True) * 100

# عرض أعلى 10 أسئلة من حيث نسبة الأخطاء المفاهيمية
miscon_per_question.head(10)



# حساب عدد التفسيرات الخاطئة لكل سؤال
miscon_per_question = df[df['Category'] == 'False_Misconception']['QuestionId'].value_counts(normalize=True) * 100

# عرض أعلى 10 أسئلة من حيث نسبة الأخطاء المفاهيمية
miscon_per_question.head(10)



# استخراج فقط التفسيرات الخاطئة للسؤال 31772
df[(df['QuestionId'] == 31772) & (df['Category'] == 'False_Misconception')]['StudentExplanation'].sample(10, random_state=42)


