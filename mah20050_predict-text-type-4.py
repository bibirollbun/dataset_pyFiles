import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


sample_submission = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/sample_submission.csv')
test = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')


sample_submission.head()


test.head()


essay = ['Compare and Contrast: The Importance of Self-Reliance and Adaptability in Healthcare Self-reliance and adaptability are both critical in healthcare but serve distinct purposes. Aaaah Self-reliance emphasizes individual responsibility, enabling professionals to make informed decisions and perform tasks independently, which ensures efficiency in patient care. In contrast, adaptability focuses on the ability to adjust to changing circumstances, such as new medical technologies or unexpected patient needs, ensuring resilience in dynamic environments. While self-reliance builds confidence and autonomy, adaptability fosters flexibility and innovation. Both are essential, complementing each other to enhance the overall quality and responsiveness of healthcare services.',
         'Management consulting plays a crucial role in resolving conflicts within marketing teams by providing an unbiased perspective and specialized expertise.Hmm.. Consultants identify the root causes of disputes, whether related to strategy, resource allocation, or team dynamics. They implement tailored conflict resolution strategies, facilitate communication, and align team goals with organizational objectives. However, the effectiveness of consulting depends on the consultant’s understanding of the company culture and their ability to foster collaboration. While consulting can drive short-term solutions, sustaining harmony requires the organization’s commitment to implementing long-term changes. Overall, management consulting is effective but not a standalone solution.',
         'Self-reliance is vital for success in software engineering, as it fosters problem-solving, adaptability, and continuous learning. Engineers often encounter challenges requiring independent research and decision-making. By relying on their skills, they can debug issues, explore new technologies, and innovate solutions without constant guidance. Self-reliance also encourages accountability and resilience, traits crucial in a fast-paced industry. However, balancing independence with collaboration is essential, as teamwork often drives significant progress in complex projects. Ultimately, self-reliance empowers software engineers to take initiative, grow professionally, and contribute meaningfully to their teams and organizations.']

sample_submission['essay'] = essay
sample_submission.head()


notes = ['word_Aaaah','word_Hmm..','word_they can']
test['notes'] = notes
test.head()


evaluation = []
for index, essay_text in enumerate(test['notes']):  # Unpack the tuple into index and essay_text
    if 'word_Hmm..' in essay_text or 'word_Aaaah' in essay_text:  # Check if the strings are present
        evaluation.append(0)
    else:
        evaluation.append(1)
test['evaluation'] = evaluation
test.head()


test = test.drop(['topic','id'],axis=1)
test.head()


sns.countplot(x='evaluation', data=test , palette=['r','g'])


fainl = pd.get_dummies(test, columns=['notes'], prefix=['notes'])

x = fainl.drop('evaluation', axis=1)
y = fainl['evaluation']
y = y.astype('int')

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

fainl.head()


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
score = logreg.score(x_train, y_train)
score2 = logreg.score(x_test, y_test)
print("Training set accuracy: ", '%.3f'%(score))
print("testing set accuracy:", '%.3f'%(score2))

