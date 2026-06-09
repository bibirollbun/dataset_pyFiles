#https://www.kaggle.com/code/redonsh/assignment-1-group-b/edit/run/233042755


#


from IPython.display import Image, display
display(Image("/kaggle/input/picture2/screenshot.png"))


import pandas as pd
# let's sort it as per given submission sample
sub = pd.read_csv('/kaggle/input/submission/submission (27).csv')
sub


# Your final best submission
sub.to_csv('submission.csv', index=False)
!head submission.csv




