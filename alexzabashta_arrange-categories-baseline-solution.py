import pandas as pd 
import os    


directory = '/kaggle/input/arrange-categories/datasets/'

row_id = []
dataset_id = []
cat_id = []
order = []


for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    df = pd.read_csv(f)

    d,n,m = map(int,filename[:-4].split("_")) 

    for cat in df:
        c = cat[3:]
        row_id.append(str(d) + "_"+ str(c))
        dataset_id.append(d)
        cat_id.append(c)    
        order.append(" ".join(map(str,df[cat].value_counts().index)))    


submission = pd.DataFrame({"ID" : row_id, "dataset" : dataset_id, "cat" : cat_id, "order" : order})
submission.to_csv("submission.csv", index=False)

