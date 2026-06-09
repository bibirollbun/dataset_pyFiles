from metric import score


import pandas as pd
row_id_column_name = "id"
solution = pd.DataFrame({'id': range(4),
                         'gesture': ['Eyebrow - pull hair']*4})
submission = pd.DataFrame({'id': range(4), 
                           'gesture': ['Forehead - pull hairline']*4})




score(solution, submission, row_id_column_name=row_id_column_name)

