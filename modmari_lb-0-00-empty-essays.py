import sys 
import pandas as pd


test_df = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")
test_df


def get_essays(df):
    
    # Load test data and create submission DataFrame
    submission = pd.DataFrame()
    submission['id'] = df['id']
    submission['essay'] = '  '
    
        
    return submission


if len(test_df) == 3:
    demo_df = test_df.loc[test_df.index.repeat(4)].reset_index(drop=True)
    demo_df = get_essays(demo_df)


%%time
submission = get_essays(test_df)
submission


submission.to_csv('submission.csv', index=False)







