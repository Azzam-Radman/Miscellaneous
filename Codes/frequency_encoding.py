import pandas as pd
import numpy as np

def freq_enc(df, cols):
    
    for col in cols:
        
        counts = pd.DataFrame(train[col].value_counts()).reset_index(drop=False)
        
        df = df.merge(counts, left_on=[col], 
                      right_on=['index'], how='left', 
                      suffixes=('', '_count')).drop('index', axis=1)
        
    return df
  
columns_ = [f for f in train.columns if f not in ['target', 'id']]

train_len = len(train)
full_data = pd.concat([train, test], axis=0)
full_data = freq_enc(full_data, columns_)

train_freq_enc = full_data.iloc[:train_len, :].reset_index(drop=True)
test_freq_enc = full_data.iloc[train_len:, :].drop('target', axis=1).reset_index(drop=True)


train_freq_enc.to_csv('train_frequecy_encoded.csv', index=False)
test_freq_enc.to_csv('test_frequecy_encoded.csv', index=False)
