import pandas as pd
import numpy as np

def freq_enc(df, cols):
    
    new_df = df.copy()
    
    for col in cols:
        
        counts = df.groupby(col)[col].agg(['count']) / len(df)
        counts = counts.reset_index(drop=False)
        
        new_df[col] = df.merge(counts, on=col, how='left', 
                      suffixes=('', f'{col}_count'))['count']
        
    return new_df
  
columns_ = [f for f in train.columns if f not in ['target', 'id']]

train_len = len(train)
#train = train.drop('target', axis=1) # drop the target values first
full_data = pd.concat([train, test], axis=0)
full_data_enc = freq_enc(full_data, columns_)

train_freq_enc = full_data_enc.iloc[:train_len, :].reset_index(drop=True)
test_freq_enc = full_data_enc.iloc[train_len:, :].reset_index(drop=True)

train_freq_enc.to_csv('train_frequecy_encoded.csv', index=False)
test_freq_enc.to_csv('test_frequecy_encoded.csv', index=False)
