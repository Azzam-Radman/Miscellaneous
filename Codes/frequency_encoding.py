import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv').drop('id', axis=1)
sample = pd.read_csv('sample_submission.csv')

cols = [f for f in train.columns if f not in ['id', 'target']]
X = train[cols]
len_X = len(X)
y = train['target']
full_data = pd.concat([X, test], axis=0).reset_index(drop=False)

def freq_enc(df, cols):
    
    for col in cols:
        grouped = df.groupby(col)['index'].agg('count') / len(df)
        grouped = grouped.to_frame().reset_index(drop=False).rename(columns={'index': f'{col}_count'})
        merged = df.merge(grouped, on=col, how='left').drop('index', axis=1)
        
        df[f"{col}_count"] = merged[f'{col}_count']
    return df


full_encoded = freq_enc(full_data, cols)
X_enc = iii.iloc[:len_X, :].drop(['index']+cols, axis=1)
test_enc = iii.iloc[len_X:, :].drop(['index']+cols, axis=1)

train_enc = pd.concat([X_enc, y], axis=1)
train_enc.to_csv('train_encoded.csv', index=False)
test_enc.to_csv('test_encoded.csv', index=False)
