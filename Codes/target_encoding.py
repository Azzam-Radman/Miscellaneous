# Cell 1

import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as mlskf
pd.set_option("display.max_rows", None, "display.max_columns", None)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv').drop('id', axis=1)
sample = pd.read_csv('sample_submission.csv')

X = train.drop('target', axis=1)
y = train.target
y_dummied = pd.get_dummies(y).values


# Cell 2

def target_encoding(df_train, df_valid, cols, target):
    
    data_frame = pd.DataFrame(columns=cols)
    
    for col in cols:
        
        dt = df_train.groupby(col)[target].agg(['mean']).reset_index(drop=False)
        tmp = df_valid.merge(dt, on=col, how='left')['mean'].values
        data_frame[col] = tmp
        
    return data_frame


# Cell 3

new_X = np.zeros_like(X.values, dtype='float64')
new_test = np.zeros_like(test.values, dtype='float64')

cols = X.columns.tolist()

n_splits = 5
    
skf = mlskf(n_splits=n_splits)
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y_dummied)):

    print("fold = ", fold)
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    full_train = pd.concat([x_train, y_train], axis=1)
    full_valid = pd.concat([x_valid, y_valid], axis=1)
    # target encode your categorical features
    target_enc_feats_valid = target_encoding(full_train, full_valid, cols, 'target')
    target_enc_feats_test = target_encoding(full_train, test, cols, 'target')
    
    X_test = target_enc_feats_test.fillna(0) # find a way to fill nulls
    X_valid = target_enc_feats_valid.fillna(0) # find a way to fill nulls
    
    x_valid, y_valid = X_valid, y_valid
    new_X[valid_idx] += x_valid.values
    new_test += X_test # only once
    new_test.to_csv('encoded_test.csv', index=False)
    # stratified cross validation
    y_dummied_2 = pd.get_dummies(y_train).values

    # nested stratified cross validation
    n_splits = 4
    skf = mlskf(n_splits=n_splits)

    full_train_holder = np.zeros((full_train.shape[0], full_train.shape[1]))
    for fold, (train_idx, valid_idx) in enumerate(skf.split(x_train, y_dummied_2)):
        x_train_2, y_train_2 = x_train.iloc[train_idx], y_train.iloc[train_idx]
        x_valid_2, y_valid_2 = x_train.iloc[valid_idx], y_train.iloc[valid_idx]

        full_train_2 = pd.concat([x_train_2, y_train_2], axis=1)
        full_valid_2 = pd.concat([x_valid_2, y_valid_2], axis=1)

        # target encode your categorical features
        target_enc_feats_valid = target_encoding(full_train_2, full_valid_2, cols, 'target')
        X_valid_2 = target_enc_feats_valid.fillna(0) # find a way to fill nulls
        x_valid, y_valid = X_valid_2, y_valid_2
        
        new_X[valid_idx] += x_valid.values
        
pd.DataFrame(new_X, columns=[f"feature_{i}" for i in range(new_X.shape[1])]).to_csv('encoded_X.csv', index=False)


