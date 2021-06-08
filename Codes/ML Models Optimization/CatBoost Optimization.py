# Multiclass problem

# Cell 1

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from catboost import CatBoostClassifier
import optuna
from functools import partial
!pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as mlskf

# Cell 2

train = pd.read_csv('../input/tps06basicnoduplitcates/train_no_duplicates.csv')
test = pd.read_csv('../input/tps06basicnoduplitcates/test.csv')
sample = pd.read_csv('../input/tps06basicnoduplitcates/sample_submission.csv')

# Cell 3

# in this dataset all features are categorical
cat_feats = [f for f in train.columns if f not in ['id', 'target']]
X = train[cat_feats]
y = train.target
y_dummied = pd.get_dummies(y).values

# Cell 4

def optimize_CB(trial, x, y):
    
    params = {
        "depth": trial.suggest_int("depth", 4, 12),
        "l2_leaf_reg": trial.suggest_loguniform('l2_leaf_reg', 1e-2, 1e2),
        "random_strength": trial.suggest_float("random_strength",0,3),
        "grow_policy":trial.suggest_categorical("grow_policy",["Depthwise","SymmetricTree","Lossguide"]),
        #'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300)
        }
            
    n_splits=5
    #seed_list=[0, 1, 2]
    skf = mlskf(n_splits=n_splits,shuffle=True,random_state=0)
    preds_valid = np.zeros((X.shape[1], y_dummied.shape[1]))
    log_loss = []
    
    # stratified KFold
    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y_dummied)):
        
        x_train, x_valid = x.iloc[train_idx, :], x.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        params["loss_function"] = 'MultiClass'
        params["cat_features"] = cat_feats

        cls = CatBoostClassifier(
                                   **params,
                                   task_type="GPU",
                                   border_count=128,
                                   learning_rate=0.03,
                                   iterations=10000,
                                   use_best_model=True,
                                   early_stopping_rounds=1000
                                   )

        cls.fit(x_train,y=y_train,
              embedding_features=None,
              use_best_model=True,
              eval_set=[(x_valid,y_valid)],
              verbose=1000)

        # predict validation set
        preds_valid = cls.predict_proba(x_valid) # since this is multiclass classification do NOT use [:, 1]

        # append scores
        log_loss.append(metrics.log_loss(y_valid, preds_valid))
    
    print("Mean log loss =", np.mean(log_loss), "std log_loss =", np.std(log_loss, ddof=1))
    cv_logloss = np.mean(log_loss)

    return cv_logloss



# Cell 5

optimization_function = partial(optimize_CB, x=X, y=y)
study = optuna.create_study(direction='minimize')
study.optimize(optimization_function, n_trials=25)

dict_log_loss = dict()
dict_2 = study.best_params
dict_2['log_loss'] = study.best_value
dict_log_loss['params'] = dict_2
dict_log_loss['Number of finished trials'] = len(study.trials)

print(dict_log_loss)
