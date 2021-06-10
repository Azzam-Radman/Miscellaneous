# Multiclass problem

# Cell 1

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import xgboost as xgb
import optuna
from functools import partial
!pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as mlskf

# Cell 2

train = pd.read_csv('../input/tabular-playground-series-may-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2021/test.csv').drop('id', axis=1)
sample = pd.read_csv('../input/tabular-playground-series-may-2021/sample_submission.csv')

# Cell 3

# in this dataset all features are categorical
cat_feats = [f for f in train.columns if f not in ['id', 'target']]
X = train[cat_feats]
y = train.target
y_dummied = pd.get_dummies(y).values

# Cell 4

# you might need to perform opt. in two to three stages
# Stage 1

def optimize_XGB(trial, x, y):
    
    params = {
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "lambda": trial.suggest_loguniform('lambda', 1e-2, 1e2),
            "grow_policy":trial.suggest_categorical("grow_policy",["depthwise", "lossguide"])
            }
            
    n_splits=5
    skf = mlskf(n_splits=n_splits,shuffle=True,random_state=0)
    log_loss = []
    
    # stratified KFold
    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y_dummied)):
        
        x_train, x_valid = x.iloc[train_idx, :], x.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        clf= xgb.XGBClassifier(
                               **params,
                               eval_metric='mlogloss',
                               subsample=0.7,
                               tree_method='gpu_hist',
                               learning_rate=0.03,
                               n_estimators=10000,
                               objective='multi:softprob',
                               num_class=9
                                )

        clf.fit(
                x_train,y=y_train,
                eval_set=[(x_valid,y_valid)],
                early_stopping_rounds=20,
                verbose=100
                )

        # predict validation set
        #best_iteration = clf.get_booster().best_ntree_limit
        #min_ = best_iteration-10
        #max_ = best_iteration+10
        preds_valid = clf.predict_proba(x_valid)#, ntree_limit=) # since this is multiclass classification do NOT use [:, 1]

        # append scores
        log_loss.append(metrics.log_loss(y_valid, preds_valid))
    
    print("Mean log loss =", np.mean(log_loss), "std log_loss =", np.std(log_loss, ddof=1))
    cv_logloss = np.mean(log_loss)

    return cv_logloss


# Cell 5

import warnings
warnings.filterwarnings('ignore')

optimization_function = partial(optimize_XGB, x=X, y=y)
study = optuna.create_study(direction='minimize')
study.optimize(optimization_function, n_trials=15)

dict_log_loss = dict()
dict_2 = study.best_params
dict_2['log_loss'] = study.best_value
dict_log_loss['params'] = dict_2
dict_log_loss['Number of finished trials'] = len(study.trials)

print(dict_log_loss)


# Cell 6 -----> reduce max_depth and lambda spaces

# Stage 2

def optimize_XGB_2(trial, x, y):
    
    params = {
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "lambda": trial.suggest_uniform('lambda', 50, 3e2)
            }
            
    n_splits=5
    skf = mlskf(n_splits=n_splits,shuffle=True,random_state=0)
    log_loss = []
    
    # stratified KFold
    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y_dummied)):
        
        x_train, x_valid = x.iloc[train_idx, :], x.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        clf= xgb.XGBClassifier(
                               **params,
                               eval_metric='mlogloss',
                               subsample=0.7,
                               tree_method='gpu_hist',
                               learning_rate=0.03,
                               n_estimators=10000,
                               objective='multi:softprob',
                               num_class=9,
                               grow_policy='lossguide'
                                )

        clf.fit(
                x_train,y=y_train,
                eval_set=[(x_valid,y_valid)],
                early_stopping_rounds=20,
                verbose=100
                )

        # predict validation set
        #best_iteration = clf.get_booster().best_ntree_limit
        #min_ = best_iteration-10
        #max_ = best_iteration+10
        preds_valid = clf.predict_proba(x_valid)#, ntree_limit=) # since this is multiclass classification do NOT use [:, 1]

        # append scores
        log_loss.append(metrics.log_loss(y_valid, preds_valid))
    
    print("Mean log loss =", np.mean(log_loss), "std log_loss =", np.std(log_loss, ddof=1))
    cv_logloss = np.mean(log_loss)

    return cv_logloss

# Cell 7

import warnings
warnings.filterwarnings('ignore')

optimization_function = partial(optimize_XGB_2, x=X, y=y)
study = optuna.create_study(direction='minimize')
study.optimize(optimization_function, n_trials=15)

dict_log_loss = dict()
dict_2 = study.best_params
dict_2['log_loss'] = study.best_value
dict_log_loss['params'] = dict_2
dict_log_loss['Number of finished trials'] = len(study.trials)

print(dict_log_loss)

# Cell 8 reduce max_depth and lambda spaces even more if needed

import warnings
warnings.filterwarnings('ignore')

optimization_function = partial(optimize_XGB_2, x=X, y=y)
study = optuna.create_study(direction='minimize')
study.optimize(optimization_function, n_trials=15)

dict_log_loss = dict()
dict_2 = study.best_params
dict_2['log_loss'] = study.best_value
dict_log_loss['params'] = dict_2
dict_log_loss['Number of finished trials'] = len(study.trials)

print(dict_log_loss)

# Cell 8

# Stage 3

def optimize_XGB_3(trial, x, y):
    
    params = {
            "max_depth": trial.suggest_int("max_depth", 6, 8),
            "lambda": trial.suggest_uniform('lambda', 280, 1500)
            }
            
    n_splits=5
    skf = mlskf(n_splits=n_splits,shuffle=True,random_state=0)
    log_loss = []
    
    # stratified KFold
    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y_dummied)):
        
        x_train, x_valid = x.iloc[train_idx, :], x.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        clf= xgb.XGBClassifier(
                               **params,
                               eval_metric='mlogloss',
                               subsample=0.7,
                               tree_method='gpu_hist',
                               learning_rate=0.03,
                               n_estimators=10000,
                               objective='multi:softprob',
                               num_class=9,
                               grow_policy='lossguide'
                                )

        clf.fit(
                x_train,y=y_train,
                eval_set=[(x_valid,y_valid)],
                early_stopping_rounds=20,
                verbose=100
                )

        # predict validation set
        #best_iteration = clf.get_booster().best_ntree_limit
        #min_ = best_iteration-10
        #max_ = best_iteration+10
        preds_valid = clf.predict_proba(x_valid)#, ntree_limit=) # since this is multiclass classification do NOT use [:, 1]

        # append scores
        log_loss.append(metrics.log_loss(y_valid, preds_valid))
    
    print("Mean log loss =", np.mean(log_loss), "std log_loss =", np.std(log_loss, ddof=1))
    cv_logloss = np.mean(log_loss)

    return cv_logloss


# Cell 9

import warnings
warnings.filterwarnings('ignore')

optimization_function = partial(optimize_XGB_3, x=X, y=y)
study = optuna.create_study(direction='minimize')
study.optimize(optimization_function, n_trials=15)

dict_log_loss = dict()
dict_2 = study.best_params
dict_2['log_loss'] = study.best_value
dict_log_loss['params'] = dict_2
dict_log_loss['Number of finished trials'] = len(study.trials)

print(dict_log_loss)


# Cell 10 ------> train the final model and evaluate its perfomance

import warnings
warnings.filterwarnings('ignore')

params = {'max_depth': 7, 'lambda': 606.0800135737054, 'grow_policy': 'lossguide'}


n_splits=5
seed_list=[0, 1, 2]
skf = mlskf(n_splits=n_splits,shuffle=True,random_state=0)
preds_valid_array = np.zeros((X.shape[0], y_dummied.shape[1]))
preds_test_array = np.zeros((test.shape[0], y_dummied.shape[1]))
log_loss_valid = []

# stratified KFold
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y_dummied)):
    
    print("==========================")
    print(" Fold =", fold)
    x_train, x_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    for seed in seed_list:
        
        print("Seed =", seed)
        params['random_state'] = seed

        clf = xgb.XGBClassifier(
                               **params,
                               eval_metric='mlogloss',
                               subsample=0.7,
                               tree_method='gpu_hist',
                               learning_rate=0.03,
                               n_estimators=10000,
                               objective='multi:softprob',
                               num_class=9
                               )

        clf.fit(
                x_train,y=y_train,
                eval_set=[(x_valid,y_valid)],
                early_stopping_rounds=20,
                verbose=100
             )

        # predict validation and test sets
        preds_valid = clf.predict_proba(x_valid) # since this is multiclass classification do NOT use [:, 1]
        preds_test = clf.predict_proba(test.drop('id', axis=1))
        
        # fill the arrays
        preds_valid_array[valid_idx] += preds_valid / len(seed_list)
        preds_test_array += preds_test / (len(seed_list) * n_splits)
        # append scores
        log_loss_valid.append(metrics.log_loss(y_valid, preds_valid))

print("Mean log loss =", np.mean(log_loss_valid), "std log_loss =", np.std(log_loss_valid, ddof=1))

sample.iloc[:, 1:] = preds_test_array
sample.to_csv('xgb_test.csv', index=False)

pd.DataFrame(preds_valid_array, columns=[f"feature_{i}" for i in range(preds_valid_array.shape[1])]).to_csv('xgb_valid.csv', index=False)
