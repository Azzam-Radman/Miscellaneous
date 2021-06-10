# Cell 1

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import lightgbm as lgb
import optuna
from functools import partial
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as mlskf


# Cell 2

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv').drop('id', axis=1)
sample = pd.read_csv('sample_submission.csv')


# Cell 3

cat_feats = [f for f in train.columns if f not in ['id', 'target']]
X = train[cat_feats]
y = train.target
y_dummied = pd.get_dummies(y).values


# Cell 4

# first, optimize the min_child_weight
def optimize_LGBM(trial, x, y):
    
    params = {
            "min_child_weight": trial.suggest_uniform('min_child_weight', 1, 50)
            #"num_leaves": trial.suggest_int("num_leaves", 20, 80),
            #"lambda": trial.suggest_uniform('lambda', 1e-2, 1e3)
            }
            
    n_splits=5
    skf = mlskf(n_splits=n_splits,shuffle=True,random_state=0)
    log_loss = []
    
    # stratified KFold
    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y_dummied)):
        
        x_train, x_valid = x.iloc[train_idx, :], x.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        clf= lgb.LGBMClassifier(
                               **params,
                               metric='multi_logloss',
                               subsample=0.7,
                               learning_rate=0.03,
                               n_estimators=10000,
                               objective='multiclass',
                               num_class=9,
                               n_jobs=-1
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

optimization_function = partial(optimize_LGBM, x=X, y=y)
study = optuna.create_study(direction='minimize')
study.optimize(optimization_function, n_trials=10)

dict_log_loss = dict()
dict_2 = study.best_params
dict_2['log_loss'] = study.best_value
dict_log_loss['params'] = dict_2
dict_log_loss['Number of finished trials'] = len(study.trials)

print(dict_log_loss)


# Cell 6

# second, optimize num_laeves

def optimize_LGBM(trial, x, y):
    
    params = {
            "min_child_weight": 42.75786520869068, 	# from previous optimization
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            #"lambda": trial.suggest_uniform('lambda', 1e-2, 1e3)
            }
            
    n_splits=5
    skf = mlskf(n_splits=n_splits,shuffle=True,random_state=0)
    log_loss = []
    
    # stratified KFold
    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y_dummied)):
        
        x_train, x_valid = x.iloc[train_idx, :], x.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        clf= lgb.LGBMClassifier(
                               **params,
                               metric='multi_logloss',
                               subsample=0.7,
                               learning_rate=0.03,
                               n_estimators=10000,
                               objective='multiclass',
                               num_class=9,
                               n_jobs=-1
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

optimization_function = partial(optimize_LGBM, x=X, y=y)
study = optuna.create_study(direction='minimize')
study.optimize(optimization_function, n_trials=10)

dict_log_loss = dict()
dict_2 = study.best_params
dict_2['log_loss'] = study.best_value
dict_log_loss['params'] = dict_2
dict_log_loss['Number of finished trials'] = len(study.trials)

print(dict_log_loss)


# Cell 8

# eventually, optimize lambda_l1 or lambda_l2(lambda)

def optimize_LGBM(trial, x, y):
    
    params = {
            "min_child_weight": 42.75786520869068, # from previous optimization
            "num_leaves": 23,	# from previous optimization
            "lambda": trial.suggest_uniform('lambda', 0, 1e3)
            }
            
    n_splits=5
    skf = mlskf(n_splits=n_splits,shuffle=True,random_state=0)
    log_loss = []
    
    # stratified KFold
    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y_dummied)):
        
        x_train, x_valid = x.iloc[train_idx, :], x.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        clf= lgb.LGBMClassifier(
                               **params,
                               metric='multi_logloss',
                               subsample=0.7,
                               learning_rate=0.03,
                               n_estimators=10000,
                               objective='multiclass',
                               num_class=9,
                               n_jobs=-1
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

optimization_function = partial(optimize_LGBM, x=X, y=y)
study = optuna.create_study(direction='minimize')
study.optimize(optimization_function, n_trials=15)

dict_log_loss = dict()
dict_2 = study.best_params
dict_2['log_loss'] = study.best_value
dict_log_loss['params'] = dict_2
dict_log_loss['Number of finished trials'] = len(study.trials)

print(dict_log_loss)


# Cell 10

# finally, build your final model and compare training and validation performances
import warnings
warnings.filterwarnings('ignore')

params = {
        "min_child_weight": 42.75786520869068,
            "num_leaves": 23,
            "lambda": 666.3786684769562
        }


n_splits=5
seed_list=[0, 1, 2]
skf = mlskf(n_splits=n_splits,shuffle=True,random_state=0)
preds_valid_array = np.zeros((X.shape[0], y_dummied.shape[1]))
preds_test_array = np.zeros((test.shape[0], y_dummied.shape[1]))
log_loss_valid = []
log_loss_train = []

# stratified KFold
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y_dummied)):
    
    print("==========================")
    print(" Fold =", fold)
    x_train, x_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
      
    clf = lgb.LGBMClassifier(
                               **params,
                           metric='multi_logloss',
                           subsample=0.7,
                           learning_rate=0.03,
                           n_estimators=10000,
                           objective='multiclass',
                           num_class=9,
                           n_jobs=-1
                               )

    clf.fit(
            x_train,y=y_train,
            eval_set=[(x_valid,y_valid)],
            early_stopping_rounds=20,
            verbose=100
            )

    # predict validation and test sets
    preds_valid = clf.predict_proba(x_valid) # since this is multiclass classification do NOT use [:, 1]
    preds_test = clf.predict_proba(test)
    preds_train = clf.predict_proba(x_train)

    # fill the arrays
    preds_valid_array[valid_idx] += preds_valid / len(seed_list)
    preds_test_array += preds_test / (len(seed_list) * n_splits)
    # append scores
    log_loss_valid.append(metrics.log_loss(y_valid, preds_valid))
    log_loss_train.append(metrics.log_loss(y_train, preds_train))

print("Mean log loss valid =", np.mean(log_loss_valid), "std log_loss valid =", np.std(log_loss_valid, ddof=1))
print("Mean log loss train =", np.mean(log_loss_train), "std log_loss train =", np.std(log_loss_train, ddof=1))

sample.iloc[:, 1:] = preds_test_array
sample.to_csv('lgbm_opt_test.csv', index=False)

pd.DataFrame(preds_valid_array, columns=[f"feature_{i}" for i in range(preds_valid_array.shape[1])]).to_csv('lgbm_opt_valid.csv', index=False)
