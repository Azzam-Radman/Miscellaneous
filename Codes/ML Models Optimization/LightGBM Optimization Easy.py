def optimize_LGBM(trial, x_train, x_valid, y_train, y_valid, 
                  min_child_weight=None, num_leaves=None, reg_lambda=None):
    
    # if you want to optimize the parameter, don't pass anything
    # if you want to use the default value, pass 'default'
    # if you want to specify a value, pass that value
    
    if min_child_weight == None:
        min_child_weight = trial.suggest_uniform("min_child_weight", 1e-3, 1e3)
    elif min_child_weight == 'default':
        min_child_weight = 1e-3
    else:
        min_child_weight = min_child_weight
        
    if num_leaves == None:
        num_leaves = trial.suggest_int("num_leaves", 10, 1000)
    elif num_leaves == 'default':
        num_leaves = 31
    else:
        num_leaves = num_leaves
        
    if reg_lambda == None:
        reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-2, 1e4)
    elif reg_lambda == 'default':
        reg_lambda = 0.0
    else:
        reg_lambda = reg_lambda
       
    
    params = {
            "min_child_weight": min_child_weight,
            "num_leaves": num_leaves,
            "reg_lambda": reg_lambda
            }

    model= lgbm.LGBMRegressor(
                           **params,
                           objective='mae',
                           metric='mae',
                           subsample=0.7,
                           learning_rate=0.03,
                           n_estimators=10000,
                           n_jobs=-1
                           )

    model.fit(
            x_train, y_train,
            eval_set=[(x_valid,y_valid)],
            verbose=100,
            early_stopping_rounds=100
            )

    
    valid_preds = model.predict(x_valid)
    train_preds = model.predict(x_train)
    
    score_valid = mean_absolute_error(y_valid, valid_preds)
    score_train = mean_absolute_error(y_train, train_preds)
    
    print("mae valid =", score_valid)
    print("mae train =", score_train)

    return score_valid
    
# optimize min_child_weight
import warnings
warnings.filterwarnings('ignore')

optimization_function = partial(optimize_LGBM, x_train=x_train, x_valid=x_valid,
                                y_train=y_train.target1, y_valid=y_valid.target1,
                                #min_child_weight,
                                num_leaves='default', 
                                reg_lambda='default'
                               )
study = optuna.create_study(direction='minimize')
study.optimize(optimization_function, n_trials=15)

dict_mae = dict()
dict_2 = study.best_params
dict_2['mae'] = study.best_value
dict_mae['params'] = dict_2
dict_mae['Number of finished trials'] = len(study.trials)

print(dict_mae)
