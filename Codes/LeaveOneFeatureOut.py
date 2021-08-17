
class LOFO(object):
    
    def __init__(self, data, labels, model, n_splits, eval_metric):
        self._data = data
        self._labels = labels
        self.model = model
        self.n_splits = n_splits
        self.eval_metric = eval_metric
        
        
    def kfold(self, x, y, model, n_splits, eval_metric):
        from sklearn import model_selection
        
        preds_valid_array = np.zeros(y.shape[0])
        
        train_scores = []
        valid_scores = []
        
        kf = model_selection.KFold(n_splits=n_splits)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(x)):
            
            print(f"===================== Fold {fold+1} =====================")
            x_train, y_train = x[train_idx, :], y[train_idx]
            x_valid, y_valid = x[valid_idx, :], y[valid_idx]
            
            self.model.fit(
                          x_train, y_train,
                          eval_set=[(x_valid, y_valid)],
                          verbose=100
                         )
            
            preds_valid = model.predict(x_valid)
            preds_train = model.predict(x_train)
            
            valid_score = np.sqrt(eval_metric(y_valid, preds_valid))
            train_score = np.sqrt(eval_metric(y_train, preds_train))
            
            valid_scores.append(valid_score)
            train_scores.append(train_score)
            
            preds_valid_array[valid_idx] += preds_valid
            
        print("Mean valid score =", np.mean(valid_scores), "STD valid score = ", np.std(valid_scores, ddof=1))
        print("Mean train score =", np.mean(train_scores), "STD train score = ", np.std(train_scores, ddof=1))
        
        cv_score = np.mean(valid_scores)
        return cv_score, preds_valid_array
    
    def selectionLoop(self, x, y, model, n_splits, eval_metric):
        
        print("All Features")
        cv_score, preds_valid = self.kfold(x, y, model, n_splits, eval_metric)
        score = cv_score
        scores = []
        good_scores = []
        scores.append(score)
        good_scores.append(score)
        harmful_features = []
        print("=================================================")
        
        
        for i in range(x.shape[1]):
            
            print(f"Drop Feature {i}")
            x2 = pd.DataFrame(x, columns=[f"col_{i}" for i in range(x.shape[1])])
            x2 = x2.drop(x2.columns[i], axis=1)
            x2 = x2.dropna(axis=1, how='all').values
            cv_score, preds_valid = self.kfold(x2, y, model, n_splits, eval_metric)  
            if cv_score < score:
                score = cv_score
                print("Improved Score =", score)
                good_scores.append(score)
                harmful_features.append(i)
                x = pd.DataFrame(x, columns=[f"col_{i}" for i in range(x.shape[1])])
                x.iloc[:, i] = np.nan
                x = x.values
                print("=================================================")
                
            else:
                continue
        
        print("Good scores :", good_scores)
        print("Harmful features :", harmful_features)
        
        return good_scores, harmful_features
    
    def transform(self, X):
        
        X = self._data
        y = self._labels
        model = self.model
        n_splits = self.n_splits
        eval_metric = self.eval_metric
        
        good_scores, harmful_features = self.selectionLoop(X.values, y.values, model, n_splits, eval_metric)
        X = X.drop(X.columns[harmful_features], axis=1)
        X = X.dropna(axis=1, how='all')
        test = test.drop(test.columns[harmful_features], axis=1)
        
        return X, test
      
# X and y should be dataframes      
lofo = LOFO(X, y, model=CatBoostRegressor(
                                   learning_rate=0.03,
                                   iterations=10000,
                                   loss_function='RMSE',
                                   eval_metric='RMSE',
                                   use_best_model=True,
                                   early_stopping_rounds=100
                                   ), n_splits=5, eval_metric=metrics.mean_squared_error)

X, test = LOFO.transform(lofo, X)
