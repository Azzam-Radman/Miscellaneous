import pandas as pd

from sklearn import metrics

class GreedyFeatureSelection:
    """
    A simple and custom class for greedy feature selection.
    You will need to modify it quite a bit to make it suitable
    for your dataset.
    """
    def evaluate_score(self, X, y):
        """
        This function evaluates model on data and returns
        Area Under ROC Curve (AUC)
        Note: We fit the data  and calculate AUC on same data. 
        WE ARE OVERFITTING HERE.
        But this is also a way to achieve greedy selection.
        k-fold will take k times longer.

        If you want to implement it in really correct way, 
        calcuate OOF  AUC and return mean AUC over k folds.
        This requires only a few lines of change.

        :param X: training data
        :param y: targets
        :return: overfitted area under the roc curve
        """
        _index = (train['date'] < 20210401)
        x_train = X[_index]
        y_train = y[_index]
        x_valid = X[~_index]
        y_valid = y[~_index]

        model= lgbm.LGBMRegressor(
                       objective='mae',
                       metric='mae',
                       subsample=0.7,
                       learning_rate=0.03,
                       n_estimators=10000,
                       n_jobs=-1
                       )

        model.fit(
                x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                verbose=100,
                early_stopping_rounds=100
                )
        
        valid_preds = model.predict(x_valid)
        mae = mean_absolute_error(y_valid, valid_preds)
        return mae

    def _feature_selection(self, X, y):
        """
        This function does the actual greedy selection 
        :param X: data, numpy array
        :param y: targets, numpy array
        :return: (best scores, best features)
        """
        # initialize good features list
        # and best scores to keep track of both
        good_features = []
        best_scores = []

        # calculate the number of features
        num_features = X.shape[1]

        # infinite loop
        counter = 0
        while True:
            # initialize best feature and score of this loop
            this_feature = None
            best_score = 20

            # loop over all features
            for feature in range(num_features):
                # if feature is already in good features,
                # skip this for loop
                if feature in good_features:
                    continue
                # selected features are all godd features till now
                # and current feature
                selected_features = good_features + [feature]
                # remove all other features from data
                xtrain = X[:, selected_features]
                # calculate the score, in our case, AUC
                score = self.evaluate_score(xtrain, y)
                # if score is greater than the best score
                # of this loop, change best score and this feature
                if score < best_score:
                    this_feature = feature
                    best_score = score
                    counter += 1
                    print(f"Feature is {feature}")
                    print(f"Number of selected features {counter}")
                    # if we had selected a feature, add it
                    # to the good feature list and update best scores list
                    if this_feature != None:
                        good_features.append(this_feature)
                        best_scores.append(best_score)

                    # if we did not improve during the previous round,
                    # exit the while loop
                    if len(best_scores) > 2:
                        if best_scores[-1] > best_scores[-2]:
                            break
            # return best scores and good features
            # we need to remove the last data point as it is
            # the point the did not make an improvement
            return best_scores, good_features

    def __call__(self, X, y):
        """
        Call function will call the class on a set of arguments
        """
        # select features, return scores and selected indices
        scores, features = self._feature_selection(X, y)
        # transform data with selected features
        return X[:, features], features, scores

if __name__ == "__main__":
    
    X = train_X.values
    y = train_y.values

    # transform data by greedy feature selection
    X_transform, features, scores = GreedyFeatureSelection()(X, y)
