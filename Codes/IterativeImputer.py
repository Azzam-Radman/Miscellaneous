# It finds missing values by modeling each feature with missing values as a function of others. 
# This process is done in a step-by-step round-robin fashion. At each step, a single feature with missing values 
# is chosen as a target (y) and the rest are chosen as feature array (X). 
# Then, a regressor is used to predict the missing values in y and this process is 
# continued for each feature until max_iter times (a parameter of IterativeImputer).

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

imp_mean = IterativeImputer(estimator=BayesianRidge())
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])

X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
X = imp_mean.transform(X)
