import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

PLOT_TREE_IMPORTANCE = 0
# load and apply XGB on the trimmed and preprocessed dataset
df = pd.read_csv("../../datasets/ames_housing_trimmed_processed.csv")
print(df.shape)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# test = pd.read_csv("../input/test.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("trimmed dataset (using sklearn API) RMSE: %f" % (rmse))


# use learning API with linear base learner on trimmed dataset
DM_train = xgb.DMatrix(data=X_train, label=y_train)

DM_test = xgb.DMatrix(data=X_test, label=y_test)

# use linear as base learner
params = {"booster": "gblinear", "objective": "reg:linear"}

xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=5)

preds = xg_reg.predict(DM_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("trimmed dataset (using Learning API, linear as base learner) RMSE: %f" % (rmse))

# use tree as base learner
params = {"booster": "gbtree", "objective": "reg:linear"}

xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=5)

preds = xg_reg.predict(DM_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("trimmed dataset (using Learning API, tree as base learner) RMSE: %f" % (rmse))


# load and apply XGB on the unpreprocessed dataset
train = pd.read_csv("../../datasets/ames_unprocessed_data.csv")


print(train.shape)
print(train.info())

# only LotFrontage has missing values
train.LotFrontage = train.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (train.dtypes == 'object')

# Get list of categorical column names
categorical_columns = train.columns[categorical_mask].tolist()

# non_categorical_columns = train.columns[~categorical_mask].tolist()

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
# train[categorical_columns] = train[categorical_columns].apply(lambda x: le.fit_transform(x))
train[categorical_columns] = train[categorical_columns].apply(lambda x: le.fit_transform(x))


# Print the head of the LabelEncoded categorical columns
print(train[categorical_columns].head())

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categorical_features=categorical_mask, sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: train_encoded
train_encoded = ohe.fit_transform(train)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(type(train_encoded))

# Print the shape of the original DataFrame
print(train.shape)

# Print the shape of the transformed array
print(train_encoded.shape)

X = train.iloc[:, :-1]
y = train.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("unprocessed dataset (using sklearn API) RMSE: %f" % (rmse))

# Linear Base Learner has to use learning API only
DM_train = xgb.DMatrix(data=X_train, label=y_train)

DM_test = xgb.DMatrix(data=X_test, label=y_test)

params = {"booster": "gblinear", "objective": "reg:linear"}

xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=10)

preds = xg_reg.predict(DM_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("unprocessed dataset (using Learning API, linear as base learner) RMSE: %f" % (rmse))

# tree base learner in learning API, notice the score is much higher than sklearn API
params = {"booster": "gbtree", "objective": "reg:linear"}

xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=10)

preds = xg_reg.predict(DM_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("unprocessed dataset (using Learning API, decision tree as base learner) RMSE: %f" % (rmse))

X_dmatrix = xgb.DMatrix(data=X, label=y)

if PLOT_TREE_IMPORTANCE:
    # Visualizing individual XGBoost trees
    params = {"objective": "reg:linear", "max_depth": 2}
    # Train the model: xg_reg
    xg_reg = xgb.train(params=params, dtrain=X_dmatrix, num_boost_round=10)

    # Plot the first tree
    xgb.plot_tree(xg_reg, num_trees=0)
    plt.show()

    # Plot the fifth tree
    xgb.plot_tree(xg_reg, num_trees=4)
    plt.show()

    # Plot the last tree sideways
    xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR")
    plt.show()

    # Visualizing feature importances: What features are most important in the dataset
    # Counting the number of times each feature is split on across all boosting rounds (trees) in the model, and then visualizing the result as a bar graph, with the features ordered according to how many times they appear. XGBoost has a plot_importance() function that allows you to do exactly this.
    params = {"objective": "reg:linear", "max_depth": 4}

    # Train the model: xg_reg
    xg_reg = xgb.train(params=params, dtrain=X_dmatrix, num_boost_round=10)

    # Plot the feature importances
    xgb.plot_importance(xg_reg)
    plt.show()


# tune regularization in base learner
params = {"objective": "reg:linear", "max_depth": 4}

reg_params = [0.1, 1, 10]
rmses_reg = []

for reg in reg_params:
    params["lambda"] = reg
    cv_results = xgb.cv(dtrain=X_dmatrix, params=params, nfold=4,
                        num_boost_round=10, metrics="rmse", as_pandas=True, seed=123)
    rmses_reg.append(cv_results["test-rmse-mean"].tail(1).values[0])

print("Best rmse as a function of reg:")
print(pd.DataFrame(list(zip(reg_params, rmses_reg)), columns=["reg", "rmse"]))

# tune num_boost_round in base learner
# Create the parameter dictionary for each tree: params
params = {"objective": "reg:linear", "max_depth": 3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=X_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)

    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses, columns=["num_boosting_rounds", "rmse"]))


# Automated boosting round selection using early_stopping
# Create the parameter dictionary for each tree: params
params = {"objective": "reg:linear", "max_depth": 4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=X_dmatrix, params=params, nfold=3, num_boost_round=50, metrics="rmse", early_stopping_rounds=10, as_pandas=True, seed=123)

# Print cv_results
print(cv_results)


# tune eta (learning rate)
# Create the parameter dictionary for each tree (boosting round)
params = {"objective": "reg:linear", "max_depth": 3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta
for curr_val in eta_vals:

    params["eta"] = curr_val

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=X_dmatrix, params=params, nfold=3, num_boost_round=10,
                        metrics="rmse", early_stopping_rounds=5, as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta", "best_rmse"]))

# tune eta (learning rate)
# Create the parameter dictionary for each tree (boosting round)
params = {"objective": "reg:linear"}

# Create list of max_depth values
max_depths = [2, 5, 10, 20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:

    params["max_depth"] = curr_val

    # Perform cross-validation
    cv_results = xgb.cv(dtrain=X_dmatrix, params=params, nfold=2,
                        num_boost_round=10, metrics="rmse", early_stopping_rounds=5,
                        as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)), columns=["max_depth", "best_rmse"]))


# Create the parameter dictionary
params = {"objective": "reg:linear", "max_depth": 3}

# Create list of hyperparameter values
colsample_bytree_vals = [0.1, 0.5, 0.8, 1]
best_rmse = []

# Systematically vary the hyperparameter value
for curr_val in colsample_bytree_vals:

    params["colsample_bytree"] = curr_val

    # Perform cross-validation
    cv_results = xgb.cv(dtrain=X_dmatrix, params=params, nfold=2,
                        num_boost_round=5, early_stopping_rounds=10,
                        metrics="rmse", as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree", "best_rmse"]))


# tuning hyperparameters with grid search and random search
gbm_param_grid = {'learning_rate': [0.01, 0.1, 0.5, 0.9],
                  'n_estimators': [200],
                  'subsample': [0.3, 0.5, 0.9]
                  }

gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid,
                        scoring='neg_mean_squared_error', cv=4, verbose=1)

grid_mse.fit(X, y)

print("Best parameters found: ", grid_mse.best_params_)


gbm_param_grid = {'learning_rate': np.arange(0.05, 1.05, 0.05),
                  'n_estimators': [200],
                  'subsample': np.arange(0.05, 1.05, 0.05)
                  }

gbm = xgb.XGBRegressor()
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid,
                                    n_iter=25, scoring='neg_mean_squared_error', cv=4, verbose=1)

randomized_mse.fit(X, y)

print("Best parameters found: ", randomized_mse.best_params_)
