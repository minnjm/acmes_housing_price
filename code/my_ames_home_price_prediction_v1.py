# -*- coding: utf-8 -*-
"""
Created on Dec 2017

@author: mzhao

Using pipeline and sklearn xgboost to train model and predict Ames, Iowa housing price

https://www.kaggle.com/c/house-prices-advanced-regression-techniques

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import sys
sys.path.append('../../libs')
from custom_transformers import Dictifier
from custom_transformers import DFFeatureUnion
from custom_transformers import DummyTransformer, ColumnExtractor
from custom_transformers import DFImputer, DFAlphaImputer

F_CV = False
F_Sklearner = False

df = pd.read_csv("../data/train.csv", index_col=0)
train = df.iloc[:, :-1]
y = df.iloc[:, -1]

test = pd.read_csv("../data/test.csv", index_col=0)

print(train.shape, test.shape)
print(train.head())
print(train.info())

# print(train.describe())

# print(train.dtypes=='object')

# Create a boolean mask for categorical columns
categorical_mask = (train.dtypes == 'object')

# Get list of categorical column names
categorical_columns = train.columns[categorical_mask].tolist()

non_categorical_columns = train.columns[~categorical_mask].tolist()


numeric_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']


categorical_feature = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

numeric_featurizer = Pipeline([
    ('numeric_extractor', ColumnExtractor(non_categorical_columns)),
    ('numeric_imputer', DFImputer(strategy="median"))
])

categorical_featurizer = Pipeline([
    ('categorical_extractor', ColumnExtractor(categorical_columns)),
    ('categorical_imputer', DFAlphaImputer()),
    ('vectorizer', DummyTransformer())
])


features = DFFeatureUnion([
    ('numeric_fe', numeric_featurizer),
    ('categorical_fe', categorical_featurizer)
])

pipeline = Pipeline([
    ('feature_union', features)
    # ('regr', regressor)
])

train = pipeline.fit_transform(train)
test = pipeline.transform(test)

print(train.shape, test.shape)

# prepare for booster
# for Sklearn API
regressor = xgb.XGBRegressor(objective='reg:linear', seed=123,
                             max_depth=4,
                             colsample_bytree=0.1,
                             subsample=0.9,
                             learning_rate=0.1,
                             gamma=0.1,
                             reg_alpha=0.1,
                             reg_lambda=1,
                             min_child_weight=1
                             )

# for Learning API
params = {'booster': 'gbtree', 'objective': 'reg:linear',
          'max_depth': 4,
          'colsample_bytree': 0.1,
          'subsample': 0.9,
          'learning_rate': 0.1,
          'gamma': 0.1,
          'reg_alpha': 0.1,
          'reg_lambda': 1,
          'min_child_weight': 1
          }


if F_CV:
    if F_Sklearner:  # XGBoost Sklearn API
        gbm_param_grid = {
            'n_estimators': [700],
            'max_depth': [4],
            'colsample_bytree': [0.1],
            'subsample': [0.9],
            'learning_rate': [0.1],
            'gamma': [0.1],
            'reg_alpha': [0.1],
            'reg_lambda': [1],
            'min_child_weight': [1]
        }

        # regressor = xgb.XGBRegressor(objective='reg:linear', seed=123)

        grid_mse = GridSearchCV(param_grid=gbm_param_grid, estimator=regressor,
                                scoring="neg_mean_squared_error", cv=10, verbose=1)

        grid_mse.fit(train, y)

        # Print the optimal parameters and best score
        print("Tuned XGBRegressor grid score: {}".format(grid_mse.grid_scores_))
        # print("Tuned XGBRegressor grid results: ")
        # print(pd.DataFrame(grid_mse.cv_results_))
        print("Tuned XGBRegressor best parameters: {}".format(grid_mse.best_params_))
        print("Tuned XGBRegressor best score: {}".format(grid_mse.best_score_))

        print("Best RMSE: {}".format(np.sqrt(abs(grid_mse.best_score_))))

    else:  # XGBoost Learning API

        dmatrix = xgb.DMatrix(data=train, label=y)

        # # tuning space
        # space = [0.88, 0.9, 0.92]

        # final_rmse_per_round = []

        # min_rmse = 0
        # min_hyper = 0

        # for curr_val in space:

        #     params['subsample'] = curr_val

        #     cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=10, num_boost_round=500, metrics="rmse", early_stopping_rounds=20, as_pandas=True, seed=123)

        #     if min_rmse == 0 or min_rmse > cv_results['test-rmse-mean'].values[-1]:
        #         min_rmse = cv_results['test-rmse-mean'].values[-1]
        #         min_hyper = curr_val

        # print(min_hyper, min_rmse)

        # final_rmse_per_round = []

        cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=10, num_boost_round=700, metrics="rmse", early_stopping_rounds=20, as_pandas=True, seed=125)

        print(cv_results)

        #     final_rmse_per_round.append(cv_results['test-rmse-mean'].values[-1])

        # number_round_rmse = list(zip(num_rounds, final_rmse_per_round))

        # print(pd.DataFrame(number_round_rmse, columns=['number of boosting rounds', 'rmse']))

else:

    X_train, X_test, y_train, y_test = train_test_split(train, y,
                                                        test_size=0.20, random_state=134)

    df_2preds = X_test

    if F_Sklearner:  # Sklearn API

        regressor.fit(X_train, y_train)
        preds = regressor.predict(df_2preds)

    else:  # Learner API
        dmatrix = xgb.DMatrix(data=X_train, label=y_train)

        booster = xgb.train(dtrain=dmatrix, params=params, num_boost_round=700)

        preds = booster.predict(xgb.DMatrix(data=df_2preds))

    # rmse = np.sqrt(mean_squared_error(np.log(y_test), np.log(preds)))
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("housing price prediction (using sklearn API) RMSE (log): %f" % (rmse))

    df_preds = pd.DataFrame(data=preds, index=df_2preds.index, columns=['SalePrice'])

    print(df_2preds.head())
    print(df_preds.head())

    # df_preds.to_csv('submission_acme_house_price.csv')


# print(type(X))
# print(X.shape)
# print(X.isnull().sum())

# # Use DictVectorizer since it can handle feature values no in a sample (mapping) gracefully.
# # Convert train into a dictionary: train_dict
# train_dict = train.to_dict('records')

# # Create the DictVectorizer object: dv
# dv = DictVectorizer(sparse=False)

# # Apply dv on df: df_encoded
# train_encoded = dv.fit_transform(train_dict)

# # Print the resulting first five rows
# print(type(train_encoded))
# print(train_encoded.shape)
# print(train_encoded[:5, :])

# Print the vocabulary
# print(dv.vocabulary_)

# # build a pipeline
# # Setup the pipeline steps: steps
# X = train

# print(X.isnull().sum())

# steps = [("ohe_onestep", DictVectorizer(sparse=False)),
#          ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]

# # Create the pipeline: xgb_pipeline
# xgb_pipeline = Pipeline(steps=steps)

# # Fit the pipeline
# # xgb_pipeline.fit(X.to_dict('records'), y)

# # Cross-validate the model
# cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict('records'), y,
#                                    scoring='neg_mean_squared_error', cv=10)

# # Print the 10-fold RMSE
# print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))

# # Apply numeric imputer
# numeric_imputation_mapper = DataFrameMapper(
#     [([numeric_feature], Imputer(strategy="median")) for numeric_feature in non_categorical_columns],
#     input_df=True,
#     df_out=True
# )

# # Apply categorical imputer
# categorical_imputation_mapper = DataFrameMapper(
#     [([category_feature], CategoricalImputer()) for category_feature in categorical_columns],
#     input_df=True,
#     df_out=True
# )

# # Combine the numeric and categorical transformations
# numeric_categorical_union = FeatureUnion([
#     ("num_mapper", numeric_imputation_mapper),
#     ("cat_mapper", categorical_imputation_mapper)
# ])

# # Create full pipeline
# pipeline = Pipeline([
#     ("featureunion", numeric_categorical_union),
#     ("dictifier", Dictifier()),
#     ("vectorizer", DictVectorizer(sort=False)),
#     ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))
# ])

# gbm_param_grid = {
#     'xgb_model__subsample': np.arange(0.05, 1, 0.05),
#     'xgb_model__max_depth': np.arange(3, 20, 1),
#     'xgb_model__colsample_bytree': np.arange(0.1, 1.05, 0.05)
# }

# randomized_net_mse = RandomizedSearchCV(estimator=pipeline,
#                                         param_distributions=gbm_param_grid,
#                                         n_iter=10,
#                                         scoring='neg_mean_squared_error',
#                                         cv=4)

# randomized_net_mse.fit(X, y)

# print("Best rmse: ", np.sqrt(np.abs(randomized_net_mse.best_score_)))

# print("Best model: ", randomized_neg_mse.best_estimator_)

# cross_val_scores = cross_val_score(pipeline, X, y,
#                                    scoring='neg_mean_squared_error', cv=10)

# # Print the 10-fold RMSE
# print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))


# # Perform cross-validation with early stopping: cv_results
# cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=50, metrics="rmse", early_stopping_rounds=10, as_pandas=True, seed=123)

# # Print cv_results
# print(cv_results)
