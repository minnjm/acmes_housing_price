# -*- coding: utf-8 -*-
"""
Created on Dec 2017

@author: mzhao

Using pipeline and sklearn xgboost to train model and predict Ames, Iowa housing price

https://www.kaggle.com/c/house-prices-advanced-regression-techniques

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
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
from custom_transformers import Log1pTransformer, ZeroFillTransformer
from custom_transformers import FeaturesDropper, ValueFillTransformer

F_CV = True
F_Sklearner = True
random.seed(23)

df = pd.read_csv("../data/train.csv", index_col=0)
train = df.iloc[:, :-1]
label = df.iloc[:, -1]

test = pd.read_csv("../data/test.csv", index_col=0)

print(train.shape, test.shape)
display(train.head())
print(train.info())
print(test.info())

rows_train = train.shape[0]
df = pd.concat([train, test])

# token_counts = pd.value_counts(df['BsmtFinType1'].values, sort=True)
# token_counts.plot.barh()
# plt.show()

# numeric features should be categorical
n2c_feature = ['MSSubClass']

df[n2c_feature] = df[n2c_feature].astype(str)

# remove highly correlated features
# TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF
# remove features with large percentage missing values

# drop features due to low correlation with house prices
removal_feature = ['OverallCond', 'BsmtFinSF2',
                   'LowQualFinSF', 'BsmtHalfBath', '3SsnPorch', 'PoolArea',
                   'MiscVal', 'MoSold', 'YrSold']
# # removal_feature = ['TotalBsmtSF', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']

valcolumns = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'FireplaceQu', 'GarageQual', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
# valfilltransformer = ValueFillTransformer(valcolumns, "MISSING")
# df = valfilltransformer.fit_transform(df)

# token_counts = pd.value_counts(df['Alley'].values, sort=True)
# token_counts.plot.barh()
# plt.show()

# features_slim = df.columns.tolist()
# print(features_slim)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == 'object')

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()
categorical_columns = [x for x in categorical_columns if x not in removal_feature]

non_categorical_columns = df.columns[~categorical_mask].tolist()
non_categorical_columns = [x for x in non_categorical_columns if x not in removal_feature]


numeric_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']


categorical_feature = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# let machine learning to choose right imputing strategy
numeric_imputer_p = DFFeatureUnion([
    ('imp_mean', DFImputer(strategy='mean')),
    ('imp_median', DFImputer(strategy='median')),
    ('imp_mode', DFImputer(strategy='most_frequent'))
])


numeric_featurizer = Pipeline([
    ('numeric_extractor', ColumnExtractor(non_categorical_columns)),
    # ('numeric_imputer', numeric_imputer_p),
    # ('numeric_imputer', ZeroFillTransformer())
    ('numeric_imputer', DFImputer('median'))
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
    ('valfilltransformer', ValueFillTransformer(valcolumns, "MISSING")),
    ('feature_drop', FeaturesDropper(removal_feature)),
    ('feature_union', features)
    # ('regr', regressor)
])

print(df.info())

df = pipeline.fit_transform(df)

print(type(df))
train = df.iloc[:rows_train, :]
test = df.iloc[rows_train:, :]
# train = pipeline.fit_transform(train)
# test = pipeline.transform(test)

# df['Alley'].info()
# token_counts = pd.value_counts(df['Alley'].values, sort=True)
# token_counts.plot.barh()
# plt.show()

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
          'max_depth': 3,
          'colsample_bytree': 0.2,
          'subsample': 0.8,
          'learning_rate': 0.1,
          'gamma': 0.08,
          'reg_alpha': 0.08,
          'reg_lambda': 2,
          'min_child_weight': 1
          }


if F_CV:
    if F_Sklearner:  # XGBoost Sklearn API
        gbm_param_grid = {
            'n_estimators': [1200],
            'max_depth': [3, 4, 5],
            'colsample_bytree': [0.1, 0.2, 0.3],
            'subsample': [0.6, 0.8, 0.9],
            'learning_rate': [0.08],
            'gamma': [0.06, 0.08, 0.1],
            'reg_alpha': [0.06, 0.08, 0.1],
            'reg_lambda': [1, 2],
            'min_child_weight': [1, 2]
        }

        grid_mse = GridSearchCV(param_grid=gbm_param_grid, estimator=regressor,
                                scoring='neg_mean_squared_error', cv=5, verbose=1)

        grid_mse.fit(train, label)

        # Print the optimal parameters and best score
        print("Tuned XGBRegressor grid score: {}".format(grid_mse.grid_scores_))
        print("Tuned XGBRegressor best parameters: {}".format(grid_mse.best_params_))
        print("Tuned XGBRegressor best score: {}".format(grid_mse.best_score_))

        print("Best RMSE: {}".format(np.sqrt(abs(grid_mse.best_score_))))

    else:  # XGBoost Learning API
        pass

        dmatrix = xgb.DMatrix(data=train, label=label)

        cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=10, num_boost_round=1200, metrics="rmse", early_stopping_rounds=50, as_pandas=True, seed=134)

        print(cv_results)

else:

    X_train, X_test, y_train, y_test = train_test_split(train, label,
                                                        test_size=0.20, random_state=123)

    df_2preds = test
    y_test.to_csv('testset_acme_house_price.csv')

    # X_train = train
    # y_train = label

    if F_Sklearner:  # Sklearn API

        regressor.fit(X_train, y_train)
        preds = regressor.predict(df_2preds)

    else:  # Learner API
        dmatrix = xgb.DMatrix(data=train, label=label)

        xg_reg = xgb.train(dtrain=dmatrix, params=params, num_boost_round=700)

        # xgb.plot_importance(booster=xg_reg, importance_type='gain')
        # plt.show()

        max = 30
        xgb.plot_importance(dict(sorted(xg_reg.get_fscore().items(), reverse=True, key=lambda x: x[1])[:max]), height=0.8)
        plt.show()

        preds = xg_reg.predict(xgb.DMatrix(data=df_2preds))

    # preds = np.exp(preds) - 1
    # y_test = np.exp(y_test) - 1

    # # rmse = np.sqrt(mean_squared_error(np.log(y_test), np.log(preds)))
    # rmse = np.sqrt(mean_squared_error(y_test, preds))

    # print("housing price prediction (using sklearn API) RMSE (log): %f" % (rmse))

    # df_preds = pd.DataFrame(data=preds, index=df_2preds.index, columns=['SalePrice'])

    # df_preds.to_csv('submission_acme_house_price.csv')
