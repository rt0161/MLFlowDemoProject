import os,sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

import mlflow
import mlflow.sklearn
import mlflow.xgboost, mlflow.xgboost

# load in file from data path
base_path=Path().resolve().parents[0]
data_path = os.path.join(base_path,'02-data/')
dataname = 'prep_car_input.csv'

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def get_train_test_split(df, target_col, test_size=0.3):
    """this function gets the train/test split of incoming data"""
    all_col = df.columns.values
    idx_of_target= np.where(all_col==target_col)
    X = df[df.columns.difference([target_col])]
    y = df.iloc[:,idx_of_target[0]]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state = 44)

    return X_train, X_test, y_train, y_test

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def detect_arg_dtype(arg_in):
    if arg_in.isdigit():
        return int(arg_in)
    elif isfloat(arg_in):
        return float(arg_in)
    elif arg_in in ['True','False']:
        return bool(arg_in)
    else:
        return str(arg_in)

def get_model(arg_in):
    arg_len = len(arg_in)
    if arg_len>1:
        alg_method = str(arg_in[1])
    else:
        print("no method asserted, please try again.")
        return False
    if alg_method == 'logistic_regression':
        penalty = str(arg_in[2]) if arg_len > 2 else 'l2'
        fit_intercept = bool(arg_in[3]) if arg_len > 3 else True
        max_iter = int(arg_in[4]) if arg_len > 4 else 100
        model = LogisticRegression(penalty=penalty, fit_intercept=fit_intercept, random_state = 44)
        params = {'penalty': penalty,'fit_intercept': fit_intercept, 'max_iter': max_iter}
    elif alg_method == 'randomforest':
        n_estimators = detect_arg_dtype(arg_in[2]) if arg_len > 2 else 10
        max_features = detect_arg_dtype(arg_in[3]) if arg_len > 3 else 'auto'
        max_depth = detect_arg_dtype(arg_in[4]) if arg_len> 4 else None
        max_leaf_nodes = detect_arg_dtype(arg_in[5]) if arg_len> 5 else None
        model = RandomForestRegressor(n_estimators=n_estimators,max_features=max_features, max_depth = max_depth, max_leaf_nodes= max_leaf_nodes, random_state = 44)
        params = {'n_estimators': n_estimators,'max_features': max_features,'max_depth': max_depth,'max_leaf_nodes': max_leaf_nodes}
    elif alg_method == 'xgb':
        learning_rate = detect_arg_dtype(arg_in[2]) if arg_len > 2 else 0.6
        max_depth = detect_arg_dtype(arg_in[3]) if arg_len > 3 else 5
        n_estimators = detect_arg_dtype(arg_in[4]) if arg_len > 4 else 10
        subsample = detect_arg_dtype(arg_in[5]) if arg_len > 5 else 0.6
        colsample_bytree = detect_arg_dtype(arg_in[6]) if arg_len > 6 else 0.4
        objective = 'reg:squarederror'
        model = xgb.XGBRegressor(objective =objective, colsample_bytree = colsample_bytree, learning_rate = learning_rate,
                max_depth = max_depth, n_estimators = n_estimators, subsample = subsample, random_state = 44)
        params = {'n_estimators':n_estimators,'learning_rate':learning_rate,'max_depth':max_depth,'subsample':subsample,'colsample_bytree': colsample_bytree}
    elif alg_method=='adaboost':
        n_estimators = detect_arg_dtype(arg_in[2]) if arg_len > 2 else 10
        learning_rate = detect_arg_dtype(arg_in[3]) if arg_len > 3 else 0.1
        loss = detect_arg_dtype(arg_in[4]) if arg_len > 4 else 'linear'
        model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, random_state = 44)
        params = {'n_estimators':n_estimators,'learning_rate':learning_rate,'loss': loss}
    return model, params


if __name__ == "__main__":

    # load data
    df_carmod = pd.read_csv(data_path+dataname, index_col=None)

    # get training method from argument, initiate the model for use
    model, param = get_model(sys.argv)

    print(model.__class__.__name__)
    print(model.get_params())

    # get train/test train_test_split
    if model.__class__.__name__=='XGBRegressor':
        all_col = df_carmod.columns.values
        target_col = 'price'
        idx_of_target= np.where(all_col==target_col)
        X = df_carmod[df_carmod.columns.difference([target_col])]
        y = df_carmod.iloc[:,idx_of_target[0]]
        data_dmatrix = xgb.DMatrix(data=X,label=y)
        X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.2)
    else:
        X_train, X_test, y_train, y_test=get_train_test_split(df_carmod,'price',0.2)

    # set experiment Location
    mlflow.set_experiment("model_build")
    experiment = mlflow.get_experiment_by_name("model_build")

    # start model fit within mlflow
    with mlflow.start_run() as run:
        # logging parameters
        mlflow.log_param("regressor",model.__class__.__name__)
        for prm in param.keys():
            mlflow.log_param(prm, param[prm])
        
        # start fitting
        model.fit(X_train,y_train)
        # logged the X_test data for later testing of model deployment
        X_test.to_csv('../02-data/data_model_built_xtest_set.csv', index= False)
        y_test.to_csv('../02-data/data_model_built_ytest_set.csv', index= False)
        mlflow.log_artifact('../02-data/data_model_built_xtest_set.csv')
        mlflow.log_artifact('../02-data/data_model_built_ytest_set.csv')

        # evaluate model
        y_pred = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, y_pred)
        mlflow.log_metric("root-mean-squre-error",rmse)
        mlflow.log_metric("mean-absolute-error",mae)
        mlflow.log_metric("r2",r2)

        # log models
        if model.__class__.__name__ in ['LogisticRegression','RandomForestRegressor','AdaBoostRegressor']:
            mlflow.sklearn.log_model(model, "model")
        else:
            mlflow.xgboost.log_model(model, "model")





