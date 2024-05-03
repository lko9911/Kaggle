import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def show_models(trained_models):
    return trained_models

def evaluate_models_all(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('Linear Regression', LinearRegression()),
        ('Ridge Regression', Ridge()),
        ('Lasso Regression', Lasso()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree Regressor', DecisionTreeRegressor()),
        ('Random Forest Regressor', RandomForestRegressor()),
        ('Support Vector Machine Regressor', SVR()),
        ('K-Nearest Neighbors Regressor', KNeighborsRegressor()),
        ('Gradient Boosting Regressor', GradientBoostingRegressor()),
        ('XGBoost Regressor', XGBRegressor()),
        ('LightGBM Regressor', LGBMRegressor())
    ]

    pipelines = {name: Pipeline([('scaler', StandardScaler()), (name, model)]) for name, model in models}

    results = {}
    trained_models = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R²': r2}
        trained_models[name] = pipeline.named_steps[name]

    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)

    return results_df

def evaluate_models_LRLE(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('Linear Regression', LinearRegression()),
        ('Ridge Regression', Ridge()),
        ('Lasso Regression', Lasso()),
        ('ElasticNet', ElasticNet()),
    ]

    pipelines = {name: Pipeline([('scaler', StandardScaler()), (name, model)]) for name, model in models}

    results = {}
    trained_models = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R²': r2}
        trained_models[name] = pipeline.named_steps[name]

    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)

    return results_df

def evaluate_models_RGXL(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('Random Forest Regressor', RandomForestRegressor()),
        ('Gradient Boosting Regressor', GradientBoostingRegressor()),
        ('XGBoost Regressor', XGBRegressor()),
        ('LightGBM Regressor', LGBMRegressor())
    ]

    pipelines = {name: Pipeline([('scaler', StandardScaler()), (name, model)]) for name, model in models}

    results = {}
    trained_models = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R²': r2}
        trained_models[name] = pipeline.named_steps[name]

    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)

    return results_df

def evaluate_models_X(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('XGBoost Regressor', XGBRegressor())
    ]

    pipelines = {name: Pipeline([('scaler', StandardScaler()), (name, model)]) for name, model in models}

    results = {}
    trained_models = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R²': r2}
        trained_models[name] = pipeline.named_steps[name]

    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)

    return results_df

def evaluate_models_L(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('LightGBM Regressor', LGBMRegressor())
    ]

    pipelines = {name: Pipeline([('scaler', StandardScaler()), (name, model)]) for name, model in models}

    results = {}
    trained_models = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R²': r2}
        trained_models[name] = pipeline.named_steps[name]

    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)

    return results_df

def evaluate_models_R(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('Random Forest Regressor', RandomForestRegressor())
    ]

    pipelines = {name: Pipeline([('scaler', StandardScaler()), (name, model)]) for name, model in models}

    results = {}
    trained_models = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R²': r2}
        trained_models[name] = pipeline.named_steps[name]

    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)

    return results_df

def evaluate_models_G(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('Gradient Boosting Regressor', GradientBoostingRegressor())
    ]

    pipelines = {name: Pipeline([('scaler', StandardScaler()), (name, model)]) for name, model in models}

    results = {}
    trained_models = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R²': r2}
        trained_models[name] = pipeline.named_steps[name]

    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)

    return results_df

