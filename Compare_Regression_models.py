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

def evaluate_models(X, y):
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

    return results_df, trained_models

def plot_feature_importance(model_name, model, n_features, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-n_features:]
        plt.figure(figsize=(12, 6))
        sns.barplot(y=feature_names[indices], x=importances[indices], palette="viridis")
        plt.title(f'Top {n_features} Feature Importances for {model_name}', fontsize=18)
        plt.xticks(rotation=0, fontsize=12)
        plt.xlabel('Feature', fontsize=14)
        plt.ylabel('Importance', fontsize=14)
        plt.grid(linestyle='--')
        plt.show()
    else:
        print(f"Model {model_name} does not support feature importance.")
