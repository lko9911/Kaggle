import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def convert_to_binary(target):
    threshold = target.mean()
    return (target > threshold).astype(int)

def evaluate_models(X, y):
    y_binary = convert_to_binary(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    models = [
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree Classifier', DecisionTreeClassifier()),
        ('Random Forest Classifier', RandomForestClassifier()),
        ('Support Vector Classifier', SVC()),
        ('K-Nearest Neighbors Classifier', KNeighborsClassifier()),
        ('XGBoost Classifier', XGBClassifier()),
        ('LightGBM Classifier', LGBMClassifier())
    ]

    pipelines = {name: Pipeline([('scaler', StandardScaler()), (name, model)]) for name, model in models}

    results = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {'Accuracy': accuracy}
    
    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)

    return results_df
