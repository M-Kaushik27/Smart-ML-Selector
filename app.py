from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
import threadpoolctl
threadpoolctl.threadpool_limits(limits=-1)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, silhouette_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore") 

app = Flask(__name__)

# Define models with optimal hyperparameters for classification
def get_classification_models():
    models = {
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'SVC': SVC(C=1.0, kernel='rbf', random_state=42),
        'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }
    return models

# Define models with optimal hyperparameters for regression
def get_regression_models():
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'SVR': SVR(C=1.0, kernel='rbf'),
        'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=10, random_state=42),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    return models

# Define clustering models with default hyperparameters
def get_clustering_models(n_clusters=3):
    models = {
        'KMeans': KMeans(n_clusters=n_clusters, random_state=42),
        'AgglomerativeClustering': AgglomerativeClustering(n_clusters=n_clusters),
        'MeanShift': MeanShift(),
        'DBSCAN': DBSCAN(eps=0.5)
    }
    return models

def process_data(file, task_type):
    data = pd.read_csv(file)
    print("Columns in the uploaded dataset:", data.columns.tolist())
    
    target_column = request.form.get('target_column')
    if not target_column or target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found. Available columns: {data.columns.tolist()}")
    
    X = data.drop(columns=target_column)
    y = data[target_column]

    # Encoding categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    label_encoder = LabelEncoder()
    for col in categorical_columns:
        X[col] = label_encoder.fit_transform(X[col].astype(str))
    
    if task_type == 'classification' and y.dtype == 'object':
        y = label_encoder.fit_transform(y)
    
    # Handling missing values
    X = X.apply(lambda col: col.fillna(col.mean()) if col.name in numeric_columns else col.fillna(col.mode()[0]))

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling numeric columns
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    return X_train, X_test, y_train, y_test, target_column

def perform_task(task_type, X_train, X_test, y_train, y_test, n_clusters=3):
    results = {}

    if task_type == 'classification':
        models = get_classification_models()
        for name, model in models.items():
            # Apply GridSearchCV for tuning and cross-validation
            param_grid = get_param_distributions(name)
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results[name] = {'Accuracy': accuracy, 'F1 Score': f1}
        
        optimal_model = max(results, key=lambda k: results[k]['Accuracy'])

    elif task_type == 'regression':
        models = get_regression_models()
        for name, model in models.items():
            # GridSearchCV with cross-validation for tuning
            param_grid = get_param_distributions(name)
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'MSE': mse, 'R²': r2}
        
        optimal_model = max(results, key=lambda k: results[k]['R²'])

    else:  # clustering
        models = get_clustering_models(n_clusters=n_clusters)
        silhouette_scores = {}
        for name, model in models.items():
            model.fit(X_train)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_train)
            silhouette = silhouette_score(X_train, labels) if len(set(labels)) > 1 else 0
            silhouette_scores[name] = silhouette
            results[name] = {'Silhouette Score': silhouette}
        
        optimal_model = max(silhouette_scores, key=silhouette_scores.get)

    return optimal_model, results

def get_param_distributions(model_name):
    # Define hyperparameter distributions for RandomizedSearchCV
    param_distributions = {
        # Classification Algorithms
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l2'],
            'max_iter': [100, 200, 500]
        },
        'Support Vector Machine': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto']
        },
        'Random Forest Classifier': {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10]
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'Naive Bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  # Only for GaussianNB
        },
        'Decision Tree Classifier': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'Neural Network Classifier': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 25, 10)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'learning_rate_init': [0.001, 0.01, 0.1]
        },
        'AdaBoost Classifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        },

        # Regression Algorithms
        'Linear Regression': {},
        'Polynomial Regression': {
            'degree': [2, 3, 4, 5]  # PolynomialFeatures(degree=)
        },
        'Support Vector Regression': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto']
        },
        'Random Forest Regressor': {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'Decision Tree Regressor': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting Regressor': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10]
        },
        'Neural Network Regressor': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 25, 10)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'learning_rate_init': [0.001, 0.01, 0.1]
        },

        # Clustering Algorithms
        'K-Means Clustering': {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'init': ['k-means++', 'random'],
            'n_init': [10, 20, 30]
        },
        'Hierarchical Clustering': {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'linkage': ['ward', 'complete', 'average', 'single'],
            'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
        },
        'DBSCAN': {
            'eps': [0.1, 0.5, 1, 5],
            'min_samples': [5, 10, 20],
            'metric': ['euclidean', 'manhattan', 'cosine']
        },
        'Mean Shift': {
            'bandwidth': [None, 0.5, 1, 2, 5]
        }
    }
    
    return param_distributions.get(model_name, {})

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to retrieve target columns from uploaded CSV
@app.route('/target_columns', methods=['POST'])
def target_columns():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        columns = df.select_dtypes(include=['int64', 'float64', 'object']).columns.tolist()
        return jsonify({'target_columns': columns})
    return jsonify({'target_columns': []})

# Route to process file and task
@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    task_type = request.form['task']
    n_clusters = int(request.form.get('n_clusters', 3)) if task_type == 'clustering' else 3
    X_train, X_test, y_train, y_test, target_column = process_data(file, task_type)
    optimal_model, results = perform_task(task_type, X_train, X_test, y_train, y_test, n_clusters)

    # Plot results for classification and regression tasks
    if task_type in ['classification', 'regression']:
        metric_1 = [list(res.values())[0] for res in results.values()]
        metric_2 = [list(res.values())[1] for res in results.values()]
        model_names = list(results.keys())

        plt.figure(figsize=(10, 6))
        plt.scatter(metric_1, metric_2, s=100)
        for i, model in enumerate(model_names):
            plt.text(metric_1[i], metric_2[i], model, fontsize=9, ha='right')
        
        plt.xlabel('Metric 1 (Accuracy/F1 for Classification, MSE for Regression)') 
        plt.ylabel('Metric 2 (F1 for Classification, R² for Regression)')
        plt.title(f'Comparison of Models for {task_type.capitalize()}')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', optimal_model=optimal_model, results=results, plot_url=img_base64)

    # Plot results for clustering task
    if task_type == 'clustering':
        model_names = list(results.keys())
        silhouette_scores = [res['Silhouette Score'] for res in results.values()]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=model_names, y=silhouette_scores)
        plt.xlabel('Clustering Algorithm')
        plt.ylabel('Silhouette Score')
        plt.title('Comparison of Clustering Algorithms')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', optimal_model=optimal_model, results=results, plot_url=img_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)