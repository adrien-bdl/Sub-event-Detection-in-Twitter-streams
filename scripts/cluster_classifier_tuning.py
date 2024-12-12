"""
We do hyperparameter tuning on our cluster-classifier model. We begin by a grid search
with cross validation on the clustering part, and then a model selection for the classifier part
"""

#%% Imports
import pandas as pd
import numpy as np
from src.ClusterClassifier import ClusterEstimator
from src import utils
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

pd.options.display.float_format = '{:.2f}'.format  # Limit to 2 decimal places


# %% Read the data

if __name__ == "__main__":

    path_dfs = "challenge_data/train_BERT"
    df_BERT = utils.read_BERT(path_dfs=path_dfs).drop(columns="Timestamp")
    X, y = utils.df_to_tensors(df_BERT)

    # %% Parameter tuning for clustering

    n_sample = 500 # number of periods to select
    sample = np.sort(np.random.choice(len(X), replace=False, size=n_sample))
    X_sample = X[sample]
    y_sample = y[sample]


    param_grid_clus = {
        "n_clusters": [1, 3, 5],
        "min_cluster_size": [40, 50, 70],
        "cluster_selection_epsilon": [0, 0.025, 0.05, 0.075],
        "cluster_selection_method": ["eom", "leaf"],
        "alpha": [0.75, 1.0, 1.25, 1.5],
    }

    cluster_clf = ClusterEstimator()
    grid_clf = GridSearchCV(cluster_clf, param_grid_clus, n_jobs=-1)
    search = grid_clf.fit(X_sample, y_sample)
    results = results = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score")
    results.to_csv("../../cv_results/cv_results.csv")

    # %% Best clustering parameters from previous grid searchs

    best_cluster_params = { 
        "n_clusters": 1,
        "min_cluster_size": 40,
        "cluster_selection_epsilon": 0.07,
        "cluster_selection_method": "eom",
        "alpha": 1.0
    }

    # %% Parameter tuning for classification

    candidate_clfs = [
        (LogisticRegression(), {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'solver': ['liblinear', 'saga'],  # Optimizer to use
        'penalty': ['l2'],  # Regularization method
        'max_iter': [500, 1000]  # Max iterations for convergence
        }),

        (SVC(), {
        'C': [0.01, 0.1, 1, 10],  # Regularization strength
        'kernel': ['linear', 'rbf'],  # Kernel type
        'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf'
        'class_weight': [None, 'balanced']  # Adjust weights for imbalanced classes
        }),

        (RandomForestClassifier(), {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'criterion': ['gini', 'entropy'],  # Split quality measure
        'max_depth': [None, 5, 10, 20],  # Max depth of tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split
        'min_samples_leaf': [1, 2, 4],  # Minimum samples required at leaf node
        }),

        (GradientBoostingClassifier(), {
        'n_estimators': [50, 100, 150],  # Number of boosting stages
        'learning_rate': [0.001, 0.01, 0.1, 0.5],  # Step size at each iteration
        'max_depth': [3, 4, 5],  # Maximum depth of individual trees
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split
        'min_samples_leaf': [1, 2, 4],  # Minimum samples required at leaf node
        'subsample': [0.8, 0.9, 1.0]  # Fraction of samples used for fitting
        }),

        (MLPClassifier(), {
        'hidden_layer_sizes': [(50,), (100,), (100, 100), (200,)],  # Size of hidden layers
        'activation': ['relu', 'tanh', 'logistic'],  # Activation function
        'solver': ['adam', 'sgd', 'lbfgs'],  # Optimizer
        'alpha': [0.0001, 0.001, 0.01],  # L2 regularization term
        'learning_rate': ['constant', 'invscaling', 'adaptive'],  # Learning rate schedule
        'max_iter': [200, 500, 1000]
        }),

        (KNeighborsClassifier(metric="cosine"), {
        'n_neighbors': [3, 5, 7, 10, 15],  # Number of neighbors
        'weights': ['uniform', 'distance'],  # Weight function used in prediction
        }),

        (GaussianNB(), {
        'var_smoothing': [1e-9, 1e-8, 1e-7]  # Smoothing parameter to avoid division by zero
        })
    ]

    cluster_clf = ClusterEstimator(**best_cluster_params)
    X_clustered = cluster_clf.predict(X, return_X_clustered=True)
    
    for clf, param_grid in candidate_clfs:
        print(f"--- {type(clf)} ---")
        grid_clf = RandomizedSearchCV(clf, param_grid, n_jobs=-1, n_iter=20)
        search = grid_clf.fit(X_clustered, y)
        results = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score")
        results.to_csv(f"cv_results/cv_results_{type(clf)}.csv")

    # %% Best classifier parameters

    best_classifier_params = {'solver': 'adam', 'max_iter': 1000, 'learning_rate': 'adaptive',
                               'hidden_layer_sizes': (100,), 'alpha': 0.01, 'activation': 'tanh'}
    best_clf = MLPClassifier(**best_classifier_params)

    # %% Compute accuracy on train/test split

    for i in range(3):

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        cluster_clf = ClusterEstimator(**best_cluster_params, clf=best_clf)
        cluster_clf.fit(X_train, y_train)
        print(f"Accuracy score - split {i + 1} : {cluster_clf.score(X_test, y_test):.3f}")

# %%
