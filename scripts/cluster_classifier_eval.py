"""
We run our cluster-classifier model on the evaluation dataset.
"""
#%% Imports
import pandas as pd
from src.ClusterClassifier import ClusterEstimator
from src import utils
from sklearn.neural_network import MLPClassifier

pd.options.display.float_format = '{:.2f}'.format  # Limit to 2 decimal places
# %% Read the data

if __name__ == "__main__":

    path_dfs = "challenge_data/train_BERT"
    df_BERT = utils.read_BERT(path_dfs=path_dfs).drop(columns="Timestamp")
    X, y = utils.df_to_tensors(df_BERT)

    # %% kaggle submission

    path_eval = "challenge_data/eval_BERT"
    path_write = "challenge_data/submissions/cluster_clf.csv"

    clf = MLPClassifier(**{'solver': 'adam', 'max_iter': 1000, 'learning_rate': 'adaptive',
                        'hidden_layer_sizes': (100,), 'alpha': 0.01, 'activation': 'tanh'})

    cluster_clf = ClusterEstimator(n_clusters=1, min_cluster_size=40, cluster_selection_epsilon=0.07,
                                cluster_selection_method="eom", alpha=1.0, clf=clf)

    cluster_clf.fit(X, y)

    df_eval = utils.read_BERT(path_dfs=path_eval).drop(columns="Timestamp")
    X_eval, ids = utils.df_to_tensors(df_eval, eval=True)
    y_eval = cluster_clf.predict(X_eval)
    df_kaggle = pd.DataFrame(data= {"ID": ids, "EventType": y_eval})
    df_kaggle.to_csv(path_write, index=False)

