"""
Definition of our cluster-classifier model following the sklearn framework 
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn import cluster
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.metrics import accuracy_score


def get_cluster_means(X, cl, n_clusters=1):
    """
    X has shape (N_periods, N_tweets_per_period, N_features). For each period, we apply
    a clustering on the corresponding tweets and average the embedding over the n_clusters main
    clusters.
    """

    n_features = X.shape[2]
    X_reduced = []

    for i in range(len(X)):

        labels = cl.fit_predict(X[i])
        counts = Counter(labels[labels != -1])

        if len(counts) == 0:
            X_reduced.append(X[i].mean(axis=0))

        else:

            mean = np.zeros(shape=(n_features,))
            for label, count in counts.most_common(n_clusters):
                mean += cl.centroids_[label]
            
            X_reduced.append(mean / n_clusters)

    return np.stack(X_reduced)


class ClusterEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, n_clusters=1, min_cluster_size=5, cluster_selection_epsilon=0.0,
                 cluster_selection_method="eom", alpha=1.0, clf=None):
        
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.alpha = alpha 

        if clf:
            self.clf = clf
        else:
            self.clf = LogisticRegression(max_iter=1000) 

    def fit(self, X, y):

        clus = cluster.HDBSCAN(metric="cosine", store_centers="centroid", min_cluster_size=self.min_cluster_size,
                                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                                    cluster_selection_method=self.cluster_selection_method, alpha=self.alpha)

        X = get_cluster_means(X, clus, n_clusters=self.n_clusters)

        self.clf.fit(X, y)

    def predict(self, X, return_X_clustered=False):

        clus = cluster.HDBSCAN(metric="cosine", store_centers="centroid", min_cluster_size=self.min_cluster_size,
                            cluster_selection_epsilon=self.cluster_selection_epsilon,
                            cluster_selection_method=self.cluster_selection_method, alpha=self.alpha)

        X = get_cluster_means(X, clus, n_clusters=self.n_clusters)

        if return_X_clustered:
            return X
        
        return self.clf.predict(X)
    
    def score(self, X, y):
        
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)