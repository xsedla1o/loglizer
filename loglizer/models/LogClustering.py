"""
The implementation of Log Clustering model for anomaly detection.

Authors:
    LogPAI Team

Reference:
    [1] Qingwei Lin, Hongyu Zhang, Jian-Guang Lou, Yu Zhang, Xuewei Chen. Log Clustering
        based Problem Identification for Online Service Systems. International Conference
        on Software Engineering (ICSE), 2016.

"""

import numpy as np
from numpy import linalg as LA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cdist

from ..utils import metrics


class LogClustering(object):
    def __init__(
        self,
        max_dist=0.3,
        anomaly_threshold=0.3,
        mode="online",
        num_bootstrap_samples=1000,
    ):
        """
        Attributes
        ----------
            max_dist: float, the threshold to stop the clustering process
            anomaly_threshold: float, the threshold for anomaly detection
            mode: str, 'offline' or 'online' mode for clustering
            num_bootstrap_samples: int, online clustering starts with a bootstraping process, which
                determines the initial cluster representatives offline using a subset of samples
            representatives: ndarray, the representative samples of clusters, of shape
                num_clusters-by-num_events
            cluster_size_dict: dict, the size of each cluster, used to update representatives online
        """
        self.max_dist = max_dist
        self.anomaly_threshold = anomaly_threshold
        self.mode = mode
        self.num_bootstrap_samples = num_bootstrap_samples
        self.representatives = list()
        self.np_reps = None
        self.cluster_size_dict = dict()

    def fit(self, X):
        print("====== Model summary ======")
        if self.mode == "offline":
            # The offline mode can process about 10K samples only due to huge memory consumption.
            self._offline_clustering(X)
        elif self.mode == "online":
            # Bootstrapping phase
            if self.num_bootstrap_samples > 0:
                X_bootstrap = X[0 : self.num_bootstrap_samples, :]
                self._offline_clustering(X_bootstrap)
            # Online learning phase
            if X.shape[0] > self.num_bootstrap_samples:
                self._online_clustering(X)

    def predict(self, X, batch_size=1024):
        y_pred = []
        for i in range(0, X.shape[0], batch_size):
            batch = X[i : i + batch_size]
            distances = cdist(batch, self.np_reps, metric="cosine")
            min_distances = np.min(distances, axis=1)
            y_pred.append((min_distances > self.anomaly_threshold).astype(int))
        return np.concatenate(y_pred)

    def evaluate(self, X, y_true):
        print("====== Evaluation summary ======")
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print(
            "Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n".format(
                precision, recall, f1
            )
        )
        return precision, recall, f1

    def _offline_clustering(self, X):
        print("Starting offline clustering...")
        p_dist = pdist(X, metric="cosine")
        Z = linkage(p_dist, "complete")
        cluster_index = fcluster(Z, self.max_dist, criterion="distance")
        self._extract_representatives(X, cluster_index)
        self.np_reps = np.asarray(self.representatives, dtype=np.float64)
        print("Processed {} instances.".format(X.shape[0]))
        print("Found {} clusters offline.\n".format(len(self.representatives)))
        # print('The representive vectors are:')
        # pprint.pprint(self.representatives.tolist())

    def _extract_representatives(self, X, cluster_index):
        num_clusters = len(set(cluster_index))
        for clu in range(num_clusters):
            clu_idx = np.argwhere(cluster_index == clu + 1)[:, 0]
            self.cluster_size_dict[clu] = clu_idx.shape[0]
            repre_center = np.average(X[clu_idx, :], axis=0)
            self.representatives.append(repre_center)

    def _online_clustering(self, X):
        print("Starting online clustering...")
        self.np_reps = np.asarray(self.representatives, dtype=np.float64)
        for i in range(self.num_bootstrap_samples, X.shape[0]):
            instance_vec = X[i, :].reshape(1, -1)
            if self.np_reps.shape[0] > 0:
                dists = cdist(self.np_reps, instance_vec, metric="cosine").flatten()
                clu_id = np.argmin(dists)
                min_dist = dists[clu_id]

                if min_dist <= self.max_dist:
                    new_size = self.cluster_size_dict[clu_id] + 1
                    self.cluster_size_dict[clu_id] = new_size
                    self.np_reps[clu_id] += (
                        instance_vec.flatten() - self.np_reps[clu_id]
                    ) / new_size
                    continue

            new_id = self.np_reps.shape[0]
            self.cluster_size_dict[new_id] = 1
            self.np_reps = np.vstack([self.np_reps, instance_vec])

        print("Processed {} instances.".format(X.shape[0]))
        print("Found {} clusters online.\n".format(self.np_reps.shape[0]))

    def _distance_metric(self, x1, x2):
        norm = LA.norm(x1) * LA.norm(x2)
        distance = 1 - np.dot(x1, x2) / (norm + 1e-8)
        if distance < 1e-8:
            distance = 0
        return distance

    def _get_min_cluster_dist(self, instance_vec):
        min_index = -1
        min_dist = float("inf")
        for i in range(len(self.representatives)):
            cluster_rep = self.representatives[i]
            dist = self._distance_metric(instance_vec, cluster_rep)
            if dist < 1e-8:
                min_dist = 0
                min_index = i
                break
            elif dist < min_dist:
                min_dist = dist
                min_index = i
        return min_dist, min_index

    def _get_min_cluster_dist_vec(self, instance_vec):
        cluster_reps = self.np_reps
        norms = LA.norm(cluster_reps, axis=1) * LA.norm(instance_vec)
        distances = 1 - np.dot(cluster_reps, instance_vec) / (norms + 1e-8)
        distances = np.where(distances < 1e-8, 0, distances)
        min_index = np.argmin(distances)
        min_dist = distances[min_index]
        return min_dist, min_index
