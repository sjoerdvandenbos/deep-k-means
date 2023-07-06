from argparse import ArgumentParser
from datetime import datetime
from json import dumps
from math import sqrt
from pathlib import Path
from typing import Tuple, List

import numpy as np
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from susi import SOMClustering

from utils import get_ground_truths, map_clusterlabels_to_groundtruth, cluster_acc, get_dataset_dir

N_CLUSTERS_CLUSTERING_ALGORITHMS = ("KMEANS", "AGGLOMERATIVE_CLUSTERING", "SOM")
MANIFOLD_CLUSTERING_ALGORITHMS = ("DBSCAN", "OPTICS")
ALL_CLUSTERING_ALGORITHMS = (*N_CLUSTERS_CLUSTERING_ALGORITHMS, *MANIFOLD_CLUSTERING_ALGORITHMS)


class Clustering:
    def __init__(self, metrics_dir, dataset_dir, embeds):
        self.metrics_dir = metrics_dir
        self.dataset_dir = dataset_dir
        self.embeds = embeds
        self.ground_truths = None

    @classmethod
    def from_metrics_dir(cls, metrics_dir):
        dataset_dir = get_dataset_dir(metrics_dir)
        embeds = cls._load_embeds(metrics_dir)
        new_instance = cls(metrics_dir, dataset_dir, embeds)
        new_instance._read_ground_truths()
        return new_instance

    def explore_clusterings_and_hyperparams(self, eps_vals, min_sample_vals):
        for clustering in N_CLUSTERS_CLUSTERING_ALGORITHMS:
            plot_data = self.evaluate_n_clusters(clustering)
            self.plot_accuracies(plot_data, clustering)

        # dbscan_accs = self.dbscan_hyperparam_search(eps_vals, min_sample_vals)
        # optics_accs = self.optics_hyperparam_search(eps_vals, min_sample_vals)
        # self.plot_hyperparam_search(dbscan_accs, eps_vals, min_sample_vals, "DBSCAN")
        # self.plot_hyperparam_search(optics_accs, eps_vals, min_sample_vals, "OPTICS")

    def plot_hyperparam_search(self, accuracy_mesh: np.ndarray,
                               eps_vals: np.ndarray, min_samples: np.ndarray, clustering_algo) -> None:
        x_len = eps_vals.shape[0]
        y_len = min_samples.shape[0]
        # Map the 1D hyperparam arrays to 2D meshes with their values repeated in the other dimension.
        eps_mesh = np.repeat(np.reshape(eps_vals, (1, -1)), y_len, axis=0)
        min_samples_mesh = np.repeat(np.reshape(min_samples, (-1, 1)), x_len, axis=1)

        plt.pcolormesh(eps_mesh, min_samples_mesh, accuracy_mesh, cmap="hot", shading="nearest")
        plt.title(f"{clustering_algo} hyper parameter search")
        plt.xlabel("max dist")
        plt.ylabel("min samples")
        plt.colorbar()
        plt.savefig(self.metrics_dir / f"{clustering_algo}_hyperparam_search.jpg")
        plt.close()

    def plot_accuracies(self, plot_data: tuple, clustering_algo: str) -> None:
        plt.plot(*plot_data, "bo")
        plt.title(f"{clustering_algo} Accuracy vs n clusters")
        plt.ylabel("accuracy")
        plt.xlabel("number of clusters")
        plt.savefig(self.metrics_dir / f"{clustering_algo}.jpg")
        plt.close()

    def evaluate_n_clusters(self, clustering_algo: str) -> (np.ndarray, np.ndarray):
        """
        Tries a range of numbers of clusters for the user to use in the elbow method to find the optimal
        number of clusters.
        """
        min_n_clusters = np.unique(self.ground_truths).shape[0]
        potential_n_clusters = np.arange(min_n_clusters, 50)
        accuracies = np.zeros_like(potential_n_clusters, dtype=float)
        for i, k in enumerate(potential_n_clusters):
            cluster_labels = self._cluster_predict(k, clustering_algo)
            acc = self._get_clustering_accuracy(cluster_labels, k)
            accuracies[i] = acc
        return potential_n_clusters, accuracies

    def dbscan_hyperparam_search(self, eps_vals: np.ndarray, min_samples: np.ndarray) -> np.ndarray:
        return self._manifold_hyperparam_search(DBSCAN, eps_vals, min_samples)

    def optics_hyperparam_search(self, eps_vals: np.ndarray, min_samples: np.ndarray) -> np.ndarray:
        return self._manifold_hyperparam_search(OPTICS, eps_vals, min_samples)

    def predict_dbscan(self, **kwargs):
        return self._predict_manifold_clustering(DBSCAN, **kwargs)

    def predict_optics(self, **kwargs):
        return self._predict_manifold_clustering(OPTICS, **kwargs)

    def _manifold_hyperparam_search(self, clustering_algo: any,
                                    eps_vals: np.ndarray, min_samples: np.ndarray) -> np.ndarray:
        acc_mesh = np.zeros((min_samples.shape[0], eps_vals.shape[0]), dtype=float)
        print("Doing hyperparam search, this could take a  while...")
        for i, eps in enumerate(eps_vals):
            for j, min_val in enumerate(min_samples):
                db_scan = clustering_algo(eps=eps, min_samples=min_val)
                predict = db_scan.fit_predict(self.embeds)
                acc = self._get_clustering_accuracy(predict)
                acc_mesh[j, i] = acc
        return acc_mesh

    def _predict_manifold_clustering(self, clustering_algo: any, **kwargs) -> None:
        if "eps" not in kwargs.keys() and "min_samples" not in kwargs.keys():
            kwargs = {"eps": 70, "min_samples": 12}
        db_scan = clustering_algo(**kwargs, n_jobs=-1)
        predict = db_scan.fit_predict(self.embeds)
        n_outliers = np.count_nonzero(predict == -1)
        n_clusters = predict.max() + 1
        percentage_outliers = n_outliers / predict.shape[0]
        acc = self._get_clustering_accuracy(predict)
        result = f"DBSCAN accuracy: {acc}, \npercentage outliers: {percentage_outliers}," \
                 f"\nnumber clusters: {n_clusters}\n"
        print(result)
        dbscan_results_file = self.metrics_dir / "dbscan_results.txt"
        with dbscan_results_file.open("w+") as f:
            f.write(result)
            f.write(f"")
            f.write("arguments:")
            f.write(dumps(kwargs))

    def _get_clustering_accuracy(self, cluster_labels: np.ndarray, n_clusters: int = -1) -> float:
        try:
            predicts = map_clusterlabels_to_groundtruth(self.ground_truths, cluster_labels, n_clusters).numpy()
            result = cluster_acc(self.ground_truths, predicts)
            return result
        except RuntimeError:
            return 0.

    def _cluster_predict(self, k: int, clustering_algo: str) -> np.ndarray:
        if clustering_algo == "KMEANS":
            return self._predict_kmeans(k)
        elif clustering_algo == "AGGLOMERATIVE_CLUSTERING":
            return self._predict_agglomerative_clustering(k)
        elif clustering_algo == "SOM":
            if self._is_squared_number(k):
                return self._predict_self_organizing_maps(k)
            else:
                return np.full((self.embeds.shape[0],), -1, dtype=int)
        else:
            raise Exception(
                f"Invalid clustering algorithm: {clustering_algo}, clustering algorithm must be in "
                f"{N_CLUSTERS_CLUSTERING_ALGORITHMS}"
            )

    def _predict_kmeans(self, k: int) -> np.ndarray:
        km = KMeans(n_clusters=k)
        predict = km.fit_predict(self.embeds)
        return predict

    def _predict_agglomerative_clustering(self, n_clusters: int) -> np.ndarray:
        ac = AgglomerativeClustering(n_clusters=n_clusters)
        predict = ac.fit_predict(self.embeds)
        return predict

    def _predict_self_organizing_maps(self, k: int):
        if not self._is_squared_number(k):
            raise ValueError(f"Input should be a squared number, got{k}")
        grid_length = int(sqrt(k))
        som = SOMClustering(grid_length, grid_length)
        som.fit(self.embeds)
        bmus = som.get_bmus(self.embeds)
        if bmus is not None:
            predict = self._som_bmus_to_label(bmus)
        else:
            raise RuntimeError(f"SOM predictions returned: {bmus}")
        return predict

    def _som_bmus_to_label(self, bmus_predicts: List[Tuple[int, int]]) -> np.ndarray:
        [_, column_indices] = list(zip(*bmus_predicts))   # Unzip operation into list of 2 tuples.
        n_cols = max(column_indices) + 1
        labels = np.array([self._get_bmu_to_label(r, c, n_cols) for r, c, in bmus_predicts])
        return labels

    @staticmethod
    def _get_bmu_to_label(row_index: int, col_index: int, n_cols: int) -> int:
        return row_index * n_cols + col_index

    @staticmethod
    def _is_squared_number(k: int) -> bool:
        return int(sqrt(k))**2 == k

    @staticmethod
    def _load_embeds(directory: Path) -> np.ndarray:
        filename = "test_embeds.npy"
        loaded_data = np.load(directory / filename)
        return loaded_data

    def _read_ground_truths(self) -> None:
        ground_truths = get_ground_truths(self.dataset_dir, "compacted_target.npy")
        encoding = {v: k for k, v in enumerate(np.unique(ground_truths))}
        encoded_truth = np.array([encoding[gt] for gt in ground_truths])
        self.ground_truths = encoded_truth


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, required=True)
    parser.add_argument("-c", "--clustering-algo", type=str.upper, required=True)
    args = parser.parse_args()

    start = datetime.now()
    root = Path(__file__).parent
    metrics_dir = root / "metrics" / args.directory
    clustering_module = Clustering.from_metrics_dir(metrics_dir)
    eps_vals = np.arange(1, 10.5, 0.5)
    min_sample_vals = np.arange(2, 7, 1)

    print(f"Evaluating embeds using {args.clustering_algo.lower()}...")
    if args.clustering_algo == "ALL":
        clustering_module.explore_clusterings_and_hyperparams(eps_vals, min_sample_vals)
    elif args.clustering_algo == "DBSCAN":
        heatmap = clustering_module.dbscan_hyperparam_search(eps_vals, min_sample_vals)
        # heatmap = np.random.rand(min_sample_vals.shape[0], eps_vals.shape[0])
        clustering_module.plot_hyperparam_search(heatmap, eps_vals, min_sample_vals, "DBSCAN")
        # clustering_module.predict_dbscan(eps=90, min_samples=20)
    elif args.clustering_algo == "OPTICS":
        heatmap = clustering_module.optics_hyperparam_search(eps_vals, min_sample_vals)
        # heatmap = np.random.rand(min_sample_vals.shape[0], eps_vals.shape[0])
        clustering_module.plot_hyperparam_search(heatmap, eps_vals, min_sample_vals, "OPTICS")
        # clustering_module.predict_optics(eps=90, min_samples=20)
    else:
        plot_data = clustering_module.evaluate_n_clusters(args.clustering_algo)
        clustering_module.plot_accuracies(plot_data, args.clustering_algo)
    duration = relativedelta(datetime.now(), start)
    print(f"duration: {duration.normalized()}")
