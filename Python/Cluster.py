import numpy as np
import umap.umap_ as umap
import pandas as pd
from scipy import stats
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt


class Cluster:

    def __init__(self):
        pass

    @staticmethod
    def cite_umap(stvea, metric="correlation", n_neighbors=50, min_dist=0.1, negative_sample_rate=50):
        """
        This function will perform umap on cite_latent or cite_mrna data,
        and construct a 2D embedding.
        :param stvea: a STvEA object
        :param metric: the metric to calculate distance
        :param n_neighbors: the number of neighbors
        :param min_dist: the effective minimum distance between embedded points.
        :param negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
        :return:
        """
        if stvea.cite_latent.empty and stvea.cite_mRNA.empty:
            raise ValueError("stvea_object does not contain CITE-seq data")

        if not stvea.cite_latent.empty:
            # recommended
            res = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate, n_components=2).fit_transform(stvea.cite_latent)
            stvea.cite_emb = pd.DataFrame(res)

        elif not stvea.cite_mRNA.empty:
            # implemented, but not recommended
            res = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate, n_components=2).fit_transform(stvea.cite_mRNA)
            stvea.cite_emb = pd.DataFrame(res)

        return

    @staticmethod
    def run_hdbscan(latent, umap_latent, min_cluster_size_list, min_sample_list, cache_dir="./HDBSCAN_cache"):
        """
        This function is borrowed from original STvEA R library.
        https://github.com/CamaraLab/STvEA/blob/master/inst/python/consensus_clustering.py
        :param umap_latent: cite_latent data after being transformed by umap.
        :param min_cluster_size_list: vector of min_cluster_size arguments to scan over
        :param min_sample_list: vector of min_sample arguments to scan over
        :param cache_dir: a folder to store intermediary results of HDBSCAN
        :return:
        """
        latent = np.array(latent)
        num_cells = latent.shape[0]
        umap_latent = np.array(umap_latent)
        results = []
        for min_samples in min_sample_list:
            for min_cluster_size in min_cluster_size_list:
                if (cache_dir is None):
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
                                                metric="correlation")
                else:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
                                                metric="correlation", memory=cache_dir)
                hdbscan_labels = clusterer.fit_predict(umap_latent)
                results.append(hdbscan_labels.tolist())
        return results

    @staticmethod
    def parameter_scan(cite_latent, min_cluster_size_range, min_sample_range,
                       n_neighbors=50, min_dist=0.1, negative_sample_rate=50, metric="correlation"):
        # Running UMAP on the CITE-seq latent space
        print("Running UMAP on the CITE-seq latent space")
        reducer = umap.UMAP(n_components=cite_latent.shape[1], n_neighbors=n_neighbors, min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate, metric=metric)
        umap_latent = reducer.fit_transform(cite_latent)

        # Running HDBSCAN on the UMAP space
        print("Running HDBSCAN on the UMAP space")
        hdbscan_labels = Cluster().run_hdbscan(cite_latent, umap_latent, min_cluster_size_range, min_sample_range)

        all_scores = []
        hdbscan_results = []
        for label in hdbscan_labels:
            # Calculate silhouette scores
            score = silhouette_score(cite_latent, label)
            all_scores.append(score)
            hdbscan_results.append({
                "cluster_labels": label,
                "silhouette_score": score
            })

        # Plotting histogram of all silhouette scores
        plt.hist(all_scores, bins=100)
        plt.title("Histogram of silhouette scores")
        plt.xlabel("Silhouette score")
        plt.ylabel("Number of calls to HDBSCAN")
        plt.show()

        return hdbscan_results







