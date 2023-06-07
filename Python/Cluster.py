import numpy as np
import umap.umap_ as umap
import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster


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
        :param min_cluster_size_list: a vector of min_cluster_size arguments to scan over
        :param min_sample_list: a vector of min_sample arguments to scan over
        :param cache_dir: a folder to store intermediary results of HDBSCAN
        :return: a list of assigned labels.
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
    def parameter_scan(stvea, min_cluster_size_range, min_sample_range,
                       n_neighbors=50, min_dist=0.1, negative_sample_rate=50, metric="correlation"):
        """
        This function will run HDBSCAN multiple times given the vector of min_cluster_size_range and min_sample_range.
        The result will be a list of dictionaries that will record each HDBSCAN's scores and generated labels.
        :param stvea: a stvea object
        :param min_cluster_size_range: a vector of min_cluster_size arguments to scan over
        :param min_sample_range: a vector of min_sample arguments to scan over
        :param n_neighbors: the number of neighbors
        :param min_dist: the effective minimum distance between embedded points.
        :param negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
        :param metric: Pearson correlation should be used here.
        :return: a list of dictionaries that will record each HDBSCAN's scores and generated labels.
        """
        cite_latent = stvea.cite_latent
        # Running UMAP on the CITE-seq latent space
        print("Running UMAP on the CITE-seq latent space")
        reducer = umap.UMAP(n_components=cite_latent.shape[1], n_neighbors=n_neighbors, min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate, metric=metric)
        umap_latent = reducer.fit_transform(cite_latent)

        # Running HDBSCAN on the UMAP space
        print("Running HDBSCAN on the UMAP space")
        hdbscan_labels = Cluster().run_hdbscan(cite_latent, umap_latent, min_cluster_size_range, min_sample_range)

        # calculate scores
        all_scores = []
        hdbscan_results = []
        for label in hdbscan_labels:
            # Calculate silhouette scores
            #score = silhouette_score(cite_latent, label, metric="correlation")
            scores = silhouette_samples(cite_latent, label, metric="correlation")
            score = np.mean(scores)
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

        stvea.hdbscan_scans = hdbscan_results
        return

    @staticmethod
    def consensus_cluster_internal(distance_matrix, inconsistent_value=0.3, min_cluster_size=10):
        new_distance = squareform(distance_matrix)
        hierarchical_tree = linkage(new_distance, "average")
        hier_consensus_labels = fcluster(hierarchical_tree, t=inconsistent_value)
        hier_unique_labels = set(hier_consensus_labels)
        for label in hier_unique_labels:
            indices = [i for i, x in enumerate(hier_consensus_labels) if x == label]
            if len(indices) < min_cluster_size:
                for index in indices:
                    hier_consensus_labels[index] = -1
        return hier_consensus_labels.tolist()

    @staticmethod
    def consensus_cluster(stvea, silhouette_cutoff, inconsistent_value, min_cluster_size):
        hdbscan_results = stvea.hdbscan_scans
        # initialize some variables
        num_cells = len(hdbscan_results[0]['cluster_labels'])
        # initialize a consensus matrix
        consensus_matrix = np.zeros((num_cells, num_cells))
        total_runs = 0

        for result in hdbscan_results:

            if result['silhouette_score'] >= silhouette_cutoff:
                sim_matrix = np.zeros((num_cells, num_cells))

                for cell1 in range(num_cells):
                    if result['cluster_labels'][cell1] != -1:
                        sim_matrix[:, cell1] = 1 * (result['cluster_labels'] == result['cluster_labels'][cell1])

                np.fill_diagonal(sim_matrix, 1)
                consensus_matrix -= sim_matrix
                total_runs += 1

        consensus_matrix += total_runs
        if total_runs > 0:
            consensus_matrix /= total_runs
        else:
            print("Warning: No clustering runs passed the silhouette score cutoff")

        consensus_clusters = Cluster().consensus_cluster_internal(consensus_matrix,
                                                                  inconsistent_value,
                                                                  min_cluster_size)
        # don't relabel the result as R implementation did
        stvea.cite_cluster = consensus_clusters
        return







