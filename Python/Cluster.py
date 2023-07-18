import igraph as igraph
import numpy as np
import umap.umap_ as umap
import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from igraph import Graph


class Cluster:

    def __init__(self):
        pass

    @staticmethod
    def add_to_edge_list(edge_list, row, row_name):
        """
        This is a helper function invoked by cluster_codex.

        @param edge_list: an edge list.
        @param row: a pandas' dataframe row.
        @param row_name: the row's name.
        """
        row.apply(lambda col: edge_list.append([row_name, col]))

    @staticmethod
    def cluster_codex(stvea, k=30, knn_option=1):
        """
        This function will cluster codex cells.

        @param stvea: a STvEA object.
        @param k: the number of nearest neighbors to generate graph.
        The graph will be used to perform Louvain community detection.
        @param knn_option: the way to detect nearest neighbors.
        1: use Pearson distance to find nearest neighbors on CODEX protein data.
        2: use Euclidean distance to find nearest neighbors on 2D CODEX embedding data.
        @return:
        """
        # find knn
        if knn_option == 1:
            # use Pearson distance to find nearest neighbors on CODEX protein data.
            stvea.codex_knn = pd.DataFrame(
                umap.nearest_neighbors(X=stvea.codex_protein, metric="correlation", n_neighbors=k, metric_kwds={},
                                       random_state=0, angular=False)[0])
            stvea.codex_knn = stvea.codex_knn.iloc[:, 1:]
        elif knn_option == 2:
            # use Euclidean distance to find nearest neighbors on 2D CODEX embedding data.
            stvea.codex_knn = pd.DataFrame(
                umap.nearest_neighbors(X=stvea.codex_emb, metric="euclidean", n_neighbors=k, metric_kwds={},
                                       random_state=0, angular=False)[0])
            stvea.codex_knn = stvea.codex_knn.iloc[:, 1:]
        else:
            raise ValueError

        # convert to pandas dataframe
        codex_knn = pd.DataFrame(stvea.codex_knn)

        # create edge list
        edge_list = []
        codex_knn.apply(lambda row:Cluster().add_to_edge_list(edge_list, row, row.name), axis=1)

        # perform louvain community detection ()
        g = Graph(edges=edge_list)
        stvea.codex_cluster = g.community_multilevel().membership

        return


    @staticmethod
    def codex_umap(stvea, metric="correlation", n_neighbors=30, min_dist=0.1, negative_sample_rate=50):
        """
        This method perform umap on codex protein data and create a 2d embedding.

        @param stvea: a STvEA object.
        @param metric: the metric to use here, default to correlation.
        @param n_neighbors: the number of neighbors, default to 30.
        @param min_dist: the effective minimum distance between embedded points.
        @param negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
        @return:
        """
        if stvea.codex_protein.empty:
            raise ValueError("stvea object does not contain codex protein data")

        res = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate, n_components=2).fit_transform(stvea.codex_protein)
        stvea.codex_emb = pd.DataFrame(res)

        return


    @staticmethod
    def cite_umap(stvea, metric="correlation", n_neighbors=50, min_dist=0.1, negative_sample_rate=50):
        """
        This function will perform umap on cite_latent or cite_mrna data,
        and construct a 2D embedding.
        cite_latent and cite_mrna will be used here instead of protein data
        because cite_latent and cite_mrna are more informative

        @param stvea: a STvEA object.
        @param metric: the metric to calculate distance.
        @param n_neighbors: the number of neighbors.
        @param min_dist: the effective minimum distance between embedded points.
        @param negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
        @return:
        """
        if stvea.cite_latent.empty and stvea.cite_mRNA.empty:
            raise ValueError("stvea object does not contain CITE-seq mRNA or latent data")

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
    def run_hdbscan(latent, umap_latent, min_cluster_size_list, min_sample_list, metric, cache_dir="./HDBSCAN_cache"):
        """
        This function is borrowed from original STvEA R library.
        https://github.com/CamaraLab/STvEA/blob/master/inst/python/consensus_clustering.py

        @param metric: metric to run HDBSCAN
        @param umap_latent: cite_latent data after being transformed by umap.
        @param min_cluster_size_list: a vector of min_cluster_size arguments to scan over
        @param min_sample_list: a vector of min_sample arguments to scan over
        @param cache_dir: a folder to store intermediary results of HDBSCAN
        @return: a list of assigned labels.
        """
        latent = np.array(latent)
        num_cells = latent.shape[0]
        umap_latent = np.array(umap_latent)
        results = []
        for min_samples in min_sample_list:
            for min_cluster_size in min_cluster_size_list:
                if (cache_dir is None):
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
                                                metric=metric)
                else:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
                                                metric=metric, memory=cache_dir)
                hdbscan_labels = clusterer.fit_predict(umap_latent)
                results.append(hdbscan_labels.tolist())
        return results

    @staticmethod
    def parameter_scan(stvea, min_cluster_size_range, min_sample_range,
                       n_neighbors=50, min_dist=0.1, negative_sample_rate=50, metric="correlation"):
        """
        This function will run HDBSCAN multiple times given the vector of min_cluster_size_range and min_sample_range.
        The result will be a list of dictionaries that will record each HDBSCAN's scores and generated labels.
        @param stvea: a stvea object
        @param min_cluster_size_range: a vector of min_cluster_size arguments to scan over
        @param min_sample_range: a vector of min_sample arguments to scan over
        @param n_neighbors: the number of neighbors
        @param min_dist: the effective minimum distance between embedded points.
        @param negative_sample_rate: the number of negative samples to select per positive sample in the optimization process.
        @param metric: Pearson correlation should be used here.
        @return: a list of dictionaries that will record each HDBSCAN's scores and generated labels.
        """
        cite_latent = stvea.cite_latent
        # Running UMAP on the CITE-seq latent space
        print("Running UMAP on the CITE-seq latent space")
        reducer = umap.UMAP(n_components=cite_latent.shape[1], n_neighbors=n_neighbors, min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate, metric=metric)
        umap_latent = reducer.fit_transform(cite_latent)

        # Running HDBSCAN on the UMAP space
        print("Running HDBSCAN on the UMAP space")
        hdbscan_labels = Cluster().run_hdbscan(cite_latent, umap_latent, min_cluster_size_range, min_sample_range, metric=metric)

        # calculate scores
        all_scores = []
        hdbscan_results = []
        for label in hdbscan_labels:
            # Calculate silhouette scores
            #score = silhouette_score(cite_latent, label, metric="correlation")
            scores = silhouette_samples(cite_latent, label, metric="euclidean")
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
        """
        This function was borrowed from original STvEA R program.
        This function will perform clustering given the distance matrix.
        https://github.com/CamaraLab/STvEA/blob/master/inst/python/consensus_clustering.py

        @param distance_matrix: a consensus distance matrix based on HDBSCAN results.
        @param inconsistent_value: input parameter to fcluster determining where clusters are cut in the hierarchical tree.
        @param min_cluster_size: cells in clusters smaller than this value are assigned a cluster ID of -1, indicating no cluster assignment.
        @return: a list of labels
        """
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
        """
        This function will first generate a distance matrix based on HDBSCAN results,
        and then invoke clustering function to perform clustering based on the distance matrix.
        @param stvea: a STvEA object.
        @param silhouette_cutoff: HDBSCAN results below this cutoff will be discarded.
        @param inconsistent_value: input parameter to fcluster determining where clusters are cut in the hierarchical tree.
        @param min_cluster_size: cells in clusters smaller than this value are assigned a cluster ID of -1, indicating no cluster assignment.
        @return
        """
        # initialize some variables
        hdbscan_results = stvea.hdbscan_scans
        num_cells = len(hdbscan_results[0]['cluster_labels'])
        # initialize a consensus matrix
        consensus_matrix = np.zeros((num_cells, num_cells))
        total_runs = 0

        # scan each result in the hdbscan_results
        for result in hdbscan_results:

            # if the silhouette score is larger than the silhouette_cutoff
            if result['silhouette_score'] >= silhouette_cutoff:

                # calculate a similar matrix sim_matrix
                sim_matrix = np.zeros((num_cells, num_cells))

                # loop through each cell
                for cell1 in range(num_cells):

                    # if a cell belongs to the same cluster
                    # 1 in the entry
                    if result['cluster_labels'][cell1] != -1:
                        condition = 1 * (pd.Series(result['cluster_labels']) == result['cluster_labels'][cell1])
                        sim_matrix[:, cell1] = condition

                # set diagonal entries in the matrix to be 1
                np.fill_diagonal(sim_matrix, 1)

                # add the results to the consensus_matrix
                consensus_matrix -= sim_matrix
                total_runs += 1

        # flip the result to become a distance matrix
        consensus_matrix += total_runs

        if total_runs > 0:
            # normalize
            consensus_matrix /= total_runs
        else:
            print("Warning: No clustering runs passed the silhouette score cutoff")
            exit(1)

        # perform clustering
        consensus_clusters = Cluster().consensus_cluster_internal(consensus_matrix,
                                                                  inconsistent_value,
                                                                  min_cluster_size)
        # relabel the result as R implementation did
        original_array = np.array(consensus_clusters)

        # Find unique elements and sort them
        unique_elements = np.sort(np.unique(original_array))

        # Create a pandas series with index as unique_elements and values as sequence from -1
        map_series = pd.Series(np.concatenate([np.arange(-1, 0), np.arange(1, len(unique_elements))]), index=unique_elements)

        # Map original_array to new values using map_series
        new_array = map_series[original_array].values

        # store the result in the STvEA object
        stvea.cite_cluster = new_array

        return













