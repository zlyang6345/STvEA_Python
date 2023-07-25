import STvEA
import DataProcessor
import Cluster
import Mapping
import Annotation
from copy import deepcopy
import pandas as pd


class Controller:
    st = STvEA.STvEA()
    dpr = None
    cl = None
    mp = None
    an = None

    #
    def __init__(self):
        self.st = STvEA.STvEA()
        self.dpr = DataProcessor.DataProcessor(self.st)
        self.cl = Cluster.Cluster(self.st)
        self.mp = Mapping.Mapping(self.st)
        self.an = Annotation.Annotation(self.st)

    def pipeline(self,
                 # read_codex args
                 codex_blanks="../Data/small_dataset/codex_blanks.csv",
                 codex_protein="../Data/small_dataset/codex_protein.csv",
                 codex_size="../Data/small_dataset/codex_size.csv",
                 codex_spatial="../Data/small_dataset/codex_spatial.csv",
                 codex_preprocess=True,
                 codex_border=564000,
                 # read_cite args
                 cite_latent="../Data/small_dataset/cite_latent.csv",
                 cite_protein="../Data/small_dataset/cite_protein.csv",
                 cite_mrna="../Data/small_dataset/cite_mRNA.csv",
                 # take_subset args
                 amount_codex=-1,
                 amount_cite=-1,
                 # filter_codex args
                 size_lim=(1000, 25000),
                 blank_lower=(-1200, -1200, -1200, -1200),
                 blank_upper=(6000, 2500, 5000, 2500),
                 # clean_cite args
                 maxit=500,
                 factr=1e-9,
                 optim_init=([10, 60, 2, 0.5, 0.5],
                             [4.8, 50, 0.5, 2, 0.5],
                             [2, 18, 0.5, 2, 0.5],
                             [1, 3, 2, 2, 0.5],
                             [1, 3, 0.5, 2, 0.5]),
                 ignore_warnings=True,
                 clean_cite_method="l-bfgs-b",
                 # cluster_codex args
                 cluster_codex_k=30,
                 cluster_codex_knn_option=1,
                 # parameter_scan args
                 parameter_scan_min_cluster_size_range=tuple(range(5, 21, 4)),
                 parameter_scan_min_sample_range=tuple(range(10, 41, 3)),
                 parameter_scan_n_neighbors=50,
                 parameter_scan_min_dist=0.1,
                 parameter_scan_negative_sample_rate=50,
                 parameter_scan_metric="correlation",
                 # consensus_cluster args
                 consensus_cluster_silhouette_cutoff=0.114,
                 consensus_cluster_inconsistent_value=0.1,
                 consensus_cluster_min_cluster_size=10,
                 # map_codex_to_cite args
                 k_find_nn=80,
                 k_find_anchor=20,
                 k_filter_anchor=100,
                 k_score_anchor=80,
                 k_find_weights=100,
                 # transfer_matrix
                 k_transfer_matrix=None,
                 c_transfer_matrix=0.1
                 ):
        """
        This is the ultimate pipeline of STvEA to transfer labels from CITE-seq data to CODEX data.
        """
        # read and clean data
        self.dpr.read_codex(codex_blanks=codex_blanks,
                            codex_protein=codex_protein,
                            codex_size=codex_size,
                            codex_spatial=codex_spatial,
                            codex_preprocess=codex_preprocess,
                            codex_border=codex_border
                            )
        self.dpr.read_cite(cite_latent=cite_latent,
                           cite_protein=cite_protein,
                           cite_mrna=cite_mrna)
        self.dpr.take_subset(amount_codex=amount_codex,
                             amount_cite=amount_cite)
        self.dpr.filter_codex(size_lim=size_lim,
                              blank_lower=blank_lower,
                              blank_upper=blank_upper)
        self.dpr.clean_codex()
        self.dpr.clean_cite(maxit=maxit,
                            factr=factr,
                            optim_init=optim_init,
                            ignore_warnings=ignore_warnings,
                            method=clean_cite_method)

        # cluster CODEX cells
        self.cl.cluster_codex(k=cluster_codex_k,
                              knn_option=cluster_codex_knn_option)
        # cluster CITE cells
        self.cl.parameter_scan(min_cluster_size_range=parameter_scan_min_cluster_size_range,
                               min_sample_range=parameter_scan_min_sample_range,
                               n_neighbors=parameter_scan_n_neighbors,
                               min_dist=parameter_scan_min_dist,
                               negative_sample_rate=parameter_scan_negative_sample_rate,
                               metric=parameter_scan_metric)
        self.cl.consensus_cluster(silhouette_cutoff=consensus_cluster_silhouette_cutoff,
                                  inconsistent_value=consensus_cluster_inconsistent_value,
                                  min_cluster_size=consensus_cluster_min_cluster_size)

        # map the CODEX cells to CITE-seq cells.
        self.mp.map_codex_to_cite(k_find_nn=k_find_nn,
                                  k_find_anchor=k_find_anchor,
                                  k_filter_anchor=k_filter_anchor,
                                  k_score_anchor=k_score_anchor,
                                  k_find_weights=k_find_weights)

        # create transfer matrix to transfer values from CITE-seq to CODEX
        self.mp.transfer_matrix(k=k_transfer_matrix,
                                c=c_transfer_matrix)

