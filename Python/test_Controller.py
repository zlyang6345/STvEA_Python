import time
from unittest import TestCase
import Controller
import STvEA
import pandas as pd
import Annotation
from copy import deepcopy
import tracemalloc


class TestController(TestCase):

    def test_transfer_matrix_scalability(self):
        cn = Controller.Controller()
        amount_codex = -1
        amount_cite = -1
        cell_numbers = ((7000, 7000), (2000, 2000), (3000, 3000), (4000, 4000),
                        (5000, 5000), (6000, 6000), (7000, 7000), (8000, 8000))
        for cell_numbers in cell_numbers:
                amount_codex, amount_cite = cell_numbers
                print(f"------------------amount_codex: {amount_codex} amount_cite: {amount_cite}--------------------")
                cn.data_processor.read_codex(codex_blanks="../Data/raw_dataset/codex_blanks.csv",
                                             codex_protein="../Data/raw_dataset/codex_protein.csv",
                                             codex_size="../Data/raw_dataset/codex_size.csv",
                                             codex_spatial="../Data/raw_dataset/codex_spatial.csv",
                                             codex_preprocess=True,
                                             codex_border=564000
                                             )
                cn.data_processor.read_cite(cite_latent="../Data/raw_dataset/cite_latent.csv",
                                            cite_protein="../Data/raw_dataset/cite_protein.csv",
                                            cite_mrna="../Data/raw_dataset/cite_mRNA.csv")
                # take a certain number of cells
                cn.data_processor.take_subset(amount_codex=amount_codex,
                                              amount_cite=amount_cite)
                # map the CODEX cells to CITE-seq cells.
                cn.mapping.map_codex_to_cite(k_find_nn=80,
                                             k_find_anchor=20,
                                             k_filter_anchor=100,
                                             k_score_anchor=80,
                                             k_find_weights=100)
                t = []
                for round in range(1):
                    # create transfer matrix to transfer values from CITE-seq to CODEX
                    # start = time.time()
                    tracemalloc.start()
                    cn.mapping.transfer_matrix(k=None,
                                               c=0.1,
                                               mask_threshold=0.5,
                                               mask=False,
                                               option=2)
                    # end = time.time()
                    # t.append(end - start)
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')

                    # print("[ Top 10 ]")
                    for stat in top_stats[:10]:
                      print(stat)
                print(f"cell numbers:  {cell_numbers[0]} time: {t}")



    def test_runtime_scalability(self):
        cn = Controller.Controller()
        amount_codex = -1
        amount_cite = -1
        for round in range(2):
          print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~round: {round + 1}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
          cell_numbers = ((1000, 1000), (2000, 2000),(3000, 3000), (4000, 4000),
                          (5000, 5000), (6000, 6000), (7000, 7000), (8000, 8000))
          for cell_numbers in cell_numbers:
              amount_codex, amount_cite = cell_numbers
              print(f"------------------amount_codex: {amount_codex} amount_cite: {amount_cite}--------------------")
              cn.data_processor.read_codex(codex_blanks="../Data/raw_dataset/codex_blanks.csv",
                                           codex_protein="../Data/raw_dataset/codex_protein.csv",
                                           codex_size="../Data/raw_dataset/codex_size.csv",
                                           codex_spatial="../Data/raw_dataset/codex_spatial.csv",
                                           codex_preprocess=True,
                                           codex_border=564000
                                           )
              cn.data_processor.read_cite(cite_latent="../Data/raw_dataset/cite_latent.csv",
                                          cite_protein="../Data/raw_dataset/cite_protein.csv",
                                          cite_mrna="../Data/raw_dataset/cite_mRNA.csv")
              # take a certain number of cells
              cn.data_processor.take_subset(amount_codex=amount_codex,
                                            amount_cite=amount_cite)
              # map the CODEX cells to CITE-seq cells.
              cn.mapping.map_codex_to_cite(k_find_nn=80,
                                           k_find_anchor=20,
                                           k_filter_anchor=100,
                                           k_score_anchor=80,
                                           k_find_weights=100)
              # create transfer matrix to transfer values from CITE-seq to CODEX
              cn.mapping.transfer_matrix(k=None,
                                         c=0.1,
                                         mask_threshold=0.5,
                                         mask=False,
                                         option=2)
              print("---------------------------------------------------------------------------------------\n\n")

    def test_space_scalability(self):
        cn = Controller.Controller()
        amount_codex = -1
        amount_cite = -1
        for round in range(1):
            print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~round: {round + 1}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            cell_numbers = ((1000, 1000), )
            for cell_numbers in cell_numbers:
                amount_codex, amount_cite = cell_numbers
                print(f"------------------amount_codex: {amount_codex} amount_cite: {amount_cite}--------------------")
                cn.data_processor.read_codex(codex_blanks="../Data/raw_dataset/codex_blanks.csv",
                                             codex_protein="../Data/raw_dataset/codex_protein.csv",
                                             codex_size="../Data/raw_dataset/codex_size.csv",
                                             codex_spatial="../Data/raw_dataset/codex_spatial.csv",
                                             codex_preprocess=True,
                                             codex_border=564000
                                             )
                cn.data_processor.read_cite(cite_latent="../Data/raw_dataset/cite_latent.csv",
                                            cite_protein="../Data/raw_dataset/cite_protein.csv",
                                            cite_mrna="../Data/raw_dataset/cite_mRNA.csv")

                # take a certain number of cells
                cn.data_processor.take_subset(amount_codex=amount_codex,
                                              amount_cite=amount_cite)

                # map the CODEX cells to CITE-seq cells.
                cn.mapping.map_codex_to_cite(k_find_nn=80,
                                             k_find_anchor=20,
                                             k_filter_anchor=100,
                                             k_score_anchor=80,
                                             k_find_weights=100)


                tracemalloc.start()

                # create transfer matrix to transfer values from CITE-seq to CODEX
                cn.mapping.transfer_matrix(k=None,
                                           c=0.1,
                                           mask_threshold=0.5,
                                           mask=False,
                                           option=2)

                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                print("[ Top 10 ]")
                for stat in top_stats[:10]:
                    print(stat)

                print("---------------------------------------------------------------------------------------\n\n")

            print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~round complete~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    @staticmethod
    def partial_evaluation(stvea,
                           annotation,
                           export=True):
        """
        This function will evaluate the performance of label transferring based on given data stored in STvEA object.
        """

        # show the CODEX protein expression level
        cluster_index = annotation.cluster_heatmap(2, 2)

        # transfer labels
        annotation.transfer_labels()

        if export:
            stvea.cite_cluster.to_csv("../Tests/ToImproveTPR/cite_cluster.csv")
            stvea.cite_mRNA.to_csv("../Tests/ToImproveTPR/cite_mRNA.csv")
            stvea.codex_cluster_names_transferred.to_csv("../Tests/ToImproveTPR/codex_cluster_names_transferred.csv")
            stvea.codex_protein.to_csv("../Tests/ToImproveTPR/codex_protein.csv")
            stvea.codex_protein_corrected.to_csv("../Tests/ToImproveTPR/codex_protein_corrected.csv")
            stvea.transfer_matrix.to_csv("../Tests/ToImproveTPR/transfer_matrix.csv")

        # user input CODEX cluster names
        annotation.cluster_names(cluster_index, 2, option=2)

        # calculate the percentage of labels that are consistent between transferred label and user-annotated CODEX labels.
        codex_clusters = deepcopy(stvea.codex_cluster)
        codex_clusters_names = codex_clusters.applymap(lambda x:
                                                       stvea.codex_cluster_name_dict.get(x))
        combined = pd.DataFrame({"Original": codex_clusters_names.iloc[:, 0],
                                 "Transferred": stvea.codex_cluster_names_transferred.iloc[:, 0]},
                                index=stvea.codex_protein_corrected.index)

        # check whether transferred labels and user-input labels equal
        equality = combined.apply(lambda x: x[0] == x[1], axis=1)

        # filter out these CODEX cells that user does not assign a CODEX cluster name or whose transferred label is null.
        # mask = ((combined["Original"] != "") & (combined["Transferred"] != ""))
        # combined = combined[mask]

        # print each cell type's result
        reality = "Original"
        test = "Transferred"
        cell_types = combined[reality].unique()
        for type in cell_types:
            type_cells_reality = combined[reality] == type
            non_type_cells_reality = ~type_cells_reality
            type_cells_test = combined[test] == type
            non_type_cells_test = ~type_cells_test

            # true positive rate
            tpr = (type_cells_reality & type_cells_test).sum() / type_cells_reality.sum()
            # true negative rate
            tnr = (non_type_cells_reality & non_type_cells_test).sum() / non_type_cells_reality.sum()

            print(f"{type}: TPR: {round(tpr * 100, 2)}% TNR: {round(tnr * 100, 2)}%")

            index = (combined["Original"] == type)
            subset = combined.loc[index,]
            transferred_majority = subset["Transferred"].value_counts().idxmax()
            print(f"Transferred majority: {transferred_majority}")
            print()

        print()

        unique_codex_clusters = codex_clusters.loc[:, 0].unique()
        for each in unique_codex_clusters:
            index = (codex_clusters.loc[:, 0] == each)
            subset = stvea.codex_cluster_names_transferred.loc[index, 0]
            subset_value_count = subset.value_counts()
            transferred_majority = subset_value_count.idxmax()
            count_sum = subset_value_count.sum()
            subset_value_percent = subset_value_count / count_sum

            print(
                f"CODEX cluster {each} ({subset.shape[0]} cells) transferred majority: {round(subset_value_percent[transferred_majority] * 100, 3)} % {transferred_majority}")

    def test_partial_evaluation(self):
        stvea = STvEA.STvEA()
        an = Annotation.Annotation(stvea)
        stvea.cite_cluster = pd.read_csv("../Tests/python_cite_cluster.csv", index_col=0, header=0).astype(int)
        stvea.cite_protein = pd.read_csv("../Tests/python_cite_clean.csv", index_col=0, header=0).astype("float64")
        stvea.transfer_matrix = pd.read_csv("../Tests/python_transfer_matrix.csv", index_col=0, header=0).astype(
            "float64")
        stvea.codex_cluster = pd.read_csv("../Tests/python_codex_clusters.csv", index_col=0, header=0).astype(int)
        stvea.codex_protein_corrected = pd.read_csv("../Tests/python_codex_protein.csv", index_col=0, header=0).astype(
            "float64")
        TestController.partial_evaluation(stvea, an)

    def test_overall_evaluation(self):
        # initialize variable
        cn = Controller.Controller()

        # this pipeline will read files
        cn.pipeline(
            # read_codex args
            codex_blanks="../Data/raw_dataset/codex_blanks.csv",
            codex_protein="../Data/raw_dataset/codex_protein.csv",
            codex_size="../Data/raw_dataset/codex_size.csv",
            codex_spatial="../Data/raw_dataset/codex_spatial.csv",
            codex_preprocess=True,
            codex_border=564000,
            # read_cite args
            cite_latent="../Data/raw_dataset/cite_latent.csv",
            cite_protein="../Data/raw_dataset/cite_protein.csv",
            cite_mrna="../Data/raw_dataset/cite_mRNA.csv",
            # take_subset args
            amount_codex=1000,  # -1 = default ≈ 9000 CODEX cells
            amount_cite=1000,  # -1 ≈ 7000 cells
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
            cluster_codex_k=4,
            cluster_codex_knn_option=1,
            # parameter_scan args
            cluster_cite_option=2,
            parameter_scan_min_cluster_size_range=tuple(range(2, 3, 1)),
            parameter_scan_min_sample_range=tuple(range(14, 15, 1)),
            parameter_scan_n_neighbors=40,
            parameter_scan_min_dist=0.1,
            parameter_scan_negative_sample_rate=50,
            parameter_scan_metric="correlation",
            silhoutte_metric="correlation",
            # consensus_cluster args
            consensus_cluster_silhouette_cutoff=None,
            consensus_cluster_inconsistent_value=0.1,
            consensus_cluster_min_cluster_size=8,
            silhouette_cutoff_percentile=95,
            # cite hdbscan
            hdbscan_min_cluster_size_range=2,
            hdbscan_min_sample_range=14,
            hdbscan_n_neighbors=50,
            hdbscan_min_dist=0.1,
            hdbscan_negative_sample_rate=50,
            hdbscan_cluster_metric="correlation",
            hdbscan_random_state=0,
            # map_codex_to_cite args
            k_find_nn=80,
            k_find_anchor=20,
            k_filter_anchor=100,
            k_score_anchor=80,
            k_find_weights=100,
            # transfer_matrix
            k_transfer_matrix=None,
            c_transfer_matrix=0.1,
            mask=True
        )

        # invoke the partial evaluation
        TestController.partial_evaluation(cn.stvea, cn.annotation, export=False)
