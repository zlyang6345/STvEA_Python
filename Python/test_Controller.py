from unittest import TestCase
import Controller
import STvEA
import pandas as pd
import Annotation
from copy import deepcopy


class TestController(TestCase):

    @staticmethod
    def partial_evaluation(stvea,
                           annotation):
        """
        This function will evaluate the performance of label transferring based on given data stored in STvEA object.
        """

        # show the CODEX protein expression level
        cluster_index = annotation.cluster_heatmap(2, 2)

        # transfer labels
        annotation.transfer_labels()

        # user input CODEX cluster names
        annotation.cluster_names(cluster_index, 2)

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
        mask = ((combined["Original"] != "") & (combined["Transferred"] != ""))
        combined = combined[mask]

        # print each cell type's result
        reality = "Original"
        test = "Transferred"
        cell_types = combined[reality].unique()
        for type in cell_types:
            type_cells_reality = list(combined[reality] == type)
            non_type_cells_reality = [not x for x in type_cells_reality]
            type_cells_test = list(combined[test] == type)
            non_type_cells_test = [not x for x in type_cells_test]
            # true positive rate
            tpr = sum([a & b for a, b in zip(type_cells_reality, type_cells_test)])/sum(type_cells_reality)
            # true negative rate
            tnr = sum([a & b for a, b in zip(non_type_cells_reality, non_type_cells_test)])/sum(non_type_cells_reality)
            print(f"{type}: TPR: {round(tpr*100, 2)}% TNR: {round(tnr*100, 2)}%")
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

            print(f"CODEX cluster {each} transferred majority: {round(subset_value_percent[transferred_majority] * 100, 3)} % {transferred_majority}")



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
            amount_codex=-1,  # -1 = default ≈ 9000 CODEX cells
            amount_cite=-1,  # -1 ≈ 7000 cells
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
            cluster_codex_k=3,
            cluster_codex_knn_option=1,
            # parameter_scan args
            parameter_scan_min_cluster_size_range=tuple(range(5, 21, 4)),
            parameter_scan_min_sample_range=tuple(range(10, 41, 3)),
            parameter_scan_n_neighbors=50,
            parameter_scan_min_dist=0.1,
            parameter_scan_negative_sample_rate=50,
            parameter_scan_metric="correlation",
            # consensus_cluster args
            consensus_cluster_silhouette_cutoff=0.126,
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
            c_transfer_matrix=0.1,
            mask=True
        )
        # invoke the partial evaluation
        TestController.partial_evaluation(cn.stvea, cn.annotation)
