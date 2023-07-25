from unittest import TestCase
import Controller
import STvEA
import pandas as pd
import Annotation
from copy import deepcopy


class TestController(TestCase):

    @staticmethod
    def partial_evaluation(stvea, annotation):
        """
        This function will evaluate the performance of label transferring based on given data stored in STvEA object.
        """
        # transfer labels
        annotation.transfer_labels()
        # show the CODEX protein expression level
        cluster_index = annotation.cluster_heatmap(2, 2)
        # user input CODEX cluster names
        annotation.cluster_names(cluster_index, 2)
        # calculate the percentage of labels that are consistent between transferred label and user-annotated CODEX labels.
        codex_clusters = deepcopy(stvea.codex_cluster)
        codex_clusters_names = codex_clusters.applymap(lambda x:
                                                       stvea.codex_cluster_name_dict.get(x, "Unknowns"))
        combined = pd.DataFrame({"Original": codex_clusters_names.iloc[:, 0],
                                 "Transferred": stvea.codex_cluster_names_transferred.iloc[:, 0]},
                                index=stvea.codex_protein_corrected.index)
        # check whether transferred labels and user-input labels
        equality = combined.apply(lambda x: x[0] == x[1], axis=1)
        # print the result
        print("Matched Proportion: " + str(equality.mean()))

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
        cn = Controller.Controller()



