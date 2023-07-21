from unittest import TestCase
import Mapping
import pandas as pd
import STvEA
import Annotation
import DataProcessor


class TestAnnotation(TestCase):
    def test_cluster_heatmap(self):
        stvea = STvEA.STvEA()

        # test cite
        stvea.cite_cluster = pd.read_csv("../Tests/R_cite_clusters.csv", index_col=0, header=0).astype("float64")
        stvea.cite_cluster = stvea.cite_cluster[stvea.cite_cluster.columns[-1]]
        stvea.cite_protein = pd.read_csv("../Tests/r_cite_clean.csv", index_col=0, header=0).astype("float64")
        Annotation.Annotation().cluster_heatmap(stvea, 1, 2)

        # test codex
        stvea.codex_cluster = pd.read_csv("../Tests/python_codex_clusters.csv", index_col=0, header=0).astype(int)
        stvea.codex_protein_corrected = pd.read_csv("../Tests/python_codex_protein.csv", index_col=0, header=0).astype("float64")
        Annotation.Annotation().cluster_heatmap(stvea, 2, 2)

    def test_cluster_names(self):
        cluster_index = ["1", "2", "3", "4"]
        stvea = STvEA.STvEA()
        Annotation.Annotation().cluster_names(cluster_index, stvea, 1)
        Annotation.Annotation().cluster_names(cluster_index, stvea, 2)

    def test_transfer_labels(self):
        stvea = STvEA.STvEA()
        stvea.cite_cluster = pd.read_csv("../Tests/python_cite_cluster.csv", index_col=0, header=0).astype(int)
        stvea.cite_protein = pd.read_csv("../Tests/python_cite_clean.csv", index_col=0, header=0).astype("float64")
        stvea.transfer_matrix = pd.read_csv("../Tests/python_transfer_matrix.csv", index_col=0, header=0).astype("float64")
        Annotation.Annotation().transfer_labels(stvea)

    def test_evaluation(self):
        stvea = STvEA.STvEA()
        stvea.cite_cluster = pd.read_csv("../Tests/python_cite_cluster.csv", index_col=0, header=0).astype(int)
        stvea.cite_protein = pd.read_csv("../Tests/python_cite_clean.csv", index_col=0, header=0).astype("float64")
        stvea.transfer_matrix = pd.read_csv("../Tests/python_transfer_matrix.csv", index_col=0, header=0).astype(
            "float64")
        stvea.codex_cluster = pd.read_csv("../Tests/python_codex_clusters.csv", index_col=0, header=0).astype(int)
        stvea.codex_protein_corrected = pd.read_csv("../Tests/python_codex_protein.csv", index_col=0, header=0).astype(
            "float64")
        Annotation.Annotation().evaluation(stvea)


