from unittest import TestCase
import Mapping
import pandas as pd
import STvEA
import Annotation
import DataProcessor


class TestAnnotation(TestCase):
    def test_cite_cluster_heatmap(self):
        stvea = STvEA.STvEA()
        stvea.cite_cluster = pd.read_csv("../Tests/R_cite_clusters.csv", index_col=0, header=0).astype("float64")
        stvea.cite_cluster[stvea.cite_cluster.columns[-1]] = stvea.cite_cluster[stvea.cite_cluster.columns[-1]].astype(
            int)
        stvea.cite_protein = pd.read_csv("../Tests/r_cite_clean.csv", index_col=0, header=0).astype("float64")
        Annotation.Annotation().cite_cluster_heatmap(stvea, 2)

    def test_cite_cluster_names(self):
        cluster_index = ["1", "2", "3", "4"]
        Annotation.Annotation().cite_cluster_names(cluster_index)
        pass

    def test_transfer_labels(self):

        pass
