from unittest import TestCase
import DataProcessor

import pandas as pd
import Cluster

import STvEA


class TestCluster(TestCase):
    def test_cite_umap(self):
        stvea = STvEA.STvEA()
        stvea.cite_latent = pd.read_csv("../Data/cite_latent.csv", index_col=0, header=0)
        stvea.cite_latent = stvea.cite_latent.apply(pd.to_numeric)
        Cluster.Cluster().cite_umap(stvea)
        stvea.cite_emb.to_csv("../Tests/python_cite_emb_from_cite_latent.csv")

        stvea = STvEA.STvEA
        data_processor = DataProcessor.DataProcessor()
        data_processor.read(stvea)
        data_processor.clean_cite(stvea)
        stvea.cite_latent = pd.DataFrame
        Cluster.Cluster().cite_umap(stvea)
        stvea.cite_emb.to_csv("../Tests/python_cite_emb_from_cite_mrna.csv")

    def test_parameter_scan(self):
        stvea = STvEA.STvEA()
        stvea.cite_latent = pd.read_csv("../Data/cite_latent.csv", index_col=0, header=0)
        stvea.cite_latent = stvea.cite_latent.apply(pd.to_numeric)
        Cluster.Cluster().parameter_scan(stvea, list(range(5, 21, 4)), list(range(10, 41, 3)))

    def test_consensus_cluster(self):
        stvea = STvEA.STvEA()
        stvea.cite_latent = pd.read_csv("../Data/cite_latent.csv", index_col=0, header=0)
        stvea.cite_latent = stvea.cite_latent.apply(pd.to_numeric)
        Cluster.Cluster().parameter_scan(stvea, list(range(5, 21, 4)), list(range(10, 41, 3)))
        Cluster.Cluster().consensus_cluster(stvea, 0.114, 0.1, 10)


