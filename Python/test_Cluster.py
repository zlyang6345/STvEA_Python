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
        cluster = Cluster.Cluster()
        cluster.cite_umap(stvea)
        stvea.cite_emb.to_csv("../Tests/python_cite_emb_from_cite_latent.csv")


        stvea = STvEA.STvEA
        data_processor = DataProcessor.DataProcessor()
        data_processor.read(stvea)
        data_processor.clean_cite(stvea)
        stvea.cite_latent = pd.DataFrame
        cluster = Cluster.Cluster()
        cluster.cite_umap(stvea)
        stvea.cite_emb.to_csv("../Tests/python_cite_emb_from_cleaned_cite_mrna.csv")





