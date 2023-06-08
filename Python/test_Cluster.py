from unittest import TestCase
import DataProcessor
from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd
import Cluster

import STvEA


class TestCluster(TestCase):

    def test_codex_umap(self):
        stvea = STvEA.STvEA()
        data_processor = DataProcessor.DataProcessor()
        data_processor.read(stvea)
        data_processor.clean_codex(stvea)
        Cluster.Cluster().codex_umap(stvea)
        fig, ax = plt.subplots(figsize=(12, 12))
        stvea.codex_emb.apply(lambda x: ax.scatter(x[0], x[1]), axis=1)
        ax.set_title("CODEX UMAP results of Python")
        plt.show()

        r_codex_df = pd.read_csv("../Tests/r_codex_umap_emb.csv", index_col=0, header=0)
        r_codex_df = r_codex_df.apply(pd.to_numeric)
        fig, ax = plt.subplots(figsize=(12, 12))
        r_codex_df.apply(lambda  x: ax.scatter(x[0], x[1]), axis=1)
        ax.set_title("CODEX UMAP results of R")
        plt.show()


    def test_cite_umap(self):
        stvea = STvEA.STvEA()
        stvea.cite_latent = pd.read_csv("../Data/cite_latent.csv", index_col=0, header=0)
        stvea.cite_latent = stvea.cite_latent.apply(pd.to_numeric)
        Cluster.Cluster().cite_umap(stvea)
        stvea.cite_emb.to_csv("../Tests/python_cite_emb_from_cite_latent.csv")

        # stvea = STvEA.STvEA
        # data_processor = DataProcessor.DataProcessor()
        # data_processor.read(stvea)
        # data_processor.clean_cite(stvea)
        # stvea.cite_latent = pd.DataFrame
        # Cluster.Cluster().cite_umap(stvea)
        # stvea.cite_emb.to_csv("../Tests/python_cite_emb_from_cite_mrna.csv")

        python_umap_emb = stvea.cite_emb

        fig, ax = plt.subplots(figsize=(12, 12))
        python_umap_emb.apply(lambda x: ax.scatter(x[0], x[1]), axis=1)
        ax.set_title("CITE-seq UMAP results of Python")
        plt.show()

        python_umap_emb = pd.read_csv("R_cite_emb_from_cite_latent.csv", index_col=0, header=0)
        python_umap_emb = python_umap_emb.apply(pd.to_numeric)

        fig, ax = plt.subplots(figsize=(12, 12))
        python_umap_emb.apply(lambda x: ax.scatter(x[0], x[1]), axis=1)
        ax.set_title("CITE-seq UMAP results of R")

        plt.show()

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
        Cluster.Cluster().cite_umap(stvea)

        x = stvea.cite_emb.iloc[:, 0]
        y = stvea.cite_emb.iloc[:, 1]

        # Extract the cluster labels
        clusters = stvea.cite_cluster

        # Create a dataframe for easier plotting
        plot_df = pd.DataFrame({"x": x, "y": y, "Cluster": clusters})
        not_minus_one = (plot_df.loc[:, "Cluster"] != -1)
        plot_df = plot_df.loc[not_minus_one, :]

        # Create the plot using seaborn for automatic color coding per cluster
        plt.figure(figsize=(12, 12))
        sns.scatterplot(data=plot_df, x="x", y="y", hue="Cluster", palette="deep", s=60)
        plt.title("Python CITE Clusters")
        plt.show()

        # plot R clusters
        r_cite_cluster_df = pd.read_csv("../Tests/R_cite_clusters.csv", index_col=0, header=0)
        r_cite_cluster_df.apply(pd.to_numeric)
        r_cite_cluster_df.rename(columns={"V1": "x", "V2": "y", "stvea_object@cite_clusters": "Cluster"}, inplace=True)

        not_minus_one = (r_cite_cluster_df.loc[:, "Cluster"] != -1)
        r_cite_cluster_df = r_cite_cluster_df.loc[not_minus_one, :]
        plt.figure(figsize=(12, 12))
        sns.scatterplot(data=r_cite_cluster_df, x="x", y="y", hue="Cluster", palette="deep", s=60)
        plt.title("R CITE Clusters")
        plt.show()








