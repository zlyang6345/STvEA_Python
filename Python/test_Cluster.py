from copy import deepcopy
from unittest import TestCase
import DataProcessor
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import Cluster
import STvEA
import Annotation


class TestCluster(TestCase):

    def test_improve_tpr(self):
        stvea = STvEA.STvEA()
        cluster = Cluster.Cluster(stvea)
        annotation = Annotation.Annotation(stvea)
        stvea.codex_protein_corrected = pd.read_csv("../Tests/ToImproveTPR/codex_protein_corrected.csv", index_col=0, header=0).astype("float64")
        stvea.codex_cluster_names_transferred = pd.read_csv("../Tests/ToImproveTPR/codex_cluster_names_transferred.csv", index_col=0, header=0).fillna("")
        stvea.codex_cluster_names_transferred.columns = stvea.codex_cluster_names_transferred.columns.astype(int)
        stvea.codex_protein = pd.read_csv("../Tests/ToImproveTPR/codex_protein.csv", index_col=0, header=0).astype("float64")
        # cluster CODEX cells
        cluster.cluster_codex(k=4, knn_option=4)

        # show the CODEX protein expression level
        cluster_index = annotation.cluster_heatmap(2, 2)
        # user input CODEX cluster names
        annotation.cluster_names(cluster_index, 2, option=2)

        codex_clusters = deepcopy(stvea.codex_cluster)
        codex_clusters_names = codex_clusters.applymap(lambda x: stvea.codex_cluster_name_dict.get(x))
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
            type_cells_reality = combined[reality] == type
            non_type_cells_reality = ~type_cells_reality
            type_cells_test = combined[test] == type
            non_type_cells_test = ~type_cells_test

            # true positive rate
            tpr = (type_cells_reality & type_cells_test).sum() / type_cells_reality.sum()
            # true negative rate
            tnr = (non_type_cells_reality & non_type_cells_test).sum() / non_type_cells_reality.sum()

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
            subset = stvea.codex_cluster_names_transferred.loc[index, "0"]
            subset_value_count = subset.value_counts()
            transferred_majority = subset_value_count.idxmax()
            count_sum = subset_value_count.sum()
            subset_value_percent = subset_value_count / count_sum
            print(f"CODEX cluster {each} transferred majority: {round(subset_value_percent[transferred_majority] * 100, 3)} % {transferred_majority}")


    def test_cluster_codex(self):

        stvea = STvEA.STvEA()
        data_processor = DataProcessor.DataProcessor(stvea)
        cl = Cluster.Cluster(stvea)

        data_processor.read_codex()
        data_processor.read_cite()
        # data_processor.filter_codex()
        data_processor.clean_codex()
        cl.codex_umap()
        cl.cluster_codex(knn_option=4)

        # plot python
        plot_df = pd.DataFrame({"x": stvea.codex_emb[0], "y": stvea.codex_emb[1],
                                "Clusters": stvea.codex_cluster[0]})
        plt.figure(figsize=(12, 12))
        sns.scatterplot(data=plot_df, x="x", y="y", hue="Clusters", palette="deep", s=60)
        plt.title("Python CODEX Clusters 1")
        plt.show()

        # plot R
        r_codex_df = pd.read_csv("../Tests/r_codex_clusters.csv", index_col=0, header=0)
        r_codex_df = r_codex_df.apply(pd.to_numeric)
        plt.figure(figsize=(12, 12))
        sns.scatterplot(data=r_codex_df, x="x", y="y", hue="Clusters", palette="deep", s=60)
        plt.title("R CODEX Clusters")
        plt.show()

    def test_codex_umap(self):
        stvea = STvEA.STvEA()
        data_processor = DataProcessor.DataProcessor(stvea)
        cl = Cluster.Cluster(stvea)

        data_processor.read_codex()
        data_processor.read_cite()
        data_processor.clean_codex()
        cl.codex_umap()
        fig, ax = plt.subplots(figsize=(12, 12))
        stvea.codex_emb.apply(lambda x: ax.scatter(x[0], x[1]), axis=1)
        ax.set_title("CODEX UMAP results of Python")
        plt.show()

        r_codex_df = pd.read_csv("../Tests/r_codex_umap_emb.csv", index_col=0, header=0)
        r_codex_df = r_codex_df.apply(pd.to_numeric)
        fig, ax = plt.subplots(figsize=(12, 12))
        r_codex_df.apply(lambda x: ax.scatter(x[0], x[1]), axis=1)
        ax.set_title("CODEX UMAP results of R")
        plt.show()

    def test_cite_umap(self):
        stvea = STvEA.STvEA()
        cl = Cluster.Cluster(stvea)
        stvea.cite_latent = pd.read_csv("../Data/small_dataset/cite_latent.csv", index_col=0, header=0)
        stvea.cite_latent = stvea.cite_latent.apply(pd.to_numeric)
        cl.cite_umap()
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

        python_umap_emb = pd.read_csv("../Tests/R_cite_emb_from_cite_latent.csv", index_col=0, header=0)
        python_umap_emb = python_umap_emb.apply(pd.to_numeric)

        fig, ax = plt.subplots(figsize=(12, 12))
        python_umap_emb.apply(lambda x: ax.scatter(x[0], x[1]), axis=1)
        ax.set_title("CITE-seq UMAP results of R")

        plt.show()

    def test_parameter_scan(self):
        stvea = STvEA.STvEA()
        cl = Cluster.Cluster(stvea)

        stvea.cite_latent = pd.read_csv("../Data/small_dataset/cite_latent.csv", index_col=0, header=0)
        stvea.cite_latent = stvea.cite_latent.apply(pd.to_numeric)
        cl.parameter_scan(stvea, list(range(5, 21, 4)), list(range(10, 41, 3)))

    def test_consensus_cluster(self):

        stvea = STvEA.STvEA()
        cl = Cluster.Cluster(stvea)

        stvea.cite_latent = pd.read_csv("../Data/small_dataset/cite_latent.csv", index_col=0, header=0)
        stvea.cite_latent = stvea.cite_latent.apply(pd.to_numeric)
        cl.parameter_scan(list(range(5, 21, 4)), list(range(10, 41, 3)))
        cl.consensus_cluster(0.114, 0.1, 10)
        cl.cite_umap()

        x = stvea.cite_emb.iloc[:, 0]
        y = stvea.cite_emb.iloc[:, 1]

        # Extract the cluster labels
        clusters = stvea.cite_cluster

        # Create a dataframe for easier plotting
        plot_df = pd.DataFrame({"x": x, "y": y, "Cluster": clusters.iloc[:, 0]})
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
