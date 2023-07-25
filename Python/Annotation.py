from copy import deepcopy
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import STvEA


class Annotation:
    stvea = STvEA.STvEA()

    def __init__(self, stvea):
        self.stvea = stvea

    def transfer_labels(self):
        """
        This function will show the gene expression levels for each CITE-seq cluster, ask user to input name for each cluster.
        These labels will be transferred to CODEX cells.
        """
        # show user the gene expression info of the cluster
        cluster_index = self.cluster_heatmap(1)

        # receive user annotation of clusters
        self.cluster_names(cluster_index, 1)
        cite_cluster_names_dict = self.stvea.cite_cluster_name_dict
        cite_cluster_names_dict[-1] = "Unknowns"

        # create indicator matrix of CITE cell cluster assignments
        cite_cluster_assignment = deepcopy(self.stvea.cite_cluster)
        cite_cluster_assignment = cite_cluster_assignment.astype("category")
        cite_cluster_assignment_dummies = pd.get_dummies(cite_cluster_assignment, prefix="", prefix_sep="")
        cite_cluster_assignment_dummies = cite_cluster_assignment_dummies.applymap(lambda x: 1 if x else 0)
        self.stvea.transfer_matrix.columns = cite_cluster_assignment_dummies.index

        # transfer labels from CITE to CODEX
        codex_cluster_names_dummies = self.stvea.transfer_matrix.dot(cite_cluster_assignment_dummies)
        codex_cluster_index = codex_cluster_names_dummies.apply(lambda row: int(row.idxmax()), axis=1)
        self.stvea.codex_cluster_names_transferred = pd.DataFrame(
            codex_cluster_index.apply(lambda x: self.stvea.cite_cluster_name_dict.get(int(x), "Unknowns")))

    def cluster_heatmap(self, dataset, option=2):
        """
        This function will generate a heatmap of clusters' protein expressions.
        @param dataset: 1 for CITE, 2 for CODEX
        @param option: 1 for static heatmap, 2 for interactive heatmap.
        @return cluster_index
        """
        if dataset == 1:
            # cite
            # cluster assignments for each cell
            clusters = self.stvea.cite_cluster
            # creat a dataframe
            combined_df = deepcopy(self.stvea.cite_protein)
            title = "CITE-seq "
        else:
            # codex
            clusters = self.stvea.codex_cluster
            # creat a dataframe
            combined_df = deepcopy(self.stvea.codex_protein_corrected)
            title = "CODEX "

        clusters.index = combined_df.index
        combined_df.insert(len(combined_df.columns), "Cluster", clusters)
        # group based on cluster and calculate the mean for each gene expression level within each cluster
        df_grouped = combined_df.groupby("Cluster").mean()
        if -1 in df_grouped.index:
            # -1 means no cluster assignment
            # drop -1
            df_grouped.drop(-1, inplace=True)
        # transpose
        df_grouped = df_grouped.transpose()

        if option == 1:
            # Now you can generate the heatmap
            plt.figure(figsize=(10, 10))
            sns.heatmap(df_grouped, cmap="viridis")
            plt.title(title + "Heatmap of Average Gene Expression by Cluster")
            plt.xlabel("Cluster")
            plt.ylabel("Gene")
            plt.show()
        else:
            # generate the interactive heatmap
            # convert the x label to string to avoid some strange behavior of plotly
            x = [str(index) for index in df_grouped.columns]
            fig = go.Figure(data=go.Heatmap(
                z=df_grouped.values,
                x=x,
                y=df_grouped.index,
                hoverongaps=False,
                colorbar={"title": "Average Gene Expression", "titleside": "right"}))

            # configure the graph
            fig.update_layout(title=title + 'Heatmap of Average Gene Expression by Cluster',
                              xaxis_title="Cluster",
                              yaxis_title="Gene",
                              xaxis_tickangle=-45)

            fig.show()

        return df_grouped.columns

    def cluster_names(self, cluster_index, dataset):
        """
        This function will receive user input for each cluster.
        @param cluster_index: a list contains all the cluster index.
        @param dataset: 1 for CITE, 2 for CODEX
        @return: a dictionary whose key is cluster index, and whose value is its name.
        """
        # let's get the number of clusters
        num_clusters = len(cluster_index)

        if dataset == 1:
            # CITE
            title = " CITE-seq "
        else:
            # CODEX
            title = " CODEX "

        # define layout
        layout = [[sg.Text('Enter the name for each' + title + 'cluster.')],
                  *[[sg.Text(cluster_name), sg.Input(key=str(i))]
                    for i, cluster_name in enumerate(cluster_index)],
                  [sg.Button('OK')]]

        # create window
        window = sg.Window('Enter' + title + 'Cluster Names', layout)

        # event loop and collecting user input
        cluster_names = {}
        while True:
            event, values = window.read()
            if event == 'OK':
                for i, cluster_index in enumerate(cluster_index):
                    cluster_names[cluster_index] = values[str(i)]
                break
            elif event == sg.WINDOW_CLOSED:
                break

        # clean up
        window.close()

        # store the dict in STvEA object
        if dataset == 1:
            # cite
            self.stvea.cite_cluster_name_dict = cluster_names
        else:
            # codex
            self.stvea.codex_cluster_name_dict = cluster_names
        return
