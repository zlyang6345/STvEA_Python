from copy import deepcopy
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import PySimpleGUI as sg


class Annotation:
    def __init__(self):
        pass

    @staticmethod
    def transfer_labels(stvea):
        """
        This function will transfer labels.
        @param stvea: a STvEA object.
        """
        # show user the gene expression info of the cluster
        cluster_index = Annotation().cite_cluster_heatmap(stvea)

        # receive user annotation of clusters
        cite_cluster_names_dict = Annotation().cite_cluster_names(cluster_index)
        cite_cluster_names_dict[-1] = "No Assignment"

        # create indicator matrix of CITE cell cluster assignments
        cite_cluster_assignment = deepcopy(stvea.cite_cluster)
        cite_cluster_assignment = cite_cluster_assignment.astype("category")
        cite_cluster_assignment_dummies = pd.get_dummies(cite_cluster_assignment, prefix="", prefix_sep="")
        cite_cluster_assignment_dummies = cite_cluster_assignment_dummies.applymap(lambda x: 1 if x else 0)
        stvea.transfer_matrix.columns = cite_cluster_assignment_dummies.index

        # transfer labels from CITE to CODEX
        codex_cluster_names_dummies = stvea.transfer_matrix.dot(cite_cluster_assignment_dummies)
        codex_cluster_names_dummies.rename(columns=cite_cluster_names_dict, inplace=True)
        stvea.codex = codex_cluster_names_dummies.apply(lambda row: row.idxmax(), axis=1)



    @staticmethod
    def cluster_heatmap(stvea, dataset, option=2):
        """
        This function will generate a heatmap of clusters' protein expressions.
        @param stvea: a STvEA object.
        @param dataset: 1 for CITE, 2 for CODEX
        @param option: 1 for static heatmap, 2 for interactive heatmap.
        @return cluster_index
        """
        if dataset == 1:
            # cite
            # cluster assignments for each cell
            clusters = stvea.cite_cluster
            # creat a combined dataframe
            combined_df = deepcopy(stvea.cite_protein)
            title = "CITE-seq "
        else:
            # codex
            clusters = stvea.codex_cluster
            # creat a combined dataframe
            combined_df = deepcopy(stvea.codex_protein_corrected)
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
            fig = go.Figure(data=go.Heatmap(
                z=df_grouped.values,
                x=df_grouped.columns,
                y=df_grouped.index,
                hoverongaps=False,
                colorbar={"title": "Average Gene Expression", "titleside": "right"}))

            # configure the graph
            fig.update_layout(title='Heatmap of Average Gene Expression by Cluster',
                              xaxis_title="Cluster",
                              yaxis_title="Gene",
                              xaxis_tickangle=-45)

            fig.show()

        return df_grouped.columns

    @staticmethod
    def cite_cluster_names(cluster_index):
        """
        This function will receive user input for each cluster.
        @param cluster_index: a list contains all the cluster index.
        @return: a dictionary whose key is cluster index, and whose value is its name.
        """
        # let's get the number of clusters
        num_clusters = len(cluster_index)

        # define layout
        layout = [[sg.Text('Enter the name for each cluster')],
                  *[[sg.Text(cluster_name), sg.Input(key=str(i))]
                    for i, cluster_name in enumerate(cluster_index)],
                  [sg.Button('OK')]]

        # create window
        window = sg.Window('Enter Cluster Names', layout)

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

        return cluster_names


