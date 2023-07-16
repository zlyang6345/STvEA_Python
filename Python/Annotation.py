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
        cluster_index = stvea.cite_cluster_heatmap(stvea)
        cite_cluster_names_dict = stvea.cite_cluster_names(cluster_index)
        cite_cluster_names_dict[-1] = "No Assignment"
        cite_cluster_assignment = deepcopy(stvea.cite_cluster[["Cluster"]])
        cite_cluster_assignment_dummies = pd.get_dummies(cite_cluster_assignment, dtype=int)
        stvea.transfer_matrix.columns = cite_cluster_assignment_dummies.index
        codex_cluster_names_dummies = stvea.transfer_matrix.dot(cite_cluster_assignment_dummies)
        codex_cluster_names_dummies.rename(columns=cite_cluster_names_dict, inplace=True)
        codex_cluster_names = codex_cluster_names_dummies.apply(lambda row: row.idxmax(), axis=1)





    @staticmethod
    def cite_cluster_heatmap(stvea, option=2):
        """
        This function will generate a heatmap to help user manually annotate CITE-seq clusters.
        @param stvea: a STvEA object.
        @param option: 1 for static heatmap, 2 for interactive heatmap.
        @return cluster_index
        """
        # cluster assignments for each cell
        clusters = stvea.cite_cluster.iloc[:, -1]
        # creat a combined dataframe
        combined_df = deepcopy(stvea.cite_protein)
        combined_df.insert(len(combined_df.columns), "Cluster", clusters)
        # group based on cluster and calculate the mean for each gene expression level within each cluster
        df_grouped = combined_df.groupby("Cluster").mean()
        df_grouped.drop(-1, inplace=True)
        # transpose
        df_grouped = df_grouped.transpose()

        if option == 1:
            # Now you can generate the heatmap
            plt.figure(figsize=(10, 10))
            sns.heatmap(df_grouped, cmap="viridis")
            plt.title("Heatmap of Average Gene Expression by Cluster")
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

            fig.update_layout(title='Heatmap of Average Gene Expression by Cluster',
                              xaxis_title="Cluster",
                              yaxis_title="Gene",
                              xaxis_tickangle=-45)

            # modify the x-axis
            fig.update_xaxes(
                tickmode='array',
                tickvals=[i + 1 for i in range(len(df_grouped.columns))],  # x position
                ticktext=df_grouped.columns.values  # tick name
            )

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


