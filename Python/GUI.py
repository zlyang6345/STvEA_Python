import dash as ds
from dash import dash_table as dstb
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd


class GUI:
    def __init__(self):
        pass


    @staticmethod
    def cite_annotation_input(stvea):
        stvea.cite_cluster.index = stvea.cite_emb.index

        # initialize an app
        app = ds.Dash(__name__)
        df = stvea.cite_emb

        # Create a basic layout that includes a graph for displaying the clusters
        # and an interactive table for displaying the features of the clicked cluster
        app.layout = html.Div([
            dcc.Graph(id='graph', config={'clickmode': 'event+select'}),
            html.Div(id='cluster-info'),
        ])

        @app.callback(
            Output('graph', 'figure'),
            [Input('graph', 'id')]
        )
        def update_graph(_):
            fig = go.Figure()

            for cluster_id in set(stvea.cite_cluster):
                df_cluster = df[stvea.cite_cluster == cluster_id]
                fig.add_trace(go.Scatter(
                    x=df_cluster['x'], y=df_cluster['y'], mode='markers',
                    marker=dict(size=10), name=f'Cluster {cluster_id}'
                ))
            return fig

        @app.callback(
            Output('cluster-info', 'children'),
            [Input('graph', 'clickData')]
        )
        def display_cluster_info(clickData):
            if clickData is None:
                return 'Click on a cluster to display its details'
            else:
                # Extract cluster_id from clicked data
                cluster_id = clickData['points'][0]['curveNumber']
                df_cluster = df[stvea.cite_cluster == cluster_id]

                return dstb.DataTable(
                    data=df_cluster.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df_cluster.columns]
                )

        app.run_server(debug=False, dev_tools_hot_reload=False)
