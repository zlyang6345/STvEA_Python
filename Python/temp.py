import pandas as pd
import networkx as nx

def construct_graph(row, name, G):
    row.apply(lambda col: G.add_edge(name, col))

G = nx.Graph()
df = pd.DataFrame({"1": [1, 2, 3, 4], "2":[5, 7, 8, 10], "3": [100, 101, 102, 104]})
df.apply(lambda row: construct_graph(row, row.name, G), axis=1)
print(df)