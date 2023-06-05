import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

python_umap_emb = pd.read_csv("python_cite_emb_from_cite_latent.csv", index_col=0, header=0)
python_umap_emb = python_umap_emb.apply(pd.to_numeric)

fig, ax = plt.subplots(figsize=(12, 12))
python_umap_emb.apply(lambda x: ax.scatter(x[0], x[1]), axis = 1)
ax.set_title("CITE-seq UMAP results of Python")

plt.show()


python_umap_emb = pd.read_csv("R_cite_emb.csv", index_col=0, header=0)
python_umap_emb = python_umap_emb.apply(pd.to_numeric)

fig, ax = plt.subplots(figsize=(12, 12))
python_umap_emb.apply(lambda x: ax.scatter(x[0], x[1]), axis = 1)
ax.set_title("CITE-seq UMAP results of R")

plt.show()