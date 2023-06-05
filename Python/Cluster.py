import numpy as np
import umap.umap_ as umap
import pandas as pd
from scipy import stats


def pearson(a, b):
    # scipy's pearsonr returns a tuple (correlation, pvalue), so take the first element
    return 1 - stats.pearsonr(a, b)[0]

class Cluster:

    def __init__(self):
        pass


    def cite_umap(self, stvea, metric=pearson, n_neighbors=50, min_dist=0.1, negative_sample_rate=50):

        if stvea.cite_latent.empty and stvea.cite_mRNA.empty:
            raise ValueError("stvea_object does not contain CITE-seq data")

        if not stvea.cite_latent.empty:
            # recommended
            res = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate, n_components=2).fit_transform(stvea.cite_latent)
            stvea.cite_emb = pd.DataFrame(res)

        elif not stvea.cite_mRNA.empty:
            # implemented, but not recommended
            res = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist,
                            negative_sample_rate=negative_sample_rate, n_components=2).fit_transform(stvea.cite_mRNA)
            stvea.cite_emb = pd.DataFrame(res)

        return