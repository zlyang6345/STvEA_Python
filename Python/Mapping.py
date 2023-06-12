import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler, scale, normalize
from sklearn.preprocessing import StandardScaler
import numpy as np
from Python.irlb import irlb
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

class Mapping:
    def __init__(self):
        pass

    @staticmethod
    def run_cca(object1, object2, standardize=True, num_cc=30):

        cells1 = object1.columns
        cells2 = object2.columns

        if standardize:
            # standardize each cell
            object1 = object1.apply(lambda col: (col - np.mean(col)) / np.std(col, ddof=1), axis=0)
            object2 = object2.apply(lambda col: (col - np.mean(col)) / np.std(col, ddof=1), axis=0)


        mat3 = np.dot(object1.T, object2)

        # u, s, v = svds(mat3, k=num_cc, tol=1e-05)
        tuple = irlb(mat3, n=num_cc, tol=1e-05, maxit=1000)
        u = tuple[0]
        v = tuple[2]

        cca_data = np.concatenate([u, v], axis=0)

        cca_data = np.array([x if np.sign(x[0]) != -1 else x * -1 for x in cca_data.T]).T

        cca_data = pd.DataFrame(cca_data, index=list(cells1) + list(cells2),
                                columns=["CC" + str(i + 1) for i in range(num_cc)])

        return cca_data

    @staticmethod
    def map_codex_to_cite(stvea):
        # find common proteins
        common_protein = [protein for protein in stvea.codex_protein.columns if protein in stvea.cite_protein.columns]

        if(len(common_protein) < 2):
            # for STvEA to properly transfer value from CODEX to CITE.
            # enough proteins are required.
            print("Too few common proteins between CODEX proteins and CITE-seq proteins")
            exit(1)

        # select common protein columns
        codex_subset = stvea.codex_protein.loc[:, common_protein]
        cite_subset = stvea.cite_protein.loc[:, common_protein]

        codex_cca, cite_cca = Mapping().run_cca(codex_subset, cite_subset, True)

        pass

    @staticmethod
    def cor_nn(data, query=None, k=5):
        """
        This function can find nearest neighbors (rows) in "data" dataset for each record (row) in "query" dataset.
        :param data: A pandas dataframe.
        :param query: A pandas dataframe.
        :param k: the number of nearest neighbors.
        :return: {'nn_idx': neighbors, 'nn_dists': distances}
        """
        if query is None:
            query = data

        # make sure the input is a dataframe
        query = pd.DataFrame(query)
        data = pd.DataFrame(data)

        # initialize neighbors and distances matrices
        neighbors = pd.DataFrame(index=range(len(query)), columns=range(k), dtype='uint32')
        distances = pd.DataFrame(index=range(len(query)), columns=range(k), dtype='float64')

        for i, row in query.iterrows():
            # calculate the Pearson correlation and then convert it into dissimilarity
            # each row in the query and each row in the data will be fed into pearsonr to calculate pearson correlation distance
            cor_dist_df = data.apply(lambda data_row: 1 - pearsonr(row, data_row)[0], axis=1)

            # get indices of k nearest neighbors
            idx = cor_dist_df.argsort()[:k]
            neighbors.iloc[i, ] = idx
            distances.iloc[i, ] = cor_dist_df[idx]

        # alternative implementation may be faster, but require more RAM.
        # calculate the Pearson correlation and then convert it into dissimilarity
        # each row in the query and each row in the data will be fed into pearsonr to calculate pearson correlation distance
        # cor_dist_df = query.apply(lambda row: data.apply(lambda inner_row: 1 - pearsonr(row, inner_row)[0],
        #                                                  axis=1)
        #                           , axis=1)
        #
        # # get indices of k nearest neighbors
        # for i, row in cor_dist_df.iterrows():
        #     idx = row.argsort()[:k]
        #     neighbors.iloc[i, ] = idx
        #     distances.iloc[i, ] = row[idx]

        # return values
        return {'nn_idx': neighbors, 'nn_dists': distances}

    @staticmethod
    def transfer_matrix(stvea):











        pass