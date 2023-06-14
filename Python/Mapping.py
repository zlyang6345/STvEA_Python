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
from scipy.spatial import KDTree
import pandas as pd
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

        cca_data = Mapping().run_cca(codex_subset, cite_subset, True)



        pass

    @staticmethod
    def find_nn_rna(ref_emb, query_emb, rna_mat, cite_index=1, k=300, eps=0):

        print("Finding neighborhoods")

        # use Pearson Correlation distance to find NN in mRNA dataset.
        if cite_index == 1:
            nn_rr = Mapping().cor_nn(data=rna_mat, k=k + 1)
            nn_qq_result = KDTree(query_emb).query(query_emb, k=k + 1, eps=eps)
            nn_qq = {"nn_idx": pd.DataFrame(nn_qq_result[1]),
                     "nn_dists": pd.DataFrame(nn_qq_result[0])}
        else:
            nn_rr_result = KDTree(ref_emb).query(ref_emb, k=k + 1, eps=eps)
            nn_rr = {"nn_idx": pd.DataFrame(nn_rr_result[1]),
                     "nn_dists": pd.DataFrame(nn_rr_result[0])}
            nn_qq = Mapping().cor_nn(data=rna_mat, k=k + 1)

        nn_rq_result = KDTree(query_emb).query(ref_emb, k=k, eps=eps)
        nn_rq = {"nn_idx": pd.DataFrame(nn_rq_result[1]),
                 "nn_dists": pd.DataFrame(nn_rq_result[0])}
        nn_qr_result = KDTree(ref_emb).query(query_emb, k=k, eps=eps)
        nn_qr = {"nn_idx": pd.DataFrame(nn_qr_result[1]),
                 "nn_dists": pd.DataFrame(nn_qr_result[0])}

        return {'nn_rr': nn_rr, 'nn_rq': nn_rq, 'nn_qr': nn_qr, 'nn_qq': nn_qq,
                'cellsr': ref_emb.index.values, 'cellsq': query_emb.index.values}

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


        # for i in range(len(query)):
        #     row = query.iloc[i, :]
        #     # calculate the Pearson correlation and then convert it into dissimilarity
        #     # each row in the query and each row in the data will be fed into pearsonr to calculate pearson correlation distance
        #     cor_dist_df = data.apply(lambda data_row: 1 - pearsonr(row, data_row)[0], axis=1)
        #
        #     # get indices of k nearest neighbors
        #     idx = cor_dist_df.argsort()[:k]
        #     neighbors.iloc[i, ] = idx
        #     distances.iloc[i, ] = cor_dist_df[idx]

        # alternative implementation may be faster, but require more RAM.
        # calculate the Pearson correlation and then convert it into dissimilarity
        # each row in the query and each row in the data will be fed into pearsonr to calculate pearson correlation distance
        # cor_dist_df = query.apply(lambda row: data.apply(lambda inner_row: 1 - pearsonr(row, inner_row)[0], axis=1), axis=1)
        # 5 min

        cor_dist_df = query.apply(lambda row: data.apply(lambda inner_row: 1 - np.corrcoef(row, inner_row)[0, 1], axis=1), axis=1)
        # 30 sec

        # get indices of k nearest neighbors
        for i in range(len(cor_dist_df)):
            row = cor_dist_df.iloc[i, :]
            idx = row.argsort()[:k]
            neighbors.iloc[i, ] = idx
            distances.iloc[i, ] = row[idx]

        # return values
        return {'nn_idx': neighbors, 'nn_dists': distances}



    @staticmethod
    def transfer_matrix(from_dataset,
                        to_dataset,
                        k=None,
                        c=0.1):
        """
        This function transfers a matrix from one dataset to another based on the CorNN function.
        :param from_dataset: A pandas dataframe, usually CITE.
        :param to_dataset: A pandas dataframe, usually CODEX.
        :param k: number of nearest neighbors to find.
        :param c: constant controls the width of the Gaussian kernel.
        :return: Transferred matrix.
        """

        if k is None:
            k = int(np.floor(len(to_dataset) * 0.002))

        # compute query knn from cor_nn
        # weight each nn based on gaussian kernel of distance
        # create weighted nn matrix as sparse matrix
        # return nn matrix
        nn_list = Mapping().cor_nn(to_dataset, from_dataset, k=k)
        nn_idx = nn_list['nn_idx']
        nn_dists_exp = np.exp(nn_list['nn_dists'] / -c)

        # row normalize the distance matrix
        nn_weights = nn_dists_exp.apply(lambda row: row/sum(row), axis=1)

        # gather entries and coords for the sparse matrix
        sparse_entries = nn_weights.flatten()
        sparse_coords = np.asarray(nn_idx)


