import math
import warnings

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

from Python.irlb import irlb
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix


class Mapping:
    def __init__(self):
        pass

    @staticmethod
    def run_cca(object1, object2, standardize=True, num_cc=30, option=1):
        """
        This function will use CCA to reduce dimensions.

        @param object1: A dataframe whose rows represent cells.
        @param object2: A dataframe whose rows represent cells.
        @param standardize: a boolean value. If true, two dataframes would be standardized column-wise.
        @param num_cc: The number of dimensions after reduction.
        @param option: a integer value to specify the way to perform SVD. 1 for irlb method. 2 for Scikit-learn svds.
        @return: a dataframe that combines the two reduced dataframes.
        """
        cells1 = object1.columns
        cells2 = object2.columns

        if standardize:
            # standardize each cell
            object1 = object1.apply(lambda col: (col - np.mean(col)) / np.std(col, ddof=1), axis=0)
            object2 = object2.apply(lambda col: (col - np.mean(col)) / np.std(col, ddof=1), axis=0)

        # perform the dot production
        mat3 = np.dot(object1.T, object2)

        if option == 1:
            # irlb method
            tuple = irlb(mat3, n=num_cc, tol=1e-05, maxit=1000)
            u = tuple[0]
            v = tuple[2]
        else:
            # scikit-learn method
            u, s, v = svds(mat3, k=num_cc, tol=1e-05)
            v = v.transpose()

        # combine the result
        cca_data = np.concatenate([u, v], axis=0)
        cca_data = np.array([x if np.sign(x[0]) != -1 else x * -1 for x in cca_data.T]).T
        cca_data = pd.DataFrame(cca_data, index=list(cells1) + list(cells2),
                                columns=["CC" + str(i + 1) for i in range(num_cc)])

        return cca_data

    @staticmethod
    def map_codex_to_cite(stvea):
        # find common proteins
        common_protein = [protein for protein in stvea.codex_protein.columns if protein in stvea.cite_protein.columns]

        if (len(common_protein) < 2):
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
    def find_anchor_pairs(neighbors, k_anchor=5):
        """
        This function will find anchors between the reference and query dataset.
        @param neighbors: A dictionary generated by find_nn_rna.
        @param k_anchor: The number of neighbors to find anchors. Fewer k_anchor should mean higher quality of anchors.
        @return: A dataframe that has three columns: cell r, cell q, and score.
        The score column is initialized here but will be calculated at next step.
        """
        print("Finding mutual nearest neighborhoods.")

        # some routine check
        max_nn = max(neighbors['nn_rq']['nn_idx'].shape[1], neighbors['nn_qr']['nn_idx'].shape[1])
        if k_anchor > max_nn:
            warnings.warn('Requested k.anchor = {}, only {} in dataset.'.format(k_anchor, max_nn))
            k_anchor = min(max_nn, k_anchor)

        # obtain reference cell's index
        ncell = np.arange(neighbors['nn_rq']['nn_idx'].shape[0])

        # initialize some values
        anchors = {'cellr': np.zeros(ncell.shape[0] * k_anchor),
                   'cellq': np.zeros(ncell.shape[0] * k_anchor),
                   'score': np.ones(ncell.shape[0] * k_anchor)}

        # find anchor pairs
        idx = 0
        for cell in ncell:
            # find each reference cell's k_anchor's nearest neighbors
            neighbors_rq = neighbors['nn_rq']['nn_idx'].iloc[cell, :k_anchor]

            # find query cells that treat the current reference as their nearest neighbors
            # cells that treat each other as nearest neighbors are anchors
            mutual_neighbors = neighbors['nn_qr']['nn_idx'].iloc[neighbors_rq, :k_anchor].values
            mutual_neighbors = mutual_neighbors == cell
            mutual_neighbors = np.where(mutual_neighbors)[0]

            # record these anchors
            for i in neighbors_rq[mutual_neighbors]:
                anchors['cellr'][idx] = cell
                anchors['cellq'][idx] = i
                idx += 1

        # truncate
        anchors['cellr'] = anchors['cellr'][:idx]
        anchors['cellq'] = anchors['cellq'][:idx]
        anchors['score'] = anchors['score'][:idx]

        # convert to dataframe
        anchors = pd.DataFrame(anchors).astype("uint32")
        return anchors

    @staticmethod
    def find_nn_rna(ref_emb, query_emb, rna_mat, cite_index=1, k=300, eps=0):
        """
        This function will find nearest neighbors between reference-reference, query-query,
        reference-query, and query-reference.
        @param ref_emb: a (cell x feature) embedding of a protein expression matrix.
        @param query_emb: a (cell x feature) embedding of protein expression matrix to be corrected.
        @param rna_mat: a (cell x feature) embedding of the mRNA expression matrix from CITE-seq.
        @param cite_index: which matrix (1 or 2) is the CITE-seq protein expression matrix.
        @param k: number of nearest neighbor to find between the matrices.
        @param eps: error bound on nearest neighbor search (see eps parameter for nn2).
        @return: {'nn_rr': nn_rr, 'nn_rq': nn_rq, 'nn_qr': nn_qr, 'nn_qq': nn_qq,
                'cellsr': ref_emb.index.values, 'cellsq': query_emb.index.values}
                nn_rr and nn_qq are also dictionary that contain "nn_idx" and "nn_dists"
        """
        print("Finding neighborhoods")

        if cite_index == 1:
            # use Pearson Correlation distance to find NN in mRNA dataset.
            nn_rr = Mapping().cor_nn(data=rna_mat, k=k + 1)
            # use KDTree to find nearest neighbors among query-query.
            nn_qq_result = KDTree(query_emb).query(query_emb, k=k + 1, eps=eps)
            nn_qq = {"nn_idx": pd.DataFrame(nn_qq_result[1]),
                     "nn_dists": pd.DataFrame(nn_qq_result[0])}
        else:
            # use KDTree to find nearest neighbors among reference-reference.
            nn_rr_result = KDTree(ref_emb).query(ref_emb, k=k + 1, eps=eps)
            nn_rr = {"nn_idx": pd.DataFrame(nn_rr_result[1]),
                     "nn_dists": pd.DataFrame(nn_rr_result[0])}
            # use Pearson Correlation distance to find NN in mRNA dataset.
            nn_qq = Mapping().cor_nn(data=rna_mat, k=k + 1)

        # find nearest neighbors among query-reference.
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

        cor_dist_df = query.apply(
            lambda row: data.apply(lambda inner_row: 1 - np.corrcoef(row, inner_row)[0, 1], axis=1),
            axis=1)
        # 30 sec

        # get indices of k nearest neighbors
        for i in range(len(cor_dist_df)):
            row = cor_dist_df.iloc[i, :]
            idx = row.argsort()[:k]
            neighbors.iloc[i,] = idx
            distances.iloc[i,] = row[idx]

        # return values
        return {'nn_idx': neighbors.astype("uint32"), 'nn_dists': distances}

    @staticmethod
    def filter_anchors(ref_mat, query_mat, anchors, k_filter=200):
        """
        This function will keep anchors that preserve original data info.
        This means that the anchor in CCA space should also be anchors in original dataset.
        @param ref_mat: a cleaned protein expression dataframe.
        @param query_mat: a cleaned protein expression dataframe.
        @param anchors: a dataframe generated in the previous step.
        @param k_filter: the number of neighbors to find in the original data space.
        @return: a dataframe of filtered anchors.
        """
        print("Filtering Anchors...")

        nn1 = Mapping().cor_nn(data=query_mat, query=ref_mat, k=k_filter)
        nn2 = Mapping().cor_nn(data=ref_mat, query=query_mat, k=k_filter)

        position1 = [False] * len(anchors)
        position2 = [False] * len(anchors)
        i = 0
        for q, r in zip(anchors['cellq'], anchors['cellr']):
            position1[i] = nn1['nn_idx'].iloc[r, :].isin([q]).any()
            position2[i] = nn2['nn_idx'].iloc[q, :].isin([r]).any()
            i += 1

        anchors = anchors[np.logical_or(position1, position2)]

        print("\tRetained ", len(anchors), " anchors")
        return anchors

    @staticmethod
    def construct_nn_mat(nn_idx, offset_i, offset_j, dims):
        """
        This is a helper function to construct matrix.
        @param nn_idx: a matrix whose row represents cells and columns represent nearest neighbors to each cell.
        @param offset_i: offset_i will be added to each row index cell.
        @param offset_j: offset_j will be added to each nearest neighbor.
        @param dims: the dimension of the matrix to be created.
        @return: the constructed matrix.
        """
        # row-wise flatten nn_idx and apply offset
        j = np.array(nn_idx).flatten() + offset_j

        # calculate i with offset
        i = np.repeat(np.arange(nn_idx.shape[0]) + offset_i, nn_idx.shape[1])

        # create sparse matrix
        nn_mat = csr_matrix((np.ones_like(i), (i, j)), shape=dims)

        return nn_mat

    @staticmethod
    def score_anchors(neighbors, anchors, num_cells_ref, num_cells_query, k_score=30):
        """
        This function will calculate a score based on number of shared neighbors for each anchor.

        s_j1j2=|N_CITEseq_j1 ∩ N_CITEseq_j2| + |N_CODEX_j1 ∩ N_CODEX_j2|

        N_CITEseq_j1 is the set of nearest CITEseq cells to cell j1 in the mRNA latent space.
        N_CITEseq_j2 is the set of nearest CITEseq cells to cell j2 in the CCA space
        N_CODEX_j1 is the set of nearest CODEX cells to j1 in the CCA space.
        N_CODEX_j2 is the set of nearest CODEX cells to j2 in the CCA space.

        @param neighbors:
        @param anchors:
        @param num_cells_ref:
        @param num_cells_query:
        @param k_score:
        @return:
        """
        # Convert anchor data frame
        anchor_df = pd.DataFrame(anchors)
        anchor_df['cellq'] += num_cells_ref

        # Determine maximum k value
        min_nn = min(
            neighbors['nn_rr']['nn_idx'].shape[1],
            neighbors['nn_rq']['nn_idx'].shape[1],
            neighbors['nn_qr']['nn_idx'].shape[1],
            neighbors['nn_qq']['nn_idx'].shape[1]
        )

        if k_score > min_nn:
            print(f'Warning: Requested k.score = {k_score}, only {min_nn} in dataset')
            k_score = min_nn

        total_cells = num_cells_ref + num_cells_query

        # Construct nearest neighbour matrices
        nn_m1 = Mapping().construct_nn_mat(neighbors['nn_rr']['nn_idx'].iloc[:, :k_score],
                                           0, 0, (total_cells, total_cells))
        nn_m2 = Mapping().construct_nn_mat(neighbors['nn_rq']['nn_idx'].iloc[:, :k_score],
                                           0, num_cells_ref, (total_cells, total_cells))
        nn_m3 = Mapping().construct_nn_mat(neighbors['nn_qr']['nn_idx'].iloc[:, :k_score],
                                           num_cells_ref, 0, (total_cells, total_cells))
        nn_m4 = Mapping().construct_nn_mat(neighbors['nn_qq']['nn_idx'].iloc[:, :k_score],
                                           num_cells_ref, num_cells_ref, (total_cells, total_cells))

        # Combine all matrices
        k_matrix = nn_m1 + nn_m2 + nn_m3 + nn_m4

        # Create sparse matrix with anchor cells
        anchor_only = coo_matrix((np.ones(len(anchor_df)), (anchor_df.iloc[:, 0], anchor_df.iloc[:, 1])),
                                 shape=(total_cells, total_cells))

        # Calculate Jaccard similarity
        jaccard_dist = k_matrix.dot(k_matrix.transpose())

        # Element-wise multiplication
        anchor_matrix = jaccard_dist.multiply(anchor_only).tocoo()

        # Create new anchor matrix
        anchor_new = pd.DataFrame({
            'cellr': anchor_matrix.row,
            'cellq': anchor_matrix.col,
            'score': anchor_matrix.data
        })

        # Rescale scores
        anchor_new['cellq'] -= num_cells_ref
        max_score, min_score = anchor_new['score'].quantile((0.9, 0.01))
        anchor_new['score'] = (anchor_new['score'] - min_score) / (max_score - min_score)
        anchor_new['score'] = anchor_new['score'].clip(0, 1)
        anchors.sort_values("cellq", ascending=True, inplace=True)
        return anchor_new

    @staticmethod
    def find_integration_matrix(ref_mat, query_mat, neighbors, anchors):
        """
        Calculate anchor vectors between reference and query dataset.

        Parameters:
        ref_mat (pd.DataFrame): a (cell x feature) protein expression matrix
        query_mat (pd.DataFrame): a (cell x feature) protein expression matrix to be corrected
        neighbors (dict): a list of neighbors
        anchors (pd.DataFrame): a list of anchors (MNN) as from FindAnchorPairs

        Returns:
        pd.DataFrame: integration matrix
        """

        print("Finding integration vectors")

        # Extract cell expression proteins
        data_use_r = ref_mat.iloc[anchors["cellr"]].reset_index(drop=True)
        data_use_q = query_mat.iloc[anchors["cellq"]].reset_index(drop=True)

        # Subtract the data frames to obtain the integration matrix
        integration_matrix = data_use_q.subtract(data_use_r)

        # Set the row names (index) to anchors_q
        integration_matrix.index = neighbors["cellsq"][anchors["cellq"]]

        return integration_matrix

    @staticmethod
    def find_weights(neighbors, anchors, query_mat, k_weight=300, sd_weight=1):
        """
        This function will find weights for anchors.
        This weight is based on the distance of query cell and anchor distance.
        @param neighbors: a dictionary generated in previous step.
        @param anchors: a dataframe that includes three columns (cellq, cellr, and score).
        @param query_mat: a dataframe whose row represents query cell and whose column represents protein.
        @param k_weight: the number of nearest anchors to use in correction.
        @param sd_weight: standard deviation of the Gaussian kernel.
        @return: a dataframe whose row represents query cell and column represents anchors.
        """
        # print a message.
        print("Finding anchors weights.")

        # initialize some variables
        cellsr = neighbors["cellsr"]
        cellsq = neighbors["cellsq"]
        anchor_cellsq = anchors["cellq"]

        # find nearest anchors to each query cell
        kna_query = Mapping().cor_nn(data=query_mat.iloc[anchor_cellsq, :], query=query_mat, k=k_weight)

        nn_dists = kna_query["nn_dists"]
        nn_dists.index = cellsq
        nn_idx = kna_query["nn_idx"]
        nn_idx.index = cellsq

        # divide each entry by that cell's kth nearest neighbor's distance.
        nn_dists = 1 - nn_dists.div(nn_dists.iloc[:, k_weight - 1], axis=0)

        # initialize a dataframe.
        dists_weights = pd.DataFrame(data=0, index=cellsq, columns=range(len(anchor_cellsq)))

        # define a helper function
        def helper(row, index):
            idx = nn_idx.loc[index]
            row[idx] = nn_dists.loc[index]
            return row

        # apply the helper function to each row.
        # each row in the dataset represents a query cell.
        # each column represents the anchor cell.
        dists_weights = dists_weights.apply(lambda row: helper(row, row.name), axis=1)

        # create a series for anchor scores.
        scores = pd.Series(anchors["score"])
        scores.index = dists_weights.columns

        # multiply each row of the dataset with its corresponding score.
        weights = dists_weights.mul(scores, axis=1)

        # calculate the Gaussian kernel function.
        weights = weights.apply(lambda x: 1 - np.exp(-1 * x / (2 * (1 / sd_weight)) ** 2))

        # normalize by row
        weights = weights.div(weights.sum(axis=1), axis=0)

        return weights

    @staticmethod
    def transform_data_matrix(query_mat, integration_matrix, weights, stvea):
        """
        This function will generate the corrected protein expression matrix.
        @param stvea: a STvEA object.
        @param query_mat: a (cell x feature) protein expression matrix to be corrected.
        @param integration_matrix: matrix of anchor vectors (output of find_integration_matrix).
        @param weights: weights of the anchors of each query cell.
        @return: a corrected query cell protein expression matrix.
        """
        print("Integrating data")
        integration_matrix.index = weights.columns
        bv = weights.dot(integration_matrix)
        bv.index = query_mat.index
        integrated = query_mat - bv
        stvea.codex_protein_corrected = integrated
        return

    @staticmethod
    def transfer_matrix(stvea,
                        k=None,
                        c=0.1):
        """
        This function transfers a matrix from one dataset to another based on the CorNN function.
        :param stvea: a STvEA object.
        :param k: number of nearest neighbors to find.
        :param c: constant controls the width of the Gaussian kernel.
        :return: Transferred matrix.
        """

        from_dataset = stvea.cite_protein
        to_dataset = stvea.codex_protein_corrected

        if k is None:
            k = int(np.floor(len(to_dataset) * 0.002))

        # compute query knn from cor_nn
        # weight each nn based on gaussian kernel of distance
        # create weighted nn matrix as sparse matrix
        # return nn matrix
        nn_list = Mapping().cor_nn(from_dataset, to_dataset, k=k)
        nn_idx = nn_list['nn_idx']
        nn_dists_exp = np.exp(nn_list['nn_dists'] / -c)

        # row normalize the distance matrix
        nn_weights = nn_dists_exp.apply(lambda row: row / sum(row), axis=1)

        # gather entries and coords for the sparse matrix
        idx_array = nn_idx.to_numpy()
        weights_array = nn_weights.to_numpy()

        # Flatten arrays and create coordinate pairs
        rows = np.repeat(np.arange(idx_array.shape[0]), idx_array.shape[1])
        cols = idx_array.flatten()
        data = weights_array.flatten()

        # Now, create a sparse matrix
        transfer_matrix = coo_matrix((data, (rows, cols)))

        # Convert to CSR format for efficient arithmetic and matrix operations
        stvea.transfer_matrix = pd.DataFrame(transfer_matrix.todense())

        stvea.transfer_matrix.index = from_dataset.index
        stvea.transfer_matrix.columns = to_dataset.index

        return
