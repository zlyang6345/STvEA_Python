import time
import warnings
from math import ceil
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from Python.irlb import irlb
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors


class Mapping:
    stvea = None

    def __init__(self, stvea):
        self.stvea = stvea

    @staticmethod
    def run_cca(object1, object2, standardize=True, num_cc=30, option=1, random_state=0):
        """
        This function will use CCA to reduce dimensions.

        @param random_state: an integer to specify the random state.
        @param object1: A dataframe whose rows represent cells.
        @param object2: A dataframe whose rows represent cells.
        @param standardize: a boolean value. If true, two dataframes would be standardized column-wise.
        @param num_cc: The number of dimensions after reduction.
        @param option: an integer value to specify the way to perform SVD. 1 for irlb method. 2 for Scikit-learn svds.
        @return: a dataframe that combines the two reduced dataframes.
        """
        start = time.time()
        cells1 = object1.columns
        cells2 = object2.columns

        if standardize:
            # standardize each cell
            object1 = object1.apply(lambda col: (col - np.mean(col)) / np.std(col, ddof=1), axis=0)
            object2 = object2.apply(lambda col: (col - np.mean(col)) / np.std(col, ddof=1), axis=0)

        # perform the dot production
        mat3 = np.dot(object1.T, object2)

        if option == 1:
            # IRLB method
            tuples = irlb(mat3, n=num_cc, tol=1e-05, maxit=1000, random_state=random_state)
            u = tuples[0]
            v = tuples[2]
        else:
            # scikit-learn method
            u, s, v = svds(mat3, k=num_cc, tol=1e-05, random_state=random_state)
            v = v.transpose()

        # combine the result
        cca_data = np.concatenate([u, v], axis=0)
        cca_data = np.array([x if np.sign(x[0]) != -1 else x * -1 for x in cca_data.T]).T
        cca_data = pd.DataFrame(cca_data, index=list(cells1) + list(cells2),
                                columns=["CC" + str(i + 1) for i in range(num_cc)])

        end = time.time()
        print(f"run_cca. Time: {round(end - start, 3)} sec")
        return cca_data

    @staticmethod
    def find_anchor_pairs(neighbors, k_anchor=5):
        """
        This function will find anchors between the reference and query dataset.
        The score column is initialized here but will be calculated at the next step.
        @param neighbors: A dictionary generated by find_nn_rna.
        @param k_anchor: The number of neighbors to find anchors.
            Fewer k_anchor should mean higher quality of anchors.
        @return: A dataframe that has three columns: cell r, cell q, and score.
        """
        start = time.time()

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
            # cells that treat each other as the nearest neighbors are anchors
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
        end = time.time()
        print(f"find_anchor_pairs Time: {round(end - start, 3)} sec")
        return anchors

    @staticmethod
    def find_nn_rna(ref_emb, query_emb, rna_mat, cite_index=1, k=300, eps=0, nn_option=2):
        """
        This function will find the nearest neighbors between reference-reference, query-query,
        reference-query, and query-reference.
        @param ref_emb: a (cell x feature) embedding of a protein expression matrix.
        @param query_emb: a (cell x feature) embedding of protein expression matrix to be corrected.
        @param rna_mat: a (cell x feature) embedding of the mRNA expression matrix from CITE-seq.
        @param cite_index: which matrix (1 or 2) is the CITE-seq protein expression matrix?
        @param k: number of the nearest neighbor to find between the matrices.
        @param eps: error bound on nearest neighbor search (see eps parameter for nn2).
        @return: {'nn_rr': nn_rr, 'nn_rq': nn_rq, 'nn_qr': nn_qr, 'nn_qq': nn_qq,
                'cellsr': ref_emb.index.values, 'cellsq': query_emb.index.values}
                nn_rr and nn_qq are also dictionary that contain "nn_idx" and "nn_dists"
        """
        nn_rr = dict()
        nn_qq = dict()
        start = time.time()
        if cite_index == 1:
            if nn_option == 1:
                # use Pearson Correlation distance to find NN in mRNA dataset.
                nn_rr = Mapping.cor_nn(data=rna_mat, k=k + 1)
            else:
                nn = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric="correlation")
                nn.fit(rna_mat)
                nn_dists, nn_idx = nn.kneighbors(rna_mat)
                nn_dists = pd.DataFrame(nn_dists, index=rna_mat.index)
                nn_idx = pd.DataFrame(nn_idx, index=rna_mat.index)
                nn_rr["nn_idx"] = nn_idx
                nn_rr["nn_dists"] = nn_dists

            # use KDTree to find the nearest neighbors among query-query.
            nn_qq_result = KDTree(query_emb).query(query_emb, k=k+1, eps=eps)
            nn_qq = {"nn_idx": pd.DataFrame(nn_qq_result[1]),
                     "nn_dists": pd.DataFrame(nn_qq_result[0])}
        else:
            # use KDTree to find the nearest neighbors among reference-reference.
            nn_rr_result = KDTree(ref_emb).query(ref_emb, k=k+1, eps=eps)
            nn_rr = {"nn_idx": pd.DataFrame(nn_rr_result[1]),
                     "nn_dists": pd.DataFrame(nn_rr_result[0])}
            # use Pearson Correlation distance to find NN in mRNA dataset.
            if nn_option == 1:
                nn_qq = Mapping.cor_nn(data=rna_mat, k=k + 1)
            else:
                nn = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric="correlation")
                nn.fit(rna_mat)
                nn_dists, nn_idx = nn.kneighbors(rna_mat)
                nn_dists = pd.DataFrame(nn_dists, index=rna_mat.index)
                nn_idx = pd.DataFrame(nn_idx, index=rna_mat.index)
                nn_qq["nn_idx"] = nn_idx
                nn_qq["nn_dists"] = nn_dists

        # find the nearest neighbors among query-reference.
        nn_rq_result = KDTree(query_emb).query(ref_emb, k=k, eps=eps)
        nn_rq = {"nn_idx": pd.DataFrame(nn_rq_result[1]),
                 "nn_dists": pd.DataFrame(nn_rq_result[0])}
        nn_qr_result = KDTree(ref_emb).query(query_emb, k=k, eps=eps)
        nn_qr = {"nn_idx": pd.DataFrame(nn_qr_result[1]),
                 "nn_dists": pd.DataFrame(nn_qr_result[0])}

        end = time.time()

        print(f"find_nn_rna Time: {round(end - start, 3)} sec")

        return {'nn_rr': nn_rr, 'nn_rq': nn_rq, 'nn_qr': nn_qr, 'nn_qq': nn_qq,
                'cellsr': ref_emb.index.values, 'cellsq': query_emb.index.values}

    @staticmethod
    def helper(query_sub, data):
        """
        A helper function to be invoked by cor_nn.
        This function will calculate each row's correlation distance with each row in another dataframe.
        @param query_sub: the query dataframe.
        @param data: a dataframe.
        @return: a length(query_sub) * length(data) dataframe.
            Each entry in the dataset represents the correlation between two rows.
        """
        cor_dist_df_sub = query_sub.apply(
            lambda row: data.apply(lambda inner_row: 1 - np.corrcoef(row, inner_row)[0, 1], axis=1),
            axis=1)

        return cor_dist_df_sub

    @staticmethod
    def row_wise_corr_dist(X, Y):
        """
        This function will calculate the correlation distance between any row in X and any row in Y.
        X and Y should have the same protein panel.
        @param X: a numpy matrix whose row represents cells and column represents protein.
        @param Y: a numpy matrix whose row represents cells and column represents protein.
        @return: a length(X) * length(Y) numpy matrix with each entry representing the correlation distance between two datasets.
        """
        n = X.shape[1]
        X_row_mean = np.mean(X, axis=1, keepdims=True)
        Y_row_mean = np.mean(Y, axis=1, keepdims=True)
        X_centered = np.subtract(X, X_row_mean)
        Y_centered = np.subtract(Y, Y_row_mean)
        numerator = np.dot(X_centered, Y_centered.T) / n
        X_row_std_inv = np.reciprocal(np.std(X, axis=1)).reshape((X.shape[0], 1))
        Y_row_std_inv = np.reciprocal(np.std(Y, axis=1)).reshape((1, Y.shape[0]))
        denominator = np.dot(X_row_std_inv, Y_row_std_inv)
        return 1 - np.multiply(numerator, denominator)

    @staticmethod
    def cor_nn(data, query=None, k=5, option=1, npartition=3):
        """
        This function can find the nearest neighbors (rows) in "data" dataset for each record (row) in "query" dataset.
        @param data: a pandas dataframe.
        @param option: an integer to specify the computational method to find nearest neighbors.
        @param npartition: an integer to specify the number of partitions of query dataset.
        @param query: a pandas dataframe.
        @param k: the number of nearest neighbors.
        @return: {'nn_idx': neighbors, 'nn_dists': distances}
        """
        global cor_dist_df

        if query is None:
            query = data

        # make sure the input is a dataframe
        query = pd.DataFrame(query)
        data = pd.DataFrame(data)
        # initialize neighbors and distances matrices
        neighbors = pd.DataFrame(index=range(len(query)), columns=range(k), dtype='uint32')
        distances = pd.DataFrame(index=range(len(query)), columns=range(k), dtype='float64')

        if option == 0:
            # regular way, fairly slow.
            cor_dist_df = query.apply(
                                    lambda row: data.apply(lambda inner_row: 1 - np.corrcoef(row, inner_row)[0, 1], axis=1),
                                axis=1)

        elif option == 1:
            # with self-defined helper function.
            # fastest
            query_mat = query.to_numpy()
            data_mat = data.to_numpy()
            cor_dist = Mapping.row_wise_corr_dist(query_mat, data_mat)
            cor_dist_df = pd.DataFrame(cor_dist, index=query.index, columns=data.index)

        elif option == 2:
            # multithread
            import concurrent.futures
            chunk = ceil(query.shape[0] / npartition)
            queries = [query.iloc[chunk * i: chunk * (i + 1), :] for i in range(npartition)]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(Mapping.helper, queries[i], data) for i in range(npartition)]
                cor_dist_df_sub_list = [future.result() for future in futures]
            cor_dist_df = pd.concat(cor_dist_df_sub_list)

        elif option == 3:
            # multiprocess
            import concurrent.futures
            chunk_num = ceil(query.shape[0] / npartition)
            queries = [query.iloc[chunk_num * i: chunk_num * (i + 1), :] for i in range(npartition)]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(Mapping.helper, queries[i], data) for i in range(npartition)]
                cor_dist_df_sub_list = [future.result() for future in futures]

            cor_dist_df = pd.concat(cor_dist_df_sub_list)

        elif option == 4:
            # invoke R's function
            from rpy2.robjects import pandas2ri
            import rpy2.robjects as ro
            from rpy2.robjects.conversion import localconverter

            with localconverter(ro.default_converter + pandas2ri.converter):

                # get the conversion context
                conv = ro.conversion.get_conversion()

                # convert pandas DataFrames to R data frames
                r_data = conv.py2rpy(data)
                r_query = conv.py2rpy(query)

                # define the R function as a string
                func = """
                function(data, query){
                    cor_dist_df <- apply(query, 1, function(row) {
                                        apply(data, 1, function(inner_row) {1 - cor(row, inner_row)})
                                    })
                    # convert to a data frame
                    cor_dist_df <- as.data.frame(cor_dist_df)
                    return(cor_dist_df)
                }
                """

                # execute the R function with ro.r()
                cor_nn_r_internal = ro.r(func)

                # call the R function with the data
                result = cor_nn_r_internal(r_data, r_query)

                # convert R DataFrame to pandas DataFrame
                cor_dist_df = conv.rpy2py(result)

        elif option == 5:
            # entirely rely on R's function.
            from rpy2.robjects import pandas2ri
            import rpy2.robjects as ro
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects.packages import importr

            base = importr('base')
            with localconverter(ro.default_converter + pandas2ri.converter):
                # get the conversion context
                conv = ro.conversion.get_conversion()

                # convert pandas DataFrames to R data frames
                r_data = conv.py2rpy(data)
                r_query = conv.py2rpy(query)

                # define the R function as a string
                func = """
                function(
                  data,
                  query = data,
                  k = 5
                ) {
                      t_data <- t(data)
                      query <- as.matrix(query)
                      neighbors <- matrix(rep(0, k*nrow(query)), ncol=k)
                      distances <- matrix(rep(0, k*nrow(query)), ncol=k)

                      for (i in 1:nrow(query)) {
                          cor_dist <- 1 - cor(query[i,], t_data)
                          idx <- order(cor_dist)[1:k]
                          neighbors[i,] <- idx
                          distances[i,] <- cor_dist[idx]
                      }
                      neighbors = as.data.frame(apply(neighbors, 2, as.integer) - 1)
                      distances = as.data.frame(distances)
                      return(list(nn_idx=neighbors, nn_dists=distances))
                }
                """

                # execute the R function with ro.r()
                cor_nn_r_internal = ro.r(func)

                # call the R function with the data
                result = cor_nn_r_internal(r_data, r_query, k)

                # Convert the R list of data frames to a Python dictionary of pandas data frames
                py_result = {key: conv.rpy2py(result[key]) for key in result.keys()}

                return py_result

        # get indices of k nearest neighbors
        for i in range(cor_dist_df.shape[0]):
            row = cor_dist_df.iloc[i, :]
            idx = row.argsort()[:k]
            neighbors.iloc[i, :] = idx
            distances.iloc[i, :] = row.iloc[idx]

        return {'nn_idx': neighbors.astype("uint32"), 'nn_dists': distances}

    @staticmethod
    def filter_anchors(ref_mat, query_mat, anchors, k_filter=200, nn_option=2):
        """
        This function will keep anchors that preserve original data info.
        This means that the anchor in CCA space should also be anchors in the original dataset.
        @param ref_mat: a cleaned protein expression dataframe.
        @param query_mat: a cleaned protein expression dataframe.
        @param anchors: a dataframe generated in the previous step.
        @param k_filter: the number of neighbors to find in the original data space.
        @return: a dataframe of filtered anchors.
        """
        start = time.time()
        nn1_idx = pd.DataFrame()
        nn2_idx = pd.DataFrame()
        if nn_option == 1:
            nn1 = Mapping.cor_nn(data=query_mat, query=ref_mat, k=k_filter)
            nn2 = Mapping.cor_nn(data=ref_mat, query=query_mat, k=k_filter)
            nn1_idx = nn1['nn_idx']
            nn2_idx = nn2['nn_idx']
        else:
            nn = NearestNeighbors(n_neighbors=k_filter, algorithm='brute', metric="correlation")
            nn.fit(query_mat)
            nn1_idx = nn.kneighbors(ref_mat)[1]
            nn1_idx = pd.DataFrame(nn1_idx, index=ref_mat.index)
            nn = NearestNeighbors(n_neighbors=k_filter, algorithm='brute', metric="correlation")
            nn.fit(ref_mat)
            nn2_idx = nn.kneighbors(query_mat)[1]
            nn2_idx = pd.DataFrame(nn2_idx, index=query_mat.index)

        position1 = [False] * len(anchors)
        position2 = [False] * len(anchors)
        i = 0
        for q, r in zip(anchors['cellq'], anchors['cellr']):
            position1[i] = nn1_idx.iloc[r, :].isin([q]).any()
            position2[i] = nn2_idx.iloc[q, :].isin([r]).any()
            i += 1

        anchors = anchors[np.logical_or(position1, position2)]
        end = time.time()
        print(f"filter_anchors Retained {len(anchors)} anchors! Time: {round(end - start, 3)} sec")
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

        s_j1j2 = |N_CITEseq_j1 ∩ N_CITEseq_j2| + |N_CODEX_j1 ∩ N_CODEX_j2|

            |   CITE    |   CODEX
        -----------------------------
         CI |           |
         TE |           |
        ----|-----------|-----------
         CO |           |
         DEX|           |
            |           |

        N_CITEseq_j1 is the set of nearest CITEseq cells to cell j1 in the mRNA latent space.
        N_CITEseq_j2 is the set of nearest CITEseq cells to cell j2 in the CCA space
        N_CODEX_j1 is the set of nearest CODEX cells to j1 in the CCA space.
        N_CODEX_j2 is the set of nearest CODEX cells to j2 in the CCA space.

        @param neighbors: a dictionary of neighbors in CCA space as output by find_nn_rna.
        @param anchors: a dataframe of anchors as output by previous functions.
        @param num_cells_ref: total number of cells in dataset1.
        @param num_cells_query: total number of cells in dataset2.
        @param k_score: number of nn to use in shared nearest neighbor scoring.
        @return: a dataframe of anchors with scores.

        """
        start = time.time()
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
        nn_m1 = Mapping.construct_nn_mat(neighbors['nn_rr']['nn_idx'].iloc[:, :k_score],
                                         0, 0, (total_cells, total_cells))
        nn_m2 = Mapping.construct_nn_mat(neighbors['nn_rq']['nn_idx'].iloc[:, :k_score],
                                         0, num_cells_ref, (total_cells, total_cells))
        nn_m3 = Mapping.construct_nn_mat(neighbors['nn_qr']['nn_idx'].iloc[:, :k_score],
                                         num_cells_ref, 0, (total_cells, total_cells))
        nn_m4 = Mapping.construct_nn_mat(neighbors['nn_qq']['nn_idx'].iloc[:, :k_score],
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
        end = time.time()
        print(f"score_anchors. Time: {round(end - start, 3)} sec")
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
        start = time.time()
        # Extract cell expression proteins
        data_use_r = ref_mat.iloc[anchors["cellr"]].reset_index(drop=True)
        data_use_q = query_mat.iloc[anchors["cellq"]].reset_index(drop=True)

        # Subtract the data frames to obtain the integration matrix
        integration_matrix = data_use_q.subtract(data_use_r)

        # Set the row names (index) to anchors_q
        integration_matrix.index = neighbors["cellsq"][anchors["cellq"]]

        end = time.time()
        print(f"find_integration_matrix Time: {round(end - start, 3)} sec")

        return integration_matrix

    @staticmethod
    def find_weights(neighbors, anchors, query_mat, k_weight=300, sd_weight=1, nn_option=2, delta=0.00001):
        """
        This function will find weights for anchors.
        This weight is based on the distance of query cell and anchor distance.
        @param neighbors: a dictionary generated in a previous step.
        @param anchors: a dataframe that includes three columns (cellq, cellr, and score).
        @param query_mat: a dataframe whose row represents query cell and whose column represents protein.
        @param k_weight: the number of nearest anchors to use in correction.
        @param sd_weight: standard deviation of the Gaussian kernel.
        @param delta: in some extreme cases, the last k_weight's entry will be zero. Add delta to avoid zero division.
        @return: a dataframe whose row represents query cell and column represents anchors.
        """
        start = time.time()

        # initialize some variables
        cellsr = neighbors["cellsr"]
        cellsq = neighbors["cellsq"]
        anchor_cellsq = anchors["cellq"]
        data = query_mat.iloc[anchor_cellsq, :]

        nn_dists = pd.DataFrame();
        nn_idx = pd.DataFrame();
        if nn_option == 1:
            # find the nearest anchors to each query cell
            kna_query = Mapping.cor_nn(data=data, query=query_mat, k=k_weight)
            nn_dists = kna_query["nn_dists"]
            nn_dists.index = cellsq
            nn_idx = kna_query["nn_idx"]
            nn_idx.index = cellsq
        else:
            nn = NearestNeighbors(n_neighbors=k_weight, algorithm='brute', metric="correlation")
            nn.fit(query_mat.iloc[anchor_cellsq, :])
            nn_dists, nn_idx = nn.kneighbors(query_mat)
            nn_dists = pd.DataFrame(nn_dists, index=cellsq)
            nn_idx = pd.DataFrame(nn_idx, index=cellsq)

        # add delta to avoid zero division.
        zeros = nn_dists.iloc[:, k_weight-1] == 0
        nn_dists.loc[zeros, k_weight-1] = nn_dists.loc[zeros, k_weight-1] + delta
        # divide each entry by that cell's kth nearest neighbor's distance.
        # convert distance to weight
        nn_dists = 1 - nn_dists.div(nn_dists.iloc[:, k_weight - 1], axis=0)

        # initialize a dataframe.
        dists_weights = pd.DataFrame(data=0.0, index=cellsq, columns=range(len(anchor_cellsq)))

        # define a helper function
        def helper(row, index):
            idx = nn_idx.loc[index, ]
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

        end = time.time()

        # print a message.
        print(f"find_weights Time: {round(end - start, 3)} sec")

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
        start = time.time()
        integration_matrix.index = weights.columns
        bv = weights.dot(integration_matrix)
        bv.index = query_mat.index
        integrated = query_mat - bv
        stvea.codex_protein_corrected = integrated
        end = time.time()
        print(f"transform_data_matrix Time: {round(end - start, 3)} sec")
        return

    def map_codex_to_cite(self,
                          k_find_nn=80,
                          k_find_anchor=20,
                          k_filter_anchor=100,
                          k_score_anchor=80,
                          k_find_weights=100,
                          nn_option=2):
        """
        This function will calibrate CODEX protein expression levels to CITE-seq protein expression levels.
        Wrap up all functions in this class.
        """
        # find common proteins
        start = time.time()
        common_protein = [protein for protein in self.stvea.codex_protein.columns if
                          protein in self.stvea.cite_protein.columns]

        if len(common_protein) < 2:
            # for STvEA to properly transfer value from CODEX to CITE.
            # enough proteins are required.
            print("Too few common proteins between CODEX proteins and CITE-seq proteins")
            exit(1)

        # select common protein columns
        codex_subset = self.stvea.codex_protein.loc[:, common_protein]
        cite_subset = self.stvea.cite_protein.loc[:, common_protein]

        # construct common CCA space.
        cca_data = Mapping.run_cca(cite_subset.T, codex_subset.T, True, num_cc=len(common_protein) - 1)

        cite_count = cite_subset.shape[0]
        # find the nearest neighbors
        # return a dict {'nn_rr': nn_rr, 'nn_rq': nn_rq, 'nn_qr': nn_qr, 'nn_qq': nn_qq,
        #                 'cellsr': ref_emb.index.values, 'cellsq': query_emb.index.values}

        if self.stvea.cite_latent.shape != (0, 0):
            cite_latent = self.stvea.cite_latent
        else:
            cite_latent = self.stvea.cite_protein

        neighbors = Mapping.find_nn_rna(ref_emb=cca_data.iloc[:cite_count, :],
                                        query_emb=cca_data.iloc[cite_count:, :],
                                        rna_mat=cite_latent,
                                        k=k_find_nn,
                                        nn_option=nn_option)

        anchors = Mapping.find_anchor_pairs(neighbors, k_find_anchor)

        anchors = Mapping.filter_anchors(cite_subset, codex_subset, anchors, k_filter_anchor, nn_option=nn_option)

        anchors = Mapping.score_anchors(neighbors, anchors, len(neighbors["nn_rr"]["nn_idx"]),
                                        len(neighbors["nn_qq"]["nn_idx"]), k_score_anchor)

        integration_matrix = Mapping.find_integration_matrix(cite_subset, codex_subset, neighbors, anchors)

        weights = Mapping.find_weights(neighbors, anchors, codex_subset, k_find_weights, nn_option=nn_option)

        Mapping.transform_data_matrix(codex_subset, integration_matrix, weights, self.stvea)

        end = time.time()
        print(f"map_codex_to_cite: {round(end - start, 3)}")

    def transfer_matrix(self,
                        k=None,
                        c=0.1,
                        mask_threshold=0.5,
                        mask=True,
                        nn_option=2):
        """
        This function builds a transfer matrix.
        @param k: number of the nearest neighbors to find.
        @param c: constant controls the width of the Gaussian kernel.
        """
        start = time.time()
        from_dataset = self.stvea.cite_protein
        to_dataset = self.stvea.codex_protein_corrected

        if k is None:
            if len(to_dataset) < 1000:
                # for small dataset
                k = int(np.floor(len(to_dataset) * 0.02))
            else:
                # regular dataset
                k = int(np.floor(len(to_dataset) * 0.002))

        # compute query knn from cor_nn
        # weight each nn based on gaussian kernel of distance
        # create weighted nn matrix as sparse matrix
        # return nn matrix
        if nn_option == 1:
            nn_list = Mapping.cor_nn(from_dataset, to_dataset, k=k)
            nn_idx = nn_list['nn_idx']
            nn_dists = nn_list['nn_dists']
        else:
            nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric="correlation")
            nn.fit(from_dataset)
            nn_dists, nn_idx = nn.kneighbors(to_dataset)
            nn_dists = pd.DataFrame(nn_dists, index=self.stvea.codex_protein_corrected.index)
            nn_idx = pd.DataFrame(nn_idx, index=self.stvea.codex_protein_corrected.index)

        nn_dists_exp = np.exp(nn_dists / -c)

        # some CODEX cells may not have near neighbors,
        # only cells below this threshold will be kept
        if mask:
            self.stvea.codex_mask = nn_dists.mean(axis=1) < mask_threshold
            self.stvea.codex_mask.index = self.stvea.codex_protein_corrected.index

        # row-normalize the distance matrix
        nn_weights = nn_dists_exp.apply(lambda row: row / sum(row), axis=1)

        # gather entries and coords for the sparse matrix
        idx_array = nn_idx.to_numpy()
        weights_array = nn_weights.to_numpy()

        # flatten arrays and create coordinate pairs
        rows = np.repeat(np.arange(idx_array.shape[0]), idx_array.shape[1])
        cols = idx_array.flatten()
        data = weights_array.flatten()

        # create a sparse matrix
        shape = (self.stvea.codex_protein_corrected.shape[0], self.stvea.cite_protein.shape[0])
        transfer_matrix = coo_matrix((data, (rows, cols)), shape=shape)

        # convert to DataFrame
        self.stvea.transfer_matrix = pd.DataFrame(transfer_matrix.todense())
        self.stvea.transfer_matrix.index = to_dataset.index
        self.stvea.transfer_matrix.columns = from_dataset.index

        end = time.time()
        print(f"transfer_matrix Time: {round(end - start, 3)} sec")

        return
