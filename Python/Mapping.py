import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler, scale, normalize
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import svds
from numpy.linalg import norm

class Mapping:
    def __init__(self):
        pass

    @staticmethod
    def run_cca(object1, object2, standardize=True, num_cc=30):

        cells1 = object1.columns
        cells2 = object2.columns

        if standardize:
            scaler = StandardScaler(with_std=True, with_mean=True)
            object1 = scaler.fit_transform(object1.T).T
            object2 = scaler.fit_transform(object2.T).T

        mat3 = np.dot(object1.T, object2)

        u, s, v = svds(mat3, k=num_cc)

        cca_data = np.concatenate([u, v.T], axis=0)

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

        codex_cca, cite_cca = Map().run_cca(codex_subset, cite_subset, True)


        pass