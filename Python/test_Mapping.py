from unittest import TestCase
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from Python import STvEA, DataProcessor, Mapping


class TestMapping(TestCase):
    def test_run_cca(self):
        # read in r result
        # cite
        #  +
        # codex
        r_cca_result = pd.read_csv("../Tests/r_cca_matrix.csv", index_col=0, header=0)
        r_cca_result = r_cca_result.apply(pd.to_numeric)

        stvea = STvEA.STvEA()
        data_processor = DataProcessor.DataProcessor()
        stvea.cite_protein = pd.read_csv("../Tests/r_cite_clean.csv", index_col=0, header=0)
        stvea.cite_protein = stvea.cite_protein.apply(pd.to_numeric)
        stvea.codex_protein = pd.read_csv("../Tests/r_codex_clean.csv", index_col=0, header=0)
        stvea.codex_protein = stvea.codex_protein.apply(pd.to_numeric)

        common_protein = [protein for protein in stvea.codex_protein.columns if protein in stvea.cite_protein.columns]
        codex_subset = stvea.codex_protein.loc[:, common_protein]
        cite_subset = stvea.cite_protein.loc[:, common_protein]

        cca_data = Mapping.Mapping().run_cca(cite_subset.T, codex_subset.T, True, num_cc=len(common_protein) - 1)

        fig, ax = plt.subplots(figsize=(12, 12))

        for i, protein in enumerate(cca_data.columns):
            x = r_cca_result.iloc[:, i]
            y = cca_data.iloc[:, i]
            ax.scatter(x, y, label=protein)

        ax.set_xlabel("R")
        ax.set_ylabel("Python")
        ax.set_title("Scatter Plot of CCA Result")

        ax.legend()
        plt.show()

    def test_cor_nn(self):
        data = np.array([[0.12077515, 0.48107759, 0.87388194, 0.24158751,],
                            [0.90204987, 0.41464509, 0.52640825, 0.18158962],
                          [0.19345103, 0.04797827, 0.38439081, 0.93312622],
                         [0.58445907, 0.22734591, 0.9730414, 0.02516982],
                         [0.59758854, 0.39400318, 0.84912261, 0.68797172],
                         [0.65016248, 0.26449771, 0.46336556, 0.68900643],
                         [0.32512395, 0.19546936, 0.04230138, 0.73893109],
                         [0.4916712, 0.38076017, 0.95395779, 0.58217801],
                         [0.40467036, 0.67843771, 0.90606747, 0.54219517],
                         [0.28174246, 0.07971738, 0.89442353, 0.69026102]])

        data = pd.DataFrame(data)
        result = Mapping.Mapping().cor_nn(data, data)
        nn_idx = result["nn_idx"]
        assert list(nn_idx.iloc[0, :]) == list([0, 8, 7, 3, 9])

