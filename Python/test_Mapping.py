from unittest import TestCase
import pandas as pd
from matplotlib import pyplot as plt

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

        cca_data = Mapping.Mapping().run_cca(cite_subset.T, codex_subset.T, True, num_cc=len(common_protein)-1)

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






