import math
from unittest import TestCase
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

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
        data = np.array([[0.12077515, 0.48107759, 0.87388194, 0.24158751, ],
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

    def test_find_nn_rna(self):
        stvea = STvEA.STvEA()
        data_processor = DataProcessor.DataProcessor()
        data_processor.read(stvea)

        r_cca_result = pd.read_csv("../Tests/r_cca_matrix.csv", index_col=0, header=0)
        r_cca_result = r_cca_result.apply(pd.to_numeric)
        cite_count = 1000
        neighbors = Mapping.Mapping().find_nn_rna(ref_emb=r_cca_result.iloc[:cite_count, :],
                                                  query_emb=r_cca_result.iloc[cite_count:, :],
                                                  rna_mat=stvea.cite_latent,
                                                  k=80)

        r_nn_qq = pd.read_csv("../Tests/r_nn_qq.csv", index_col=0, header=0)
        r_nn_qq = r_nn_qq.apply(pd.to_numeric)
        fig, ax = plt.subplots(figsize=(12, 12))

        for j in range(81):
            x = r_nn_qq.iloc[:, j]
            y = neighbors["nn_qq"]["nn_idx"].iloc[:, j]
            ax.scatter(x, y)

        ax.set_xlabel("R")
        ax.set_ylabel("Python")
        ax.set_title("Scatter Plot of nn_qq Result")
        plt.show()

        r_nn_rr = pd.read_csv("../Tests/r_nn_rr.csv", index_col=0, header=0)
        r_nn_rr = r_nn_rr.apply(pd.to_numeric)
        fig, ax = plt.subplots(figsize=(12, 12))

        for j in range(81):
            x = r_nn_rr.iloc[:, j]
            y = neighbors["nn_rr"]["nn_idx"].iloc[:, j]
            ax.scatter(x, y)

        ax.set_xlabel("R")
        ax.set_ylabel("Python")
        ax.set_title("Scatter Plot of nn_rr Result")
        plt.show()

        r_nn_qr = pd.read_csv("../Tests/r_nn_qr.csv", index_col=0, header=0)
        r_nn_qr = r_nn_qr.apply(pd.to_numeric)

        fig, ax = plt.subplots(figsize=(12, 12))

        for j in range(80):
            x = r_nn_qr.iloc[:, j]
            y = neighbors["nn_qr"]["nn_idx"].iloc[:, j]
            ax.scatter(x, y)

        ax.set_xlabel("R")
        ax.set_ylabel("Python")
        ax.set_title("Scatter Plot of nn_qr Result")
        plt.show()

        r_nn_rq = pd.read_csv("../Tests/r_nn_rq.csv", index_col=0, header=0)
        r_nn_rq = r_nn_rq.apply(pd.to_numeric)

        fig, ax = plt.subplots(figsize=(12, 12))

        for j in range(80):
            x = r_nn_rq.iloc[:, j]
            y = neighbors["nn_rq"]["nn_idx"].iloc[:, j]
            ax.scatter(x, y)

        ax.set_xlabel("R")
        ax.set_ylabel("Python")
        ax.set_title("Scatter Plot of nn_rq Result")
        plt.show()

    def test_find_anchor_pairs(sefl):

        nn_qq = pd.read_csv("../Tests/r_nn_qq.csv", index_col=0, header=0).astype("uint32")
        nn_qq_idx = nn_qq.apply(lambda x: x - 1)
        nn_qq = {
            "nn_idx": nn_qq_idx
        }

        nn_rr = pd.read_csv("../Tests/r_nn_rr.csv", index_col=0, header=0).astype("uint32")
        nn_rr_idx = nn_rr.apply(lambda x: x - 1)
        nn_rr = {
            "nn_idx": nn_rr_idx
        }

        nn_qr = pd.read_csv("../Tests/r_nn_qr.csv", index_col=0, header=0).astype("uint32")
        nn_qr_idx = nn_qr.apply(lambda x: x - 1)
        nn_qr = {
            "nn_idx": nn_qr_idx
        }

        nn_rq = pd.read_csv("../Tests/r_nn_rq.csv", index_col=0, header=0).astype("uint32")
        nn_rq_idx = nn_rq.apply(lambda x: x - 1)
        nn_rq = {
            "nn_idx": nn_rq_idx
        }

        neighbors = {'nn_rr': nn_rr, 'nn_rq': nn_rq, 'nn_qr': nn_qr, 'nn_qq': nn_qq}

        python_anchors = Mapping.Mapping().find_anchor_pairs(neighbors, k_anchor=20)
        r_anchors = pd.read_csv("../Tests/anchors.csv", index_col=0, header=0).astype("uint32")

        plt.figure(figsize=(24, 12))

        plt.subplot(1, 2, 1)
        sns.scatterplot(data=python_anchors, x="cellr", y="cellq", hue="score", legend=False)
        plt.title("Scatter Plot of Python Anchor Result")

        plt.subplot(1, 2, 2)
        sns.scatterplot(data=r_anchors, x="cellr", y="cellq", hue="score", legend=False)
        plt.title("Scatter Plot of R Anchor Result")

        plt.show()

    def test_filter_anchors(self):
        ref_mat = pd.read_csv("../Tests/r_cite_clean.csv", index_col=0, header=0).astype("float64")
        query_mat = pd.read_csv("../Tests/r_codex_clean.csv", index_col=0, header=0).astype("float64")
        anchors = pd.read_csv("../Tests/anchors.csv", index_col=0, header=0).astype("uint32")
        anchors = anchors.apply(lambda x: x - 1)  # csv generated by R, and R is one-indexed
        filterd_anchors = Mapping.Mapping().filter_anchors(ref_mat, query_mat, anchors, k_filter=100)
        assert (filterd_anchors.shape[0] == 2230)

    def test_score_anchors(self):
        filtered_anchors = pd.read_csv("../Tests/filteredAnchors.csv", index_col=0, header=0).astype("uint32")
        filtered_anchors = filtered_anchors.apply(lambda x: x - 1)

        nn_qq = pd.read_csv("../Tests/r_nn_qq.csv", index_col=0, header=0).astype("uint32")
        nn_qq_idx = nn_qq.apply(lambda x: x - 1)
        nn_qq = {
            "nn_idx": nn_qq_idx
        }

        nn_rr = pd.read_csv("../Tests/r_nn_rr.csv", index_col=0, header=0).astype("uint32")
        nn_rr_idx = nn_rr.apply(lambda x: x - 1)
        nn_rr = {
            "nn_idx": nn_rr_idx
        }

        nn_qr = pd.read_csv("../Tests/r_nn_qr.csv", index_col=0, header=0).astype("uint32")
        nn_qr_idx = nn_qr.apply(lambda x: x - 1)
        nn_qr = {
            "nn_idx": nn_qr_idx
        }

        nn_rq = pd.read_csv("../Tests/r_nn_rq.csv", index_col=0, header=0).astype("uint32")
        nn_rq_idx = nn_rq.apply(lambda x: x - 1)
        nn_rq = {
            "nn_idx": nn_rq_idx
        }

        neighbors = {'nn_rr': nn_rr, 'nn_rq': nn_rq, 'nn_qr': nn_qr, 'nn_qq': nn_qq}

        python_scored_anchors = Mapping.Mapping().score_anchors(neighbors, filtered_anchors, len(nn_rr["nn_idx"]),
                                                                len(nn_qq["nn_idx"]), 80)

        r_scored_anchors = pd.read_csv("../Tests/r_scored_anchors.csv", index_col=0, header=0,
                                       dtype={"cellr": int, "cellq": int, "score": float})
        r_scored_anchors[["cellr", "cellq"]] = r_scored_anchors[["cellr", "cellq"]] - 1
        r_scored_anchors = r_scored_anchors.sort_values(by=["cellr", "cellq"]).reset_index(drop=True)
        python_scored_anchors = python_scored_anchors.sort_values(by=["cellr", "cellq"]).reset_index(drop=True)
        concat_results = pd.concat([r_scored_anchors, python_scored_anchors], axis=1)
        assert (concat_results.apply(
            lambda row: row[0] == row[3] and row[1] == row[4] and math.isclose(row[2], row[5], rel_tol=0.0001),
            axis=1).all())

    def test_construct_nn_mat(self):
        from numpy.testing import assert_array_equal
        nn_idx = np.array([[0, 1, 2],
                           [1, 0, 2],
                           [2, 0, 1]])
        nn_mat = Mapping.Mapping().construct_nn_mat(nn_idx, 0, 0, (4, 4)).toarray()
        target = np.array([[1, 1, 1, 0],
                           [1, 1, 1, 0],
                           [1, 1, 1, 0],
                           [0, 0, 0, 0]])
        assert_array_equal(nn_mat, target)

        nn_mat = Mapping.Mapping().construct_nn_mat(nn_idx, 0, 1, (4, 4)).toarray()
        target = np.array([[0, 1, 1, 1],
                           [0, 1, 1, 1],
                           [0, 1, 1, 1],
                           [0, 0, 0, 0]])
        assert_array_equal(nn_mat, target)

    def test_find_integration_matrix(self):

        filtered_anchors = pd.read_csv("../Tests/filteredAnchors.csv", index_col=0, header=0).astype("uint32")
        filtered_anchors = filtered_anchors.apply(lambda x: x - 1)

        nn_qq = pd.read_csv("../Tests/r_nn_qq.csv", index_col=0, header=0).astype("uint32")
        nn_qq_idx = nn_qq.apply(lambda x: x - 1)
        nn_qq = {
            "nn_idx": nn_qq_idx
        }

        nn_rr = pd.read_csv("../Tests/r_nn_rr.csv", index_col=0, header=0).astype("uint32")
        nn_rr_idx = nn_rr.apply(lambda x: x - 1)
        nn_rr = {
            "nn_idx": nn_rr_idx
        }

        nn_qr = pd.read_csv("../Tests/r_nn_qr.csv", index_col=0, header=0).astype("uint32")
        nn_qr_idx = nn_qr.apply(lambda x: x - 1)
        nn_qr = {
            "nn_idx": nn_qr_idx
        }

        nn_rq = pd.read_csv("../Tests/r_nn_rq.csv", index_col=0, header=0).astype("uint32")
        nn_rq_idx = nn_rq.apply(lambda x: x - 1)
        nn_rq = {
            "nn_idx": nn_rq_idx
        }

        r_cite_clean = pd.read_csv("../Tests/r_cite_clean.csv", header=0, index_col=0).astype("float64")
        cellsr = r_cite_clean.index

        r_codex_clean = pd.read_csv("../Tests/r_codex_clean.csv", header=0, index_col=0).astype("float64")
        cellsq = r_codex_clean.index

        neighbors = {'nn_rr': nn_rr, 'nn_rq': nn_rq, 'nn_qr': nn_qr, 'nn_qq': nn_qq, "cellsr": cellsr, "cellsq": cellsq}

        ref_mat = pd.read_csv("../Tests/r_cite_clean.csv", index_col=0, header=0).astype("float64")
        query_mat = pd.read_csv("../Tests/r_codex_clean.csv", index_col=0, header=0).astype("float64")

        r_scored_anchors = pd.read_csv("../Tests/r_scored_anchors.csv", index_col=0, header=0,
                                       dtype={"cellr": int, "cellq": int, "score": float})
        r_scored_anchors[["cellr", "cellq"]] = r_scored_anchors[["cellr", "cellq"]] - 1

        python_integration_matrix = Mapping.Mapping().find_integration_matrix(ref_mat, query_mat, neighbors, r_scored_anchors)

        r_integration_matrix = pd.read_csv("../Tests/r_integration_matrix.csv", header=0, index_col=0)

        fig, ax = plt.subplots(figsize=(12, 12))
        for i, column in enumerate(python_integration_matrix.columns):
            x = r_integration_matrix[column]
            y= python_integration_matrix[column]
            ax.scatter(x, y, label=column)

        ax.set_title("Integration Matrix Result")
        ax.set_xlabel("R")
        ax.set_ylabel("Python")
        plt.legend()
        plt.show()







