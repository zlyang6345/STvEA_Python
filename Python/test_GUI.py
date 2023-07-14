from unittest import TestCase
import pandas as pd
import STvEA
import numpy as np
import GUI


class TestGUI(TestCase):
    def test_cite_annotation_input(self):
        stvea = STvEA.STvEA()
        data = np.array([['', 'x', 'y'],
                         ['Cell1', 1, 2],
                         ['Cell2', 3, 4],
                         ['Cell3', 3, 5],
                         ['Cell4', 1, 3]])
        stvea.cite_emb = pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:])
        stvea.cite_cluster = pd.Series([0, 1, 1, 0])
        GUI.GUI().cite_annotation_input(stvea)
