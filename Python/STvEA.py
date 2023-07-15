import pandas as pd
import numpy as np


class STvEA:
    cite_latent = pd.DataFrame()
    cite_mRNA = pd.DataFrame()
    cite_protein = pd.DataFrame()
    cite_emb = pd.DataFrame()
    cite_cluster = list()
    hdbscan_scans = list()
    corrected_cite = pd.DataFrame()

    codex_blanks = pd.DataFrame()
    codex_protein = pd.DataFrame()
    codex_size = pd.DataFrame()
    codex_spatial = pd.DataFrame()
    codex_emb = pd.DataFrame()
    codex_knn = pd.DataFrame()
    codex_cluster = list()

    transfer_matrix = pd.DataFrame()

    def __init__(self):
        pass
