import pandas as pd
import numpy as np


class STvEA:
    cite_latent = pd.DataFrame()
    cite_mRNA = pd.DataFrame()
    cite_protein = pd.DataFrame()
    cite_emb = pd.DataFrame()
    cite_cluster = pd.DataFrame()
    cite_cluster_name_dict = dict()
    hdbscan_scans = list()

    codex_blanks = pd.DataFrame()
    codex_protein = pd.DataFrame()
    codex_protein_corrected = pd.DataFrame()
    codex_size = pd.DataFrame()
    codex_spatial = pd.DataFrame()
    codex_emb = pd.DataFrame()
    codex_knn = pd.DataFrame()
    codex_cluster = pd.DataFrame()
    codex_cluster_names_transferred = pd.DataFrame()
    codex_cluster_name_dict = dict()

    transfer_matrix = pd.DataFrame()

    def __init__(self):
        pass
