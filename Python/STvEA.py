import pandas as pd


class STvEA:
    cite_latent = pd.DataFrame()
    cite_mRNA = pd.DataFrame()
    cite_protein = pd.DataFrame()
    cite_emb = pd.DataFrame()

    codex_blanks = pd.DataFrame()
    codex_protein = pd.DataFrame()
    codex_size = pd.DataFrame()
    codex_spatial = pd.DataFrame()
    codex_emb = pd.DataFrame()

    hdbscan_scans = list()
    cite_cluster = list()

    def __init__(self):
        pass
