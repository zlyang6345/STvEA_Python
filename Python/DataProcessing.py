import pandas as pd


class DataProcessing:

    # this method will read relevant data (csv files) into STvEA object
    def read(self, stvea):

        self.read_cite(stvea)

        self.read_codex(stvea)

    def read_cite(self, stvea):
        """
        This method will read cvs files related to CITE_seq
        :param stvea: an STvEA object
        """
        stvea.cite_latent = pd.read_csv("../Data/cite_latent.csv")

        stvea.cite_protein = pd.read_csv("../Data/cite_protein.csv")

        stvea.cite_mRNA = pd.read_csv("../Data/cite_mRNA.csv")

    def read_codex(self, stvea):
        """
        This mehod will read cvs files related to CODEX
        :param stvea: an STvEA object
        """
        stvea.codex_blanks = pd.read_csv("../Data/codex_blanks.csv")

        stvea.codex_protein = pd.read_csv("../Data/codex_protein.csv")

        stvea.codex_size = pd.read_csv("../Data/codex_size.csv")

        stvea.codex_spatial = pd.read_csv("../Data/codex_spatial.csv")


    def take_subset(self, stvea, amount_codex = -1, amount_cite = -1):

        """
        This function will take a subset of original data
        :param amount_codex: the amount of records will be kept for CODEX
        :param amount_cite: the amount of records will be kept for CITE_seq
        :param stvea: an STvEA object
        """

        if(amount_cite < len(stvea.cite_protein) and amount_cite > 0):

            stvea.cite_protein = stvea.cite_protein[1:amount_cite]

            stvea.cite_latent = stvea.cite_latent[1:amount_cite]

            stvea.cite_mRNA = stvea.cite_mRNA[1:amount_cite]

        if(amount_codex < len(stvea.codex_blanks) and amount_codex > 0):

            stvea.codex_blanks = stvea.codex_blanks[1:amount_codex]

            stvea.codex_size = stvea.codex_size[1:amount_codex]

            stvea.codex_spatial = stvea.codex_spatial[1:amount_codex]

            stvea.codex_protein = stvea.codex_protein[1:amount_codex]



