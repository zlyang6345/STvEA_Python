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

        stvea.cite_latent = pd.read_csv("../Data/cite_latent.csv", index_col = 0, header = 0)
        stvea.cite_latent = stvea.cite_latent.apply(pd.to_numeric)

        stvea.cite_protein = pd.read_csv("../Data/cite_protein.csv", index_col = 0, header = 0)
        stvea.cite_protein = stvea.cite_protein.apply(pd.to_numeric)

        stvea.cite_mRNA = pd.read_csv("../Data/cite_mRNA.csv", index_col = 0, header = 0)
        stvea.cite_mRNA = stvea.cite_mRNA.apply(pd.to_numeric)


    def read_codex(self, stvea):
        """
        This mehod will read cvs files related to CODEX
        :param stvea: an STvEA object
        """

        stvea.codex_blanks = pd.read_csv("../Data/codex_blanks.csv", index_col=0, header = 0)
        stvea.codex_blanks = stvea.codex_blanks.apply(pd.to_numeric)
        
        stvea.codex_protein = pd.read_csv("../Data/codex_protein.csv", index_col=0, header = 0)
        stvea.codex_protein = stvea.codex_protein.apply(pd.to_numeric)
        
        stvea.codex_size = pd.read_csv("../Data/codex_size.csv", index_col=0, header = 0)
        stvea.codex_size = stvea.codex_size.apply(pd.to_numeric)
        
        stvea.codex_spatial = pd.read_csv("../Data/codex_spatial.csv", index_col=0, header = 0)
        stvea.codex_spatial = stvea.codex_spatial.apply(pd.to_numeric)
        

    def take_subset(self, stvea, amount_codex=-1, amount_cite=-1):

        """
        This function will take a subset of original data
        :param amount_codex: the amount of records will be kept for CODEX
        :param amount_cite: the amount of records will be kept for CITE_seq
        :param stvea: an STvEA object
        """

        if (amount_cite < len(stvea.cite_protein) and amount_cite > 0):
            stvea.cite_protein = stvea.cite_protein[1:amount_cite]

            stvea.cite_latent = stvea.cite_latent[1:amount_cite]

            stvea.cite_mRNA = stvea.cite_mRNA[1:amount_cite]

        if (amount_codex < len(stvea.codex_blanks) and amount_codex > 0):
            stvea.codex_blanks = stvea.codex_blanks[1:amount_codex]

            stvea.codex_size = stvea.codex_size[1:amount_codex]

            stvea.codex_spatial = stvea.codex_spatial[1:amount_codex]

            stvea.codex_protein = stvea.codex_protein[1:amount_codex]

    def filter_codex(self, stvea,
                     size_lim = [1000, 25000],
                     blank_lower = [-1200, -1200, -1200, -1200],
                     blank_upper = [6000, 2500, 5000, 2500]
                     ):
        """
        We follow the gating strategy in Goltsev et al. to remove cells that are too small or large,
        or have too low or too high expression in the blank channels. If the limits aren't specified,
        the 0.025 and 0.99 quantiles are taken as the lower and upper bounds on size, and the 0.002 and 0.99
        quantiles are used for the blank channel expression. We then normalize the protein expression values
        by the total expression per cell.

        :param stvea: a STvEA object
        :param size_lim: a size limit like [1000, 25000]
        :param blank_lower: a vector of length 4 like [-1200, -1200, -1200, -1200]
        :param blank_upper: a vector of length 4 like [6000, 2500, 5000, 2500]
        """
        # If size_lim is not specified,
        # it defaults to the 0.025 and 0.99 quantiles of the size vector.
        if size_lim is None:
            size_lim = stvea.codex_size[0].quantile([0.025, 0.99])

        # if blank_upper is none, it defaults to 0.995 quantile of the column vector
        if blank_upper is None:
            blank_upper = stvea.codex_blanks.apply(lambda x: x.quantile(0.995), axis=0)

        # if blank_upper is none, it defaults to 0.002 quantile of the column vector
        if blank_lower is None:
            blank_lower = stvea.codex_blanks.apply(lambda x: x.quantile(0.002), axis=0)


        # This loop iterates over each column (blank channel) of the blanks DataFrame.
        # For each column, it checks if the expression values of the corresponding blank
        # channel in each cell are greater than or equal to the corresponding lower
        # expression cutoff (blank_lower[i]).
        blank_filter = pd.Series([True] * len(stvea.codex_blanks), index=stvea.codex_blanks.index)

        for i in range(len(stvea.codex_blanks.columns)):
            condition1 = stvea.codex_blanks.iloc[:, i] >= blank_lower[i]
            condition2 = stvea.codex_blanks.iloc[:, i] <= blank_upper[i]
            blank_filter = (blank_filter & (condition1 & condition2))

        size_filter = ((size_lim[0] <= stvea.codex_size.iloc[:, 0]) & (stvea.codex_size.iloc[:, 0] <= size_lim[1]))

        # filter the original result
        stvea.codex_size = stvea.codex_size[blank_filter & size_filter]
        stvea.codex_blanks = stvea.codex_blanks[blank_filter & size_filter]
        stvea.codex_protein = stvea.codex_protein[blank_filter & size_filter]
        stvea.codex_spatial = stvea.codex_spatial[blank_filter & size_filter]

        # print the result
        print("Filter completed")
        print(len(stvea.codex_blanks), " records preserved")

    def clean_codex(self, stvea):
        pass
