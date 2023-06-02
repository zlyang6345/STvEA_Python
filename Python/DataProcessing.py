import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.stats import nbinom
from scipy.optimize import minimize


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

        stvea.cite_latent = pd.read_csv("../Data/cite_latent.csv", index_col=0, header=0)
        stvea.cite_latent = stvea.cite_latent.apply(pd.to_numeric)

        stvea.cite_protein = pd.read_csv("../Data/cite_protein.csv", index_col=0, header=0)
        stvea.cite_protein = stvea.cite_protein.apply(pd.to_numeric)

        stvea.cite_mRNA = pd.read_csv("../Data/cite_mRNA.csv", index_col=0, header=0)
        stvea.cite_mRNA = stvea.cite_mRNA.apply(pd.to_numeric)

    def read_codex(self, stvea):
        """
        This mehod will read cvs files related to CODEX
        :param stvea: an STvEA object
        """

        stvea.codex_blanks = pd.read_csv("../Data/codex_blanks.csv", index_col=0, header=0)
        stvea.codex_blanks = stvea.codex_blanks.apply(pd.to_numeric)

        stvea.codex_protein = pd.read_csv("../Data/codex_protein.csv", index_col=0, header=0)
        stvea.codex_protein = stvea.codex_protein.apply(pd.to_numeric)

        stvea.codex_size = pd.read_csv("../Data/codex_size.csv", index_col=0, header=0)
        stvea.codex_size = stvea.codex_size.apply(pd.to_numeric)

        stvea.codex_spatial = pd.read_csv("../Data/codex_spatial.csv", index_col=0, header=0)
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
                     size_lim=[1000, 25000],
                     # don't forget to replace these values with None and modify the main method in the end !!!!!!!!!!!!!!!!!!!!!
                     blank_lower=[-1200, -1200, -1200, -1200],
                     blank_upper=[6000, 2500, 5000, 2500]
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
        if size_lim is None or size_lim[0] >= size_lim[1]:
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
        blank_filter_lower = pd.Series([False] * len(stvea.codex_blanks), index=stvea.codex_blanks.index)
        blank_filter_upper = pd.Series([True] * len(stvea.codex_blanks), index=stvea.codex_blanks.index)

        for i in range(len(stvea.codex_blanks.columns)):
            # a cell whose all blank channels below the lower cutoff will be removed
            blank_filter_lower = blank_filter_lower | (stvea.codex_blanks.iloc[:, i] >= blank_lower[i])
            # a cell whose any blank channels above the upper cutoff will be removed
            blank_filter_upper = blank_filter_upper & (stvea.codex_blanks.iloc[:, i] <= blank_upper[i])

        blank_filter = blank_filter_lower & blank_filter_upper

        # create size filter
        size_filter = ((size_lim[0] <= stvea.codex_size.iloc[:, 0]) & (stvea.codex_size.iloc[:, 0] <= size_lim[1]))

        # filter the original result
        mask = blank_filter & size_filter

        # filter tables
        stvea.codex_size = stvea.codex_size[mask]
        stvea.codex_blanks = stvea.codex_blanks[mask]
        stvea.codex_protein = stvea.codex_protein[mask]
        stvea.codex_spatial = stvea.codex_spatial[mask]

        # print the result
        print("Codex filtered!")
        print(len(stvea.codex_blanks), " records preserved")

    def clean_codex(self, stvea):
        """
        We remove noise from the CODEX protein expression by first fitting a Gaussian mixture model to the expression
        levels of each protein. The signal expression is taken as the cumulative probability according to the
        Gaussian with the higher mean.

        :param stvea: a STvEA object
        """
        # subtracts the minimum value of all the elements in codex_protein (a data frame) from each element in the
        # data frame
        codex_protein_norm = stvea.codex_protein - stvea.codex_protein.min().min()

        # Calculate the row sums
        row_sums = codex_protein_norm.sum(axis=1)

        # Calculate the average of row sums
        avg_cell_total = np.mean(row_sums)

        # find rows that are not all 0s
        nonzero = (row_sums != 0)

        # each row divided by its sum and then multiplied by avg_cell_total
        codex_protein_norm.loc[nonzero] = codex_protein_norm.loc[nonzero].div(row_sums[nonzero], axis=0) * avg_cell_total

        # For each protein
        for col in codex_protein_norm.columns:

            # Compute Gaussian mixture on each protein
            gm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)

            # fit the model
            gm.fit(codex_protein_norm[col].values.reshape(-1, 1))

            # Identify which component has the higher mean
            signal = np.argmax(gm.means_)

            # Compute cleaned data from cumulative of higher mean Gaussian
            stvea.codex_protein[col] = norm.cdf(codex_protein_norm[col], loc=gm.means_[signal, 0],
                                  scale=np.sqrt(gm.covariances_[signal, 0, 0]))

        print("CODEX Cleaned!")


    def SSE(args, p_obs):
        """
        Calculates the sum of squared errors in binned probabilities of count data

        Parameters:
        args (list): arguments used in the negative binomial mixture model
        p_obs (pandas.DataFrame): a DataFrame of the probabilities of observing a given count
                      in gene expression data, as output by running value_counts() on the gene count data
        """
        # Extract values from args
        mu1, mu2, size_reciprocal1, size_reciprocal2, mixing_prop = args

        # Convert index of p_obs from string to numeric, retaining order
        p_obs_index = p_obs.index.astype(int)

        # Expected probabilities (p_exp)
        p_exp = (mixing_prop * nbinom.pmf(p_obs_index, 1 / size_reciprocal1, mu1)) + \
                ((1 - mixing_prop) * nbinom.pmf(p_obs_index, 1 / size_reciprocal2, mu2))

        # Calculate sum of squared errors (sse)
        sse = min(np.sum((p_exp - p_obs.values.flatten()) ** 2),
                  np.iinfo(np.int32).max)

        return sse


    def clean_cite(self, stvea):

        pass
