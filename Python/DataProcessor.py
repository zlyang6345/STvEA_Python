import warnings
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.stats import nbinom
from scipy.optimize import minimize

import STvEA


class DataProcessor:
    overall_sse = 0
    stvea = STvEA.STvEA()

    def __init__(self, stvea):
        self.stvea = stvea

    # this method will read relevant data (csv files) into STvEA object
    def read(self):

        self.read_cite()
        self.read_codex()

    def read_cite(self, cite_latent="../Data/cite_latent.csv", cite_protein="../Data/cite_protein.csv",
                  cite_mrna="../Data/cite_mRNA.csv"):
        """
        This method will read cvs files related to CITE_seq.
        @param cite_latent: a string to specify the address of CITE-seq latent dataset.
        @param cite_protein: a string to specify the address of CITE-seq protein dataset.
        @param cite_mrna: a string to specify the address of CITE-seq mRNA file.
        """
        self.stvea.cite_latent = pd.read_csv(cite_latent, index_col=0, header=0)
        self.stvea.cite_latent = self.stvea.cite_latent.apply(pd.to_numeric)

        self.stvea.cite_protein = pd.read_csv(cite_protein, index_col=0, header=0)
        self.stvea.cite_protein = self.stvea.cite_protein.apply(pd.to_numeric)

        self.stvea.cite_mRNA = pd.read_csv(cite_mrna, index_col=0, header=0)
        self.stvea.cite_mRNA = self.stvea.cite_mRNA.apply(pd.to_numeric)

        print("CITE-seq data read")

    def read_codex(self, codex_blanks="../Data/codex_blanks.csv",
                   codex_protein="../Data/codex_protein.csv",
                   codex_size="../Data/codex_size.csv",
                   codex_spatial="../Data/codex_spatial.csv"):
        """
        This method will read cvs files related to CODEX.
        @param codex_blanks: a string to specify the address of CODEX blank dataset.
        @param codex_protein: a string to specify the address of CODEX protein dataset.
        @param codex_size: a string to specify the address of CODEX size dataset.
        @param codex_spatial: a string to specify the address of CODEX spatial dataset.
        """

        self.stvea.codex_blanks = pd.read_csv(codex_blanks, index_col=0, header=0)
        self.stvea.codex_blanks = self.stvea.codex_blanks.apply(pd.to_numeric)

        self.stvea.codex_protein = pd.read_csv(codex_protein, index_col=0, header=0)
        self.stvea.codex_protein = self.stvea.codex_protein.apply(pd.to_numeric)

        self.stvea.codex_size = pd.read_csv(codex_size, index_col=0, header=0)
        self.stvea.codex_size = self.stvea.codex_size.apply(pd.to_numeric)

        self.stvea.codex_spatial = pd.read_csv(codex_spatial, index_col=0, header=0)
        self.stvea.codex_spatial = self.stvea.codex_spatial.apply(pd.to_numeric)

        print("CODEX files read")

    def take_subset(self, amount_codex=-1, amount_cite=-1):

        """
        This function will take a subset of original data
        @param amount_codex: the amount of records will be kept for CODEX
        @param amount_cite: the amount of records will be kept for CITE_seq
        """

        if len(self.stvea.cite_protein) > amount_cite > 0:
            self.stvea.cite_protein = self.stvea.cite_protein[1:amount_cite]

            self.stvea.cite_latent = self.stvea.cite_latent[1:amount_cite]

            self.stvea.cite_mRNA = self.stvea.cite_mRNA[1:amount_cite]

        if len(self.stvea.codex_blanks) > amount_codex > 0:
            self.stvea.codex_blanks = self.stvea.codex_blanks[1:amount_codex]

            self.stvea.codex_size = self.stvea.codex_size[1:amount_codex]

            self.stvea.codex_spatial = self.stvea.codex_spatial[1:amount_codex]

            self.stvea.codex_protein = self.stvea.codex_protein[1:amount_codex]

    def filter_codex(self,
                     size_lim=[1000, 25000],
                     blank_lower=[-1200, -1200, -1200, -1200],
                     blank_upper=[6000, 2500, 5000, 2500]
                     ):
        """
        We follow the gating strategy in Goltsev et al. to remove cells that are too small or large,
        or have too low or too high expression in the blank channels. If the limits aren't specified,
        the 0.025 and 0.99 quantiles are taken as the lower and upper bounds on size, and the 0.002 and 0.99
        quantiles are used for the blank channel expression. We then normalize the protein expression values
        by the total expression per cell.

        @param size_lim: a size limit, default to [1000, 25000]
        @param blank_lower: a vector of length 4, default to [-1200, -1200, -1200, -1200]
        @param blank_upper: a vector of length 4, default to [6000, 2500, 5000, 2500]
        """
        # If size_lim is not specified,
        # it defaults to the 0.025 and 0.99 quantiles of the size vector.
        if size_lim is None or size_lim[0] >= size_lim[1]:
            size_lim = self.stvea.codex_size[0].quantile([0.025, 0.99])

        # if blank_upper is none, it defaults to 0.995 quantile of the column vector
        if blank_upper is None:
            blank_upper = self.stvea.codex_blanks.apply(lambda x: x.quantile(0.995), axis=0)

        # if blank_upper is none, it defaults to 0.002 quantile of the column vector
        if blank_lower is None:
            blank_lower = self.stvea.codex_blanks.apply(lambda x: x.quantile(0.002), axis=0)

        # This loop iterates over each column (blank channel) of the blanks DataFrame.
        # For each column, it checks if the expression values of the corresponding blank
        # channel in each cell are greater than or equal to the corresponding lower
        # expression cutoff (blank_lower[i]).
        blank_filter_lower = pd.Series([False] * len(self.stvea.codex_blanks), index=self.stvea.codex_blanks.index)
        blank_filter_upper = pd.Series([True] * len(self.stvea.codex_blanks), index=self.stvea.codex_blanks.index)

        for i in range(len(self.stvea.codex_blanks.columns)):
            # a cell whose all blank channels below the lower cutoff will be removed
            blank_filter_lower = blank_filter_lower | (self.stvea.codex_blanks.iloc[:, i] >= blank_lower[i])
            # a cell whose any blank channels above the upper cutoff will be removed
            blank_filter_upper = blank_filter_upper & (self.stvea.codex_blanks.iloc[:, i] <= blank_upper[i])

        blank_filter = blank_filter_lower & blank_filter_upper

        # create size filter
        size_filter = ((size_lim[0] <= self.stvea.codex_size.iloc[:, 0]) & (
                self.stvea.codex_size.iloc[:, 0] <= size_lim[1]))

        # filter the original result
        mask = blank_filter & size_filter

        # filter tables
        self.stvea.codex_size = self.stvea.codex_size[mask]
        self.stvea.codex_blanks = self.stvea.codex_blanks[mask]
        self.stvea.codex_protein = self.stvea.codex_protein[mask]
        self.stvea.codex_spatial = self.stvea.codex_spatial[mask]

        # print the result
        print("Codex filtered! ", len(self.stvea.codex_blanks), " records preserved")

    def clean_codex(self):
        """
        We remove noise from the CODEX protein expression by first fitting a Gaussian mixture model to the expression
        levels of each protein. The signal expression is taken as the cumulative probability according to the
        Gaussian with the higher mean.
        """
        # subtracts the minimum value of all the elements in codex_protein (a data frame) from each element in the
        # data frame
        codex_protein_norm = self.stvea.codex_protein - self.stvea.codex_protein.min().min()

        # Calculate the row sums
        row_sums = codex_protein_norm.sum(axis=1)

        # Calculate the average of row sums
        avg_cell_total = np.mean(row_sums)

        # find rows that are not all 0s
        nonzero = (row_sums != 0)

        # each row divided by its sum and then multiplied by avg_cell_total
        codex_protein_norm.loc[nonzero] = codex_protein_norm.loc[nonzero].div(row_sums[nonzero],
                                                                              axis=0) * avg_cell_total

        # For each protein
        for col in codex_protein_norm.columns:
            # Compute Gaussian mixture on each protein
            gm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)

            # fit the model
            gm.fit(codex_protein_norm[col].values.reshape(-1, 1))

            # Identify which component has the higher mean
            signal = np.argmax(gm.means_)

            # Compute cleaned data from cumulative of higher mean Gaussian
            self.stvea.codex_protein[col] = norm.cdf(codex_protein_norm[col], loc=gm.means_[signal, 0],
                                                     scale=np.sqrt(gm.covariances_[signal, 0, 0]))

        print("CODEX cleaned!")

    @staticmethod
    def SSE(args, p_obs):
        """
        Calculates the sum of squared errors in binned probabilities of count data

        @param args: a list used in the negative binomial mixture model
        @param p_obs: a DataFrame of the probabilities of observing a given count
                      in gene expression data, as output by running value_counts() on the gene count data
        """
        # Extract values from args
        mu1, mu2, size_reciprocal1, size_reciprocal2, mixing_prop = args
        size1 = 1 / size_reciprocal1
        size2 = 1 / size_reciprocal2
        p1 = size1 / (size1 + mu1)
        p2 = size2 / (size2 + mu2)

        p_obs_index = p_obs.index

        # Expected probabilities (p_exp)
        # http://library.isr.ist.utl.pt/docs/scipy/generated/scipy.stats.nbinom.html
        # https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/NegBinomial
        p_exp = (mixing_prop * nbinom.pmf(p_obs_index, size1, p1)) + (
                (1 - mixing_prop) * nbinom.pmf(p_obs_index, size2, p2))

        # Calculate sum of squared errors (sse)
        sse = min(sum((p_exp - p_obs.values) ** 2), np.iinfo(np.int32).max)

        return sse

    @staticmethod
    def generate_p_obs(protein_expr):
        """
        Given a protein expression series, this function will count frequencies
        and calculate each frequencies' probability.
        @param protein_expr:  a protein expression series
        @return: a dataframe.
        """
        max_express = max(protein_expr)
        p_obs = pd.Series(np.zeros(max_express + 1))
        value_counts = protein_expr.value_counts().to_dict()

        # Map the values from value_counts to p_obs
        p_obs = p_obs.index.map(value_counts)
        p_obs = pd.Series(p_obs)

        # Replace NaN values with 0
        p_obs = p_obs.fillna(0)

        # calculate probability
        p_obs = p_obs / len(protein_expr)

        return p_obs

    def fit_nb(self, protein_expr,
               col_name,
               maxit=500,
               factr=1e-9,
               optim_init=None,
               verbose=False,
               method="l-bfgs-b"):
        """
        Fits the expression values of one protein with a Negative Binomial mixture
        Takes matrices and data frames instead of STvEA_R.data class

        @param method: a string to specify the method to be used in the minimize function.
        @param col_name: column name(protein name).
        @param verbose: a boolean value to specify verbosity.
        @param protein_expr: Raw CITE-seq protein data for one protein.
        @param maxit: maximum number of iterations for optim function.
        @param factr: accuracy of optim function.
        @param optim_init: a ndarray of optional initialization parameters for the optim function, if NULL, starts at five default parameter sets and picks the better one.
        @return: cleaned protein expression.
        """
        # Create a probability distribution from the raw protein expression data
        p_obs = self.generate_p_obs(protein_expr)
        if verbose:
            print(col_name + ": ")

        if optim_init is None:
            # Sometimes negative binomial doesn't fit well with certain starting parameters, so try 5
            # optim is a general optimization function
            # [5,50,2,0.5,0.5] is the initial parameter
            # SSE is the function to minimize
            optim_init = [
                [10, 60, 2, 0.5, 0.5],
                [4.8, 50, 0.5, 2, 0.5],
                [2, 18, 0.5, 2, 0.5],
                [1, 3, 2, 2, 0.5],
                [1, 3, 0.5, 2, 0.5]
            ]

        bound = [(1e-8, None)] * 4 + [(1e-8, 1)]
        fits = list()
        scores = list()

        for index, array in enumerate(optim_init):
            fits.append(minimize(self.SSE, array, args=p_obs,
                                method=method, bounds=bound,
                                options={'maxiter': maxit, 'ftol': factr}))
            scores.append(self.SSE(fits[index].x, p_obs))

        m = min(scores)
        # pick the data with minimal SSE
        for index, score in enumerate(scores):
            if score == m:
                self.overall_sse += m
                fit = fits[index].x
                break

        mu1, mu2, size_reciprocal1, size_reciprocal2, mixing_prop = fit
        size1 = 1 / size_reciprocal1
        size2 = 1 / size_reciprocal2
        p1 = size1 / (size1 + mu1)
        p2 = size2 / (size2 + mu2)

        # Distribution with higher median is signal
        signal = np.argmax([nbinom.median(size1, p1),
                            nbinom.median(size2, p2)])

        size = 1 / fit[signal + 2]
        p = size / (size + fit[signal])
        expr_clean = nbinom.cdf(protein_expr, size, p)

        return expr_clean

    @staticmethod
    def norm_cite(cite_protein, row_sums):
        """
        This function will normalize CITE-seq cells
        @param cite_protein: a dataframe.
        @param row_sums:  row sums of original protein expression dataframe.
        @return: a dataframe of normalized CITE-seq protein expression levels.
        """
        # normalize
        # find rows that are not all 0s
        nonzero = (row_sums != 0)

        # each row divided by its sum
        cite_protein = cite_protein.loc[nonzero].div(row_sums[nonzero], axis=0)

        # subtracts the minimum value of all the elements in codex_protein (a data frame) from each element in the
        cite_protein = cite_protein - cite_protein.min()

        # divide each entry with its column's largest number
        cite_protein = cite_protein.div(cite_protein.max(axis=0), axis=1)

        return cite_protein

    def clean_cite(self, maxit=500, factr=1e-9, optim_init=None, ignore_warnings=True):
        """
        This function will use mixture negative binomial distribution models to clean CITE-seq protein data
        @param ignore_warnings: a boolean value to specify whether to ignore warnings or not.
        @param maxit: the maximum number of iterations
        @param factr: accuracy of optim function
        @param optim_init: a vector of with initialization
        """
        if ignore_warnings:
            warnings.simplefilter("ignore")

        # Calculate the row sums
        row_sums = self.stvea.cite_protein.sum(axis=1)

        self.stvea.cite_protein = self.stvea.cite_protein.apply(
            lambda col: self.fit_nb(col, col.name, maxit=500, factr=1e-9, optim_init=optim_init))

        self.stvea.cite_protein = self.norm_cite(self.stvea.cite_protein, row_sums)

        print("CITE-seq protein cleaned, overall SSE: "+str(self.overall_sse))
