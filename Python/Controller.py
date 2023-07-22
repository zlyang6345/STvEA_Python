import STvEA
import DataProcessor


class Controller:
    st = STvEA.STvEA()
    dpr = None

    def __init__(self):
        self.st = STvEA.STvEA()
        self.dpr = DataProcessor.DataProcessor(self.st)

    def pipeline(self,
                 codex_blanks="../Data/small_dataset/codex_blanks.csv",
                 codex_protein="../Data/small_dataset/codex_protein.csv",
                 codex_size="../Data/small_dataset/codex_size.csv",
                 codex_spatial="../Data/small_dataset/codex_spatial.csv",
                 cite_latent="../Data/small_dataset/cite_latent.csv",
                 cite_protein="../Data/small_dataset/cite_protein.csv",
                 cite_mrna="../Data/small_dataset/cite_mRNA.csv",
                 codex_border=-1,
                 cite_border=-1,
                 size_lim=[1000, 25000],
                 blank_lower=[-1200, -1200, -1200, -1200],
                 blank_upper=[6000, 2500, 5000, 2500],
                 maxit=500,
                 factr=1e-9,
                 optim_init=None,
                 ignore_warnings=True
                 ):
        """
        This is the pipeline of STvEA.
        """
        self.dpr.read_codex(codex_blanks=codex_blanks,
                            codex_protein=codex_protein,
                            codex_size=codex_size,
                            codex_spatial=codex_spatial)
        self.dpr.read_cite(cite_latent=cite_latent,
                           cite_protein=cite_protein,
                           cite_mrna=cite_mrna)
        self.dpr.take_subset(3000)
        self.dpr.filter_codex(size_lim=size_lim,
                              blank_lower=blank_lower,
                              blank_upper=blank_upper)
        self.dpr.clean_codex()
        self.dpr.clean_cite(maxit=maxit,
                            factr=factr,
                            optim_init=optim_init,
                            ignore_warnings=ignore_warnings)

