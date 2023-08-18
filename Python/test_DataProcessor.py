from unittest import TestCase
import STvEA, DataProcessor


class TestDataProcessor(TestCase):
    def test_read_codex(self):
        stvea = STvEA.STvEA()
        dpr = DataProcessor.DataProcessor(stvea)
        dpr.read_codex(codex_blanks="../Data/raw_dataset/codex_blanks.csv",
                       codex_protein="../Data/raw_dataset/codex_protein.csv",
                       codex_size="../Data/raw_dataset/codex_size.csv",
                       codex_spatial="../Data/raw_dataset/codex_spatial.csv",
                       codex_preprocess=True,
                       codex_border=-1)
        assert (stvea.codex_protein.shape == (85572, 30))
        spatial_row_one = stvea.codex_spatial.iloc[0, :]
        assert (spatial_row_one[0] == 2820)
        assert (spatial_row_one[1] == 40984)
        assert (spatial_row_one[2] == 8100)

        dpr.read_codex(codex_blanks="../Data/raw_dataset/codex_blanks.csv",
                       codex_protein="../Data/raw_dataset/codex_protein.csv",
                       codex_size="../Data/raw_dataset/codex_size.csv",
                       codex_spatial="../Data/raw_dataset/codex_spatial.csv",
                       codex_preprocess=True,
                       codex_border=564000)

        assert (stvea.codex_protein.shape == (9186, 30))

    def test_read_cite(self):
        stvea = STvEA.STvEA()
        dpr = DataProcessor.DataProcessor(stvea)
        dpr.read_cite(cite_latent="../Data/raw_dataset/cite_latent.csv",
                      cite_mrna="../Data/raw_dataset/cite_mRNA.csv",
                      cite_protein="../Data/raw_dataset/cite_protein.csv")
        assert stvea.cite_protein.shape == (7097, 30)
        assert stvea.cite_mRNA.shape == (7097, 11712)

    def test_clean_codex(self):
        stvea = STvEA.STvEA()
        dpr = DataProcessor.DataProcessor(stvea)
        dpr.read_codex(codex_blanks="../Data/raw_dataset/codex_blanks.csv",
                       codex_protein="../Data/raw_dataset/codex_protein.csv",
                       codex_size="../Data/raw_dataset/codex_size.csv",
                       codex_spatial="../Data/raw_dataset/codex_spatial.csv",
                       codex_preprocess=True,
                       codex_border=-1)
        dpr.take_subset(1000, -1)
        dpr.clean_codex()
        pass
