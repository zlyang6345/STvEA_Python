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
                       codex_preprocess=True)
        assert(stvea.codex_protein.shape[0] == 85572)
        spatial_row_one = stvea.codex_spatial.iloc[0, :]
        assert(spatial_row_one[0] == 2820)
        assert(spatial_row_one[1] == 40984)
        assert(spatial_row_one[2] == 8100)

        dpr.read_codex(codex_blanks="../Data/raw_dataset/codex_blanks.csv",
                       codex_protein="../Data/raw_dataset/codex_protein.csv",
                       codex_size="../Data/raw_dataset/codex_size.csv",
                       codex_spatial="../Data/raw_dataset/codex_spatial.csv",
                       codex_preprocess=True,
                       codex_border=564000)

        assert (stvea.codex_protein.shape[0] == 9186)


