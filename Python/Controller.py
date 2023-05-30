import STvEA
import DataProcessing

class Controller:
    def __init__(self):
        pass


    def main(self):
        # create a STvEA object
        stvea = STvEA.STvEA()

        # create a new data processing object
        data_processing = DataProcessing.DataProcessing()

        # reading cvs files
        print("reading cvs files...")
        data_processing.read(stvea)

        # print some summary information
        print("codex_protein.shape: \n", stvea.codex_protein)
        print("codex_spatial: \n", stvea.codex_spatial)
        print("codex_size: \n", stvea.codex_size)
        print("codex_blanks: \n", stvea.codex_blanks)

        print("cite_mRNA.shape: \n", stvea.cite_mRNA)
        print("cite_latent.shape: \n", stvea.cite_latent)
        print("cite_protein: \n", stvea.cite_protein)




# call the main method
controller = Controller()
controller.main()