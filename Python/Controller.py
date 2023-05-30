import STvEA
import DataProcessing

class Controller:
    def __init__(self):
        pass


    def main(self):
        """
        Main method for the STvEA project
        """
        # create a STvEA object
        stvea = STvEA.STvEA()

        # create a new data processing object
        data_processing = DataProcessing.DataProcessing()

        # reading cvs files
        print("reading cvs files...")
        data_processing.read(stvea)

        # print some summary information
        print("codex_protein: \n", stvea.codex_protein, "\n\n")
        print("codex_spatial: \n", stvea.codex_spatial, "\n\n")
        print("codex_size: \n", stvea.codex_size, "\n\n")
        print("codex_blanks: \n", stvea.codex_blanks, "\n\n")

        print("cite_mRNA.shape: \n", stvea.cite_mRNA, "\n\n")
        print("cite_latent.shape: \n", stvea.cite_latent, "\n\n")
        print("cite_protein: \n", stvea.cite_protein, "\n\n")

        while True:
            # whether to take subset of the data
            # ask for user's opinion
            print("Take subset of the data? Input y or n")
            user_input = input("")
            if user_input == "y":
                user_input = input("Input two numbers splited by \",\", the first for the amount of CODEX cells, "
                                   "and the second for the amount of CITE-seq cells\n"
                                   "If you don't want take subset of a dataset, just input -1, such as 1000, -1")
                amounts = user_input.split(",")
                codex_amount = int(amounts[0])
                cite_amount = int(amounts[1])
                data_processing.take_subset(stvea,codex_amount, cite_amount)
            elif user_input == "n":
                break
            else:
                continue


        while True:
            # filer and clean the data set
            # ask for user's opinion
            user_input = input("Filer and clean CODEX?\n Input y or n")
            if user_input == "y" :
                data_processing.filter_codex(stvea)
            elif user_input == "n":
                break
            else:
                continue








# call the main method
controller = Controller()
controller.main()