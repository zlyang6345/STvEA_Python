print("Here")
import STvEA
print("Here3")
import DataProcessing
print("Here2")


class Controller:
    def __init__(self):
        pass


    def main(self):
        """
        Main method for the STvEA project
        """

        welcome_info = """
        


 __          __  _                              _             _____ _______    ______          
 \ \        / / | |                            | |           / ____|__   __|  |  ____|   /\    
  \ \  /\  / /__| | ___ ___  _ __ ___   ___    | |_ ___     | (___    | |_   _| |__     /  \   
   \ \/  \/ / _ \ |/ __/ _ \| '_ ` _ \ / _ \   | __/ _ \     \___ \   | \ \ / /  __|   / /\ \  
    \  /\  /  __/ | (_| (_) | | | | | |  __/   | || (_) |    ____) |  | |\ V /| |____ / ____ \ 
     \/  \/ \___|_|\___\___/|_| |_| |_|\___|    \__\___/    |_____/   |_| \_/ |______/_/    \_\
                                                                                               
                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                          
                                                                                 
"""

        print(welcome_info)
        # create a STvEA object
        stvea = STvEA.STvEA()

        # create a new data processing object
        data_processing = DataProcessing.DataProcessing()

        # reading cvs files
        print("-------------------------------------------------------------------------")
        print("Reading cvs files...")
        print("-------------------------------------------------------------------------\n\n\n")

        data_processing.read(stvea)

        # print some summary information
        print("-------------------------Summary Info------------------------------------")
        print("codex_protein: \n", stvea.codex_protein, "\n\n")
        print("codex_spatial: \n", stvea.codex_spatial, "\n\n")
        print("codex_size: \n", stvea.codex_size, "\n\n")
        print("codex_blanks: \n", stvea.codex_blanks, "\n\n")

        print("cite_mRNA: \n", stvea.cite_mRNA, "\n\n")
        print("cite_latent: \n", stvea.cite_latent, "\n\n")
        print("cite_protein: \n", stvea.cite_protein, "\n\n")
        print("-------------------------Summary Info------------------------------------")

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
                break
            elif user_input == "n":
                break
            else:
                continue


        while True:
            # filer and clean the data set
            # ask for user's opinion
            user_input = input("Filer CODEX?\n Input y or n\n")
            if user_input == "y" :
                data_processing.filter_codex(stvea)
                break
            elif user_input == "n":
                break
            else:
                continue






# call the main method
controller = Controller()
controller.main()