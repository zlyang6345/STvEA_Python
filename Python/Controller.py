import STvEA
import DataProcessor

class Controller:
    st = STvEA.STvEA()
    dpr = None
    
    def __init__(self):
        self.st = STvEA.STvEA()
        self.dpr = DataProcessor.DataProcessor(self.st)

    def pipeline(self):
        """
        This is the pipeline of STvEA.
        """
        self.dpr.read()
        self.dpr.filter_codex()
        self.dpr.clean_codex()
        self.dpr.clean_cite()
        self.dpr.take_subset()
        self.dpr.filter_codex()
        self.dpr.clean_codex()