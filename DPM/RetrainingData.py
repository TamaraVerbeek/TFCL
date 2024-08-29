from PrefixTreeCDD.PrefixTreeClass import PrefixTree
import PrefixTreeCDD.settings as settings
from PrefixTreeCDD.CDD import Window,PrefixTree
from skmultiflow.drift_detection import ADWIN, PageHinkley

class RetrainingData:
    def __init__(self) -> None:
        self.adwin=ADWIN()
        self.ph=PageHinkley()
        self.window=Window()
        self.tree=PrefixTree()

    def get_drifts(self,data)->list:
        for _, event in data.iterrows():
        # print(event)
        # break
        # need to implement ev into form accepted from event in line above
            caseList, Dcase, currentNode, pruningCounter, traceCounter, window = self.tree.insertByEvent(caseList, Dcase,
                                                                                                    currentNode, event,
                                                                                                    pruningCounter,
                                                                                                    traceCounter,
                                                                                                    endEventsDic, window)
            eventCounter += 1

        if window.cddFlag:  # If a complete new tree has been created
            if len(window.prefixTreeList) == window.WinSize:
                # Maximum size of window reached, start concept drift detection within the window
                temp_drifts = window.conceptDriftDetection(self.adwin, self.ph, eventCounter)
        raise NotImplementedError
        return drifts
    
    def prepare_dataset(self,dataset):
        drifts=self.get_drifts(dataset)
        return self.get_retrained_dataset(drifts,dataset)
    
    def get_retrained_dataset(self,drifts,dataset):
        retrained_dataset= dataset
        raise NotImplementedError
        return retrained_dataset