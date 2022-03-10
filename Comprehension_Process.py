# import queue
import math

# parameters
IS_PAGE = True
IS_FEATURE_ANALYSIS = True
IS_VISUAL_WEIGHT = True
IS_FEATURE_LAYOUT = True


class ComprehensionProcess:
    def __init__(self, isPage):
        self.isPage = isPage

    def testProcess(self):
        print("SYSTEM: Launch the comprehension process!")
    def initialProcess(self):
        print("SYSTEM: check the comic is already a panel seqeucne or comic pages")
        print("SYSTEM: if comic pages")
        print("SYSTEM: if panel sequences")

        # add visual weight (additional info) for panels
        print("SYSTEM: analyze visual weight")
        print("SYSTEM: if rearranging panel index is needed")

        # knowledge construction mode, and stored the features models
        print("SYSTEM: feature analysis according to flags")
        print("SYSTEM: according to the flags in feature queue, using call back function to process/load different features; dictionary(feature_type, feature_flag, call_back_function)")
        print("SYSTEM: stored as features models")

        # parse testing cases
        print("SYSTEM: generating test cases from dataset")

    def filter(self, filterList):
        print("SYSTEM: select feature types according to list.")
        print("SYSTEM: check the feature model exist, if so request as asked")
        # print("SYSTEM: pass the feature model to the LSTM model and retrieve features from the models to describe new panels")

    def model(self):
        print("SYSTEM: pass the feature model to the LSTM model and retrieve features from the models to describe new panels")
        print("SYSTEM: read and predict")
        print("SYSTEM: store the result")

    def evaluation(self):
        print("SYSTEM: compare result with answers, show accuracy")



if __name__ == "__main__":

    # initial the tool and interface
    comprehension_process = ComprehensionProcess(IS_PAGE)
    comprehension_process.testProcess()
    comprehension_process.initialProcess()  
    comprehension_process.filter()  
    comprehension_process.model()  
    comprehension_process.evaluation()    
