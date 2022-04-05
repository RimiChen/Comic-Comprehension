# import queue
import math
import sys, os

from FeatureCombiner import *


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Knowledge_from_comics'))
from PreProcesser import *
from KnowledgeGraphConstructer import * 
# from KnowledgeGraph import *


# from  Knowledge.KnowledgeGraphConstructer import *

# parameters
IS_PAGE = True
IS_FEATURE_ANALYSIS = True
IS_VISUAL_WEIGHT = True
IS_FEATURE_LAYOUT = True
FILTER_LIST = [
    "Text",
    "VisualizedBy",
    "Panel"
]

class ComprehensionProcess:
    def __init__(self, isPage):
        self.isPage = isPage
        self.featureCombiner = FeatureCombiner()



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

        pre_processer = PreProcesser(DATASET_PATH, PRE_IS_BOOK_IN_FOLDERS, PRE_CUSTOM_BOOK_CHUCK, PRE_IS_IMAGES_IN_SEQUENCE, PRE_CUSTOM_SEQUENCE_PANEL, PRE_IS_PANEL_CROPPED, PRE_DEFAULT_ORDERS, PRE_ORDERED_METHOD)
        # pre_processer.printDetailInPath()
        [imagePath, annotationPath] = pre_processer.preprocessData()
        print("TEST: ", imagePath, " , ", annotationPath)

        return [imagePath, annotationPath]

    def filter(self, filterList, imagePath, annotationPath):
        # select and combine as the model required form
        print("SYSTEM: select feature types according to list.")
        print("SYSTEM: check the feature model exist, if so request as asked")
        # print("SYSTEM: pass the feature model to the LSTM model and retrieve features from the models to describe new panels")

        # featureCombiner = featureCombiner()    
        
        knowledgeGraphConstructer = KnowledgeGraphConstructer()
        knowledgeGraph = knowledgeGraphConstructer.constructKnowledgeGraph(imagePath, annotationPath)

        self.featureCombiner.combineFeatures(knowledgeGraph, FILTER_LIST)
        # out hdf5 and finish the preprocessing part of hierachical LSTM


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
    [imagePath, annotationPath] = comprehension_process.initialProcess()  
    comprehension_process.filter(FILTER_LIST, imagePath, annotationPath)  
    comprehension_process.model()  
    comprehension_process.evaluation()    
