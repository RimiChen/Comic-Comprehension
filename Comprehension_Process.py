# import queue
from asyncio.windows_events import NULL
import math
import sys, os


from ctypes import sizeof
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import numpy as np
import random
import math
import os
import h5py
import json
import sklearn
import sklearn.preprocessing

from FeatureCombiner import *


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Knowledge_from_comics'))
from PreProcesser import *
from KnowledgeGraphConstructer import * 
from FeatureFuns import *
# from KnowledgeGraph import *


# from  Knowledge.KnowledgeGraphConstructer import *

# parameters
IS_PAGE = True
IS_FEATURE_ANALYSIS = True
IS_VISUAL_WEIGHT = True
IS_FEATURE_LAYOUT = True

# hf = h5py.File(H5_FILE_NAME+'.h5', 'w')
IMAGE_FEATURE_MODEL_FORMAT = ".h5"
IMAGE_FEATURE_MODEL_PATH = "VisualizedBy_manga_image" + IMAGE_FEATURE_MODEL_FORMAT

TEXT_FEATURE_MODEL_FORMAT = ".h5"
TEXT_FEATURE_MODLE_PAHT = "VisualizedBy_manga_image" + IMAGE_FEATURE_MODEL_FORMAT

# if feature is too large, then save outside
FILTER_LIST = {
    "Text": NULL,
    "VisualizedBy": IMAGE_FEATURE_MODEL_PATH,
    "Panel": NULL
}

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

    def filter(self, filterList, imagePath, annotationPath, featureFuns):
        # select and combine as the model required form
        print("SYSTEM: select feature types according to list.")
        print("SYSTEM: check the feature model exist, if so request as asked")
        # print("SYSTEM: pass the feature model to the LSTM model and retrieve features from the models to describe new panels")

        # featureCombiner = featureCombiner()    
        
        knowledgeGraphConstructer = KnowledgeGraphConstructer(featureFuns)
        knowledgeGraph = knowledgeGraphConstructer.constructKnowledgeGraph(imagePath, annotationPath)

        # out hdf5 and finish the preprocessing part of hierachical LSTM
        featurePreTrainedModel = IMAGE_FEATURE_MODEL_PATH
        self.featureCombiner.combineFeatures(knowledgeGraph, FILTER_LIST, True)


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
 
    
    # alter the image feature model here
    imagePretrainModel = VGG16(weights='imagenet', include_top=True, pooling='avg')
    # print(type(model))
    #### we want fc layer
    imagePretrainModel.layers.pop()
    imageFeatureModel = Model(imagePretrainModel.input, imagePretrainModel.layers[-1].output)   
    encoderList = {
        "VisualizedBy": imageFeatureModel,
        "Text": NULL,
        "Panel": NULL
    }   

    featureFuns = FeatureFuns(encoderList)    
    comprehension_process.filter(FILTER_LIST, imagePath, annotationPath, featureFuns)  

    comprehension_process.model()  
    comprehension_process.evaluation()    
