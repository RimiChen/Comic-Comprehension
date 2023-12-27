import sys, os
import json
import math

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


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Knowledge_from_comics'))
from KnowledgeGraph import *

TRAIN_SPLIT = 2
DEV_SPLIT = 1
TEST_SPLIT = 3
TOTAL_SPLIT = TRAIN_SPLIT + TEST_SPLIT + DEV_SPLIT

PANEL_IMAGE_RELATION = "VisualizedBy"
TEXT_RELATION = "TextlizedBy" 
PATH_RELATION = "Locate" 
FEATURE_DESCRIBE_RELATION = "DescribedBy" 



class FeatureCombiner:
    def __init__(self):
        self.printDesciption()
        
        # Don't save this because of memory
        # self. knowledgeGraph = knowledgeGraph


    # Know default information
    def printDesciption(self):
        print("SYSTEM Action: Create object from FeatureCombiner class.")
        print("SYSTEM Info: combine features during this processor.")
    
    # TODO flexibility of split
    def combineFeatures(self, knowledgeGraph: KnowledgeGraph, featureList, isNeedEvaluate):
        print("SYSTEM Action: retrieve needed features and combine as model, save image features as a model, text features outside")

        

        comicSeuqenceList = knowledgeGraph.accessFeatures("Sequence")
        print(json.dumps( comicSeuqenceList, indent = 4))
        
        # if need to evaluate understanding abliltiy
        if isNeedEvaluate == True:
            print("SYSTEM Action: need to evaluate the understanding (encoding) process, split data into Train, Test, and Eval.")


            #print([x[0]  for x in os.walk(FOLDER) if len(x[0].replace(FOLDER,"").replace(".jpg","")) > 0])

            numVolumes = len(comicSeuqenceList)
            ## load folders, know how many books
            ## dev, train, test
            
            print("SYSTEM info: book number = ", numVolumes)
        

            TrainPro = math.floor(numVolumes / TOTAL_SPLIT * TRAIN_SPLIT)
            DevPro =  math.ceil(numVolumes / TOTAL_SPLIT * DEV_SPLIT)
            TestPro = numVolumes- (TrainPro + DevPro)

            
            DevThreshold = TrainPro
            TestThreshold = numVolumes - TestPro

            print("SYSTEM info: train ", TrainPro, " Dev ", DevPro, " Test ", TestPro, "start from (", 0,",", DevThreshold,",", TestThreshold,")")


            sequenceCount = 0
            typeTable = {}

            # Sequence Level

            # create look up table
            for sequence in comicSeuqenceList:
                if sequenceCount < DevThreshold:
                    # train
                    assignedLabel = "Train"
                    if sequence in typeTable:
                        typeTable[sequence].append("Train")
                    else:
                        typeTable[sequence] = []
                        typeTable[sequence].append("Train")

                elif sequenceCount >= DevThreshold and sequenceCount < TestThreshold:
                    # dev
                    assignedLabel = "Dev"
                    if sequence in typeTable:
                        typeTable[sequence].append(assignedLabel)
                    else:
                        typeTable[sequence] = []
                        typeTable[sequence].append(assignedLabel)                            
                elif sequenceCount >= TestThreshold:
                    # test
                    assignedLabel = "Test"
                    if sequence in typeTable:
                        typeTable[sequence].append(assignedLabel)
                    else:
                        typeTable[sequence] = []
                        typeTable[sequence].append(assignedLabel)     

                sequenceCount = sequenceCount + 1
            print(json.dumps(typeTable, indent = 4))



            
            # Panel level
            entityList = knowledgeGraph.accessFeatures("Panel")

            # for targetEntity in entityList:
            # TODO  don't return, for memory concern
            for featureName in featureList:
                if featureName == "VisualizedBy":
                    # scope = "All"
                    print("SYSTEM Action: get image features, save in ", featureList[featureName])
                    # retrieveFeatures(self, featureLabel, entityList, direction)

                    trainImageFeatures = []
                    testImageFeatures = []
                    devImageFeatures = []
                    
                    imageFeatureModel = h5py.File(featureList[featureName], 'w')


                    for subEntity in entityList:
                        subEntityList = [subEntity]
                        # panel image feature = panel_ImageFeatureList[panelVectorName]["VisualizedBy"]
                        panel_ImageFeatureList = knowledgeGraph.retrieveFeatures(featureName, subEntityList, "Out")
                        for panelVectorName in panel_ImageFeatureList:

                            # VGG feature for each image
                            panelImageFeature = panel_ImageFeatureList[panelVectorName]["VisualizedBy"]

                            processedImageFeature = self.processFeature(panelImageFeature)


                            seuqenceID = self.getSequenceIDFromPanel(panelVectorName)
                            if typeTable[seuqenceID][0] == "Train":
                                # print("---panel: ", panelVectorName, " , belongs to seuqence: ", seuqenceID, " set: Train")
                                trainImageFeatures.append(processedImageFeature)
                            elif typeTable[seuqenceID][0] == "Dev":
                                # print("---panel: ", panelVectorName, " , belongs to seuqence: ", seuqenceID, " set: Dev")
                                devImageFeatures.append(processedImageFeature)
                            elif typeTable[seuqenceID][0] == "Test":
                                # print("---panel: ", panelVectorName, " , belongs to seuqence: ", seuqenceID, " set: Test")
                                testImageFeatures.append(processedImageFeature)
                            # print("----Belong to ", seuqenceID, " seuqence")


                        


                        # Debug
                        # print(panelImageFeature.shape)
                        # print(panel_ImageFeatureList.keys())
                        # for panelVectorName in panel_ImageFeatureList:
                        #     print(panel_ImageFeatureList[panelVectorName]["VisualizedBy"].shape)

                    # print(panel_entityFeatureList.keys())
                    imageArrayTrain = np.array(trainImageFeatures)
                    imageArrayDev = np.array(devImageFeatures)
                    imageArrayTest = np.array(testImageFeatures)
                    
                    imageFeatureModel.create_dataset("train/vgg_features",data = imageArrayTrain)
                    imageFeatureModel.create_dataset("dev/vgg_features",data = imageArrayDev)
                    imageFeatureModel.create_dataset("test/vgg_features",data = imageArrayTest)    
                    
                    imageFeatureModel.close()


                # intra-panel level
                elif featureName == "Contain":
                    print("SYSTEM Action: get intra-panel features")
                    
                    trainTextFeatures = []
                    testTextFeatures = []
                    devTextFeatures = []
                    
                    # TODO better structure for store text model (intra-panel has many different things)
                    textFeatureModel = featureList[featureName]
                    textFeatureModel = h5py.File(featureList[featureName]["TextlizedBy"], 'w')


                    for subEntity in entityList:
                        subEntityList = [subEntity]
                        # panel image feature = panel_ImageFeatureList[panelVectorName]["VisualizedBy"]
                        panel_FeatureList = knowledgeGraph.retrieveFeatures(featureName, subEntityList, "Out")
                        print(json.dumps(panel_FeatureList, indent = 4))
                        # for panelVectorName in panel_FeatureList:

                        #     # VGG feature for each image
                        #     panelTextboxFeature = panel_ImageFeatureList[panelVectorName]["VisualizedBy"]

                        #     processedImageFeature = self.processFeature(panelImageFeature)


                        #     seuqenceID = self.getSequenceIDFromPanel(panelVectorName)
                        #     if typeTable[seuqenceID][0] == "Train":
                        #         # print("---panel: ", panelVectorName, " , belongs to seuqence: ", seuqenceID, " set: Train")
                        #         trainImageFeatures.append(processedImageFeature)
                        #     elif typeTable[seuqenceID][0] == "Dev":
                        #         # print("---panel: ", panelVectorName, " , belongs to seuqence: ", seuqenceID, " set: Dev")
                        #         devImageFeatures.append(processedImageFeature)
                        #     elif typeTable[seuqenceID][0] == "Test":
                        #         # print("---panel: ", panelVectorName, " , belongs to seuqence: ", seuqenceID, " set: Test")
                        #         testImageFeatures.append(processedImageFeature)
                        #     # print("----Belong to ", seuqenceID, " seuqence")


                        


                        # Debug
                        # print(panelImageFeature.shape)
                        # print(panel_ImageFeatureList.keys())
                        # for panelVectorName in panel_ImageFeatureList:
                        #     print(panel_ImageFeatureList[panelVectorName]["VisualizedBy"].shape)

                    # print(panel_entityFeatureList.keys())
                    imageArrayTrain = np.array(trainImageFeatures)
                    imageArrayDev = np.array(devImageFeatures)
                    imageArrayTest = np.array(testImageFeatures)
                    
                    imageFeatureModel.create_dataset("train/vgg_features",data = imageArrayTrain)
                    imageFeatureModel.create_dataset("dev/vgg_features",data = imageArrayDev)
                    imageFeatureModel.create_dataset("test/vgg_features",data = imageArrayTest)    
                    
                    imageFeatureModel.close()


                elif featureName == "Panel":
                    print("SYSTEM Action: get panel features")




        else:
            # TODO
            print("SYSTEM Action: don't need to evaluate the understanding (encoding) process, processing the features only.")

            # seuqnece level
            # panel level
            # intra-panel level





    # return []
    def combinePanelFeature(self, panelFeatureList):
        print("System Action: save combined feature as output")
    
    def processFeature(self, panelImageFeature):

        newImageFeature = np.diag(panelImageFeature[0])
        #vgg16_feature.resize(9, 4096)
        # print("---- shape ", panelImageFeature.shape)

        list_array = []

        numList = []
        for num in range(newImageFeature.shape[1]):
            numList.append(num)
        #print(num_list)
        random.shuffle(numList)
        #print(len(num_list))


        remains = newImageFeature.shape[1] % 9
        #print(remains)
        newNumList = numList[0:newImageFeature.shape[1]-remains]
        #print(len(new_num_list))

        new_arr = np.array(newNumList)
        end_list = np.split(new_arr, 9)
        #print(end_list)

        for i in range(9):
            list_array.append(end_list[i].tolist())

        #print("\n".join(map(str, list_array)))

        processedPanelFeature = np.zeros((9,newImageFeature.shape[1]))
        processedPanelFeature[0,:] = np.sum(newImageFeature[list_array[0],:],axis=0)
        processedPanelFeature[1,:] = np.sum(newImageFeature[list_array[1],:],axis=0)
        processedPanelFeature[2,:] = np.sum(newImageFeature[list_array[2],:],axis=0)
        processedPanelFeature[3,:] = np.sum(newImageFeature[list_array[3],:],axis=0)
        processedPanelFeature[4,:] = np.sum(newImageFeature[list_array[4],:],axis=0)
        processedPanelFeature[5,:] = np.sum(newImageFeature[list_array[5],:],axis=0)
        processedPanelFeature[6,:] = np.sum(newImageFeature[list_array[6],:],axis=0)
        processedPanelFeature[7,:] = np.sum(newImageFeature[list_array[7],:],axis=0)
        processedPanelFeature[8,:] = np.sum(newImageFeature[list_array[8],:],axis=0)        

        # print("---- shape ", processedPanelFeature.shape)
        return processedPanelFeature
    
    def getSequenceIDFromPanel(self, panelString):
        sequenceID = ""
        panelStringArray = panelString.split("_")
        sequenceID = panelStringArray[0] + "_" + panelStringArray[1]
        return sequenceID
    
    def mergeFeatureList(self, featureList_1, featureList_2):
        mergedList = featureList_1

        for entityName in featureList_2:
            if entityName in mergedList:
                for relationName in featureList_2:
                    mergedList[entityName][relationName] = featureList_2[relationName]
            else:
                mergedList[entityName] = {}
                for relationName in featureList_2:
                    mergedList[entityName][relationName] = featureList_2[relationName]
        
        return mergedList            
