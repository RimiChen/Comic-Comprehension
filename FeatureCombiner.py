import sys, os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Knowledge_from_comics'))
from KnowledgeGraph import *

class FeatureCombiner:
    def __init__(self):
        self.printDesciption()
        
        # Don't save this because of memory
        # self. knowledgeGraph = knowledgeGraph


    # Know default information
    def printDesciption(self):
        print("SYSTEM Action: Create object from FeatureCombiner class.")
        print("SYSTEM Info: combine features during this processor.")
    
    def combineFeatures(self, knowledgeGraph: KnowledgeGraph, featureList):
        print("SYSTEM Action: retrieve needed features and combine as model, save image features as a model, text features outside")

        


        # Sequence Level
        
        # Panel level
        entityList = knowledgeGraph.accessFeatures("Panel")
        print(json.dumps(entityList, indent = 4))
        panel_entityFeatureList = {}
        
        # for targetEntity in entityList:

        for featureName in featureList:
            if featureName == "VisualizedBy":
                scope = "All"
                print("SYSTEM Action: get image features")
                # retrieveFeatures(self, featureLabel, entityList, direction)
                panel_ImageFeatureList = knowledgeGraph.retrieveFeatures(featureName, entityList, "Out")
                panel_entityFeatureList  = self.mergeFeatureList(panel_entityFeatureList , panel_ImageFeatureList)
                
                # print(panel_entityFeatureList.keys())



            elif featureName == "Text":
                print("SYSTEM Action: get text features")

            elif featureName == "Panel":
                print("SYSTEM Action: get panel features")

        # intra-panel level
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
