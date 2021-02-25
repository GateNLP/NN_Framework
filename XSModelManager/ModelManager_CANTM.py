import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from .BatchIter import BatchIter
import logging
import copy
import math
import random
from pathlib import Path
from .ModelManager import ModelManager, getLogger


class ModelManager_CANTM(ModelManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def optimiseNet(self, each_batch_output):
        self.optimizer.zero_grad()
        model_pred = each_batch_output['model_output']['y_hat']
        gold_target = each_batch_output['processed_batch_item'][1]
        loss = each_batch_output['model_output']['loss']
        cls_loss = each_batch_output['model_output']['cls_loss']

        loss.backward()
        self.optimizer.step()

        loss_value = float(cls_loss.data.item())
        return loss_value


    def setOptimiser(self):
        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = None


    def getTopics(self, gensim_dict, ntop=10, cache_path=None):
        classTermMatrix = self.net.get_class_topics()
        classTopicWordList = self.getTopicList(gensim_dict, classTermMatrix, ntop=ntop)
        print('!!!!class topics!!!!!')
        for topic_idx, topic_words in enumerate(classTopicWordList):
            output_line = self.target_labels[topic_idx] + ' ' + ' '.join(topic_words)
            print(output_line)

        print('!!!!top class regularized topics!!!!!')
        x_onlyTermMatrix = self.net.get_x_only_topics()
        x_onlyWeightMatrix = self.net.get_class_regularize_topics_weights()
        x_onlyTopicWordList = self.getTopicList(gensim_dict, x_onlyTermMatrix, ntop=ntop)
        self.getTopNClassTopics(x_onlyWeightMatrix, x_onlyTopicWordList)

        print('!!!!top class aware topics!!!!!')
        xy_TermMatrix = self.net.get_topics()
        xy_WeightMatrix = self.net.get_class_aware_topics_weights()
        xyTopicWordList = self.getTopicList(gensim_dict, xy_TermMatrix, ntop=ntop)

        self.getTopNClassTopics(xy_WeightMatrix, xyTopicWordList)


        #print(x_onlyTopicWordList)
        #print(xyTopicWordList)

    def getTopNClassTopics(self, topicWeightList, topicWordList, ntop=5):
        for each_class_id, each_class_topic_weight in enumerate(topicWeightList):
            print('!!!!!'+self.target_labels[each_class_id]+'!!!!!')
            current_class_topic_weight = list(enumerate(each_class_topic_weight.cpu().numpy()))
            current_class_topic_weight = sorted(current_class_topic_weight, key=lambda k: k[1], reverse=True)
            for each_topic_id, each_topic_weight in current_class_topic_weight[:ntop]:
                print(topicWordList[each_topic_id])





    def getTopicList(self, gensim_dict, termMatrix, ntop=10, outputFile=None):
        topicWordList = []
        for each_topic in termMatrix:
            trans_list = list(enumerate(each_topic.cpu().numpy()))
            trans_list = sorted(trans_list, key=lambda k: k[1], reverse=True)
            topic_words = [gensim_dict[item[0]] for item in trans_list[:ntop]]
            #print(topic_words)
            topicWordList.append(topic_words)
        if outputFile:
            self.saveTopic(topicWordList, outputFile)
        return topicWordList





