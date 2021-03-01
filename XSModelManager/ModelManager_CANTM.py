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
from .ModelManager import ModelManager


class ModelManager_CANTM(ModelManager):
    def __init__(self, **kwargs):
        self.gensim_dict = None
        super().__init__(**kwargs)

    def optimiseNet(self, each_batch_output):
        self.optimizer.zero_grad()
        model_pred = each_batch_output['model_output']['y_hat']
        gold_target = each_batch_output['processed_batch_item'][1]
        loss = each_batch_output['model_output']['loss']
        cls_loss = each_batch_output['model_output']['cls_loss']
        log_y_hat_rec_loss = each_batch_output['model_output']['log_y_hat_rec_loss']
        loss.backward()
        self.optimizer.step()

        #print(each_batch_output['model_output'])

        loss_value = float(cls_loss.data.item())
        #loss_value = float(log_y_hat_rec_loss.data.item())
        return loss_value

    def train(self, trainDataIter, **kwargs):
        self.gensim_dict = trainDataIter.postProcessor.gensim_dict
        self.train_default(trainDataIter, **kwargs)


    def setOptimiser(self):
        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = None


    def getTopics(self, ntop=10, cache_path=None):
        classTermMatrix = self.net.get_class_topics()
        classTopicWordList = self.getTopicList(classTermMatrix, ntop=ntop)
        print('!!!!class topics!!!!!')
        for topic_idx, topic_words in enumerate(classTopicWordList):
            output_line = self.target_labels[topic_idx] + ' ' + ' '.join(topic_words)
            print(output_line)

        print('!!!!top class regularized topics!!!!!')
        x_onlyTermMatrix = self.net.get_x_only_topics()
        x_onlyWeightMatrix = self.net.get_class_regularize_topics_weights()
        x_onlyTopicWordList = self.getTopicList(x_onlyTermMatrix, ntop=ntop)
        self.getTopNClassTopics(x_onlyWeightMatrix, x_onlyTopicWordList)

        print('!!!!top class aware topics!!!!!')
        xy_TermMatrix = self.net.get_topics()
        xy_WeightMatrix = self.net.get_class_aware_topics_weights()
        xyTopicWordList = self.getTopicList(xy_TermMatrix, ntop=ntop)

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

    def getTopicList(self, termMatrix, ntop=10, outputFile=None):
        topicWordList = []
        for each_topic in termMatrix:
            trans_list = list(enumerate(each_topic.cpu().numpy()))
            trans_list = sorted(trans_list, key=lambda k: k[1], reverse=True)
            topic_words = [self.gensim_dict[item[0]] for item in trans_list[:ntop]]
            #print(topic_words)
            topicWordList.append(topic_words)
        if outputFile:
            self.saveTopic(topicWordList, outputFile)
        return topicWordList

    def save_checkpoint(self, save_path, best_score, epoch, save_entire=False):
        self.save_checkpoint_default(save_path, best_score, epoch, save_entire=save_entire)
        gensim_dict_save_path = os.path.join(save_path, 'gensim_dict.pt')
        save_dict = {'gensim_dict': self.gensim_dict}
        torch.save(save_dict, gensim_dict_save_path)

    def load_model(self, load_path):
        entrie_load_path = os.path.join(load_path, 'model.net')
        self.net = torch.load(entrie_load_path)
        self.load_checkpoint(load_path)
        self.net.to(self.device)

        gensim_dict_check_point_load_path = os.path.join(load_path, 'gensim_dict.pt')
        gensim_dict_check_point = torch.load(gensim_dict_check_point_load_path)
        self.gensim_dict = gensim_dict_check_point['gensim_dict']





