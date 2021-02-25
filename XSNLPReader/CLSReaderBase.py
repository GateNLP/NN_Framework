import random
import math
import json
import torch
import copy

class CLSReaderBase:
    def __init__(self, postProcessor=None, shuffle=False, config=None, build_dict=False, gensim_dict=None):
        self.label_count_dict = {}
        self.label_weights_list = None
        self._readConfigs(config)
        self.shuffle = shuffle
        self.gensim_dict = gensim_dict

        if postProcessor:
            self.setPostProcessor(postProcessor)
        else:
            self.postProcessor = None
        self.goPoseprocessor = True



    def _readConfigs(self, config):
        self.target_labels = []
        self.field_trans = False
        self.label_trans = False

        if config:
            self.config = config
            if 'TARGET' in config:
                self.target_labels = config['TARGET'].get('labels')
                #print(self.target_labels)

            if 'FIELD_TRANS' in config:
                self.field_trans = True

            if 'LABEL_TRANS' in config:
                self.label_trans = True


    def _transferData(self, current_data, data_name):
        if current_field in self.config[data_name]:
            return self.config[data_name][current_field]
        else:
            return current_field

    def _transferField(self, current_field):
        return self._transferData(current_field, 'FIELD_TRANS')

    def _transferLabel(self, current_label):
        return self._transferData(current_label, 'LABEL_TRANS')



    def setPostProcessor(self, postProcessor):
        self.postProcessor = postProcessor
        self.updateTargetLabels2PostProcessor()


    def updateTargetLabels2PostProcessor(self):
        self.postProcessor.labelsFields = self.target_labels


    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.all_ids)
        self._reset_iter()
        return self

    def __next__(self):
        #print(self.all_ids)
        if self.current_sample_idx < len(self.all_ids):
            current_sample = self._readNextSample()
            self.current_sample_idx += 1
            #print(current_sample)
            return current_sample

        else:
            self._reset_iter()
            raise StopIteration


    def _readNextSample(self):
        current_id = self.all_ids[self.current_sample_idx]
        #print(current_id)
        self.current_sample_dict_id = current_id
        current_sample = self.data_dict[current_id]
        if self.postProcessor and self.goPoseprocessor:
            current_sample = self.postProcessor.postProcess(current_sample)
        return current_sample

    def preCalculateEmbed(self, embd_net, embd_field, dataType=torch.long, device='cuda:0'):
        for sample, _ in self:
            x_embd = sample[embd_field]
            input_tensor = torch.tensor([x_embd], dtype=torch.long, device=device)
            with torch.no_grad():
                embd = embd_net(input_tensor)
            self.data_dict[self.current_sample_dict_id]['embd'] = embd[0].tolist()

        self.postProcessor.embd_ready = True

        #pass


    def __len__(self):
        return len(self.all_ids)

    def _reset_iter(self):
        if self.shuffle:
            random.shuffle(self.all_ids)
        self.current_sample_idx = 0
        #print(self.all_ids)
        self.current_sample_dict_id = self.all_ids[self.current_sample_idx]

    def count_samples(self):
        self.goPoseprocessor = False
        self.label_count_dict = {}
        self.label_count_list = [0]*len(self.postProcessor.labelsFields)
        for item in self:
            #print(item)
            annotation = item[self.target_field]
            annotation_idx = self.postProcessor.labelsFields.index(annotation)
            self.label_count_list[annotation_idx] += 1
            if annotation not in self.label_count_dict:
                self.label_count_dict[annotation] = 0
            self.label_count_dict[annotation] += 1
        print(self.label_count_dict)
        print(self.label_count_list)
        self.goPoseprocessor = True

    def cal_sample_weights(self):
        self.count_samples()
        self.label_weights_list = []
        max_count = max(self.label_count_list)
        for i in range(len(self.label_count_list)):
            current_count = self.label_count_list[i]
            sample_weight = max_count/current_count
            self.label_weights_list.append(sample_weight)
        print(self.label_weights_list)

    def buildDict(self, no_below=3, no_above=0.7, keep_n=5000):
        from gensim.corpora.dictionary import Dictionary
        if 'GENSIM_DICT' in self.config:
            no_below = int(self.config['GENSIM_DICT'].get('no_below', 3))
            no_above = float(self.config['GENSIM_DICT'].get('no_above', 0.7))
            keep_n = int(self.config['GENSIM_DICT'].get('keep_n', 5000))

        ori_pp_mode = copy.deepcopy(self.postProcessor.postProcessMethod)
        ori_go_postprocess = copy.deepcopy(self.goPoseprocessor)
        self.postProcessor.postProcessMethod = 'postProcess4Dict'
        self.goPoseprocessor = True
        self._reset_iter()
        #print(next(self))
        self.gensim_dict = Dictionary(self)
        self._reset_iter()
        self.postProcessor.postProcessMethod = ori_pp_mode
        self.goPoseprocessor = ori_go_postprocess
        self.gensim_dict.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

        
        if self.postProcessor:
            self.postProcessor.gensim_dict = self.gensim_dict




