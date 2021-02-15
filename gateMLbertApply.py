import sys
import os
from gatenlp import interact, GateNlpPr, Document
from XSModelManager.ModelManager import ModelManager
from XSNLPReader import GateReader
import logging
from XSNLPReader.readerPostProcessor import BertPostProcessor
import re
import torch
import copy

def getLogger(name, terminator='\n'):
    logger = logging.getLogger(name)
    cHandle = logging.StreamHandler()
    cHandle.terminator = terminator
    logger.addHandler(cHandle)
    return logger


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was


def single_att_reconstruction(single_token_list, attention_weights, cut_point=10):
    #print(single_token_list)
    bert_hash_token = re.compile('##.*')
    recon_token_list = []
    recon_att_weight_list = []
    attention_weights = attention_weights[1:]
    #print(single_token_list)
    #print(len(single_token_list), len(attention_weights))
    single_token_list = single_token_list[:len(attention_weights)]
    #print(len(single_token_list), len(attention_weights))
    for i in range(len(single_token_list)):
        current_token = single_token_list[i]
        current_att_weight = attention_weights[i]
        m = bert_hash_token.match(current_token)
        if m:
            current_reconst_token = current_token[2:]
            recon_token_list[-1] += current_reconst_token
            recon_att_weight_list[-1].append(current_att_weight)
        else:
            recon_token_list.append(current_token)
            recon_att_weight_list.append([current_att_weight])

    recon_att_weight_list_new = []
    for item in recon_att_weight_list:
        if len(item) > 1:
            recon_att_weight_list_new.append(sum(item)/len(item))
        else:
            recon_att_weight_list_new.append(item[0])

    normalise_att_weights = torch.nn.functional.softmax(torch.tensor(recon_att_weight_list_new))
    topn_values, topn_indices = torch.sort(normalise_att_weights, descending=True)
    topn_values = topn_values[:cut_point]
    topn_indices = topn_indices[:cut_point]

    return recon_token_list, topn_indices.to('cpu').detach().numpy(), topn_values.to('cpu').detach().numpy()

def construct_offset_id(text, recon_token_list):
    off_set_dict = {}
    text = text.lower()
    token_id = 0
    off_set_id = 0
    stored_off_set = 0
    found=False
    #print(text)
    #print(recon_token_list)
    while token_id != (len(recon_token_list)):
        current_token = recon_token_list[token_id]
        current_string = ''
        for current_off_set in range(off_set_id, len(text)):
            current_char = text[current_off_set]
            current_string += current_char
            if current_string == current_token:
                off_set_dict[token_id] = [copy.deepcopy(off_set_id), copy.deepcopy(current_off_set+1)]
                off_set_id = current_off_set+1
                token_id += 1
                found = True
                stored_off_set = copy.deepcopy(off_set_id)
                break

        if found:
            found = False
        else:
            off_set_id += 1

        if off_set_id > len(text):
            token_id += 1
            off_set_id = copy.deepcopy(stored_off_set)
            #print(current_token)
    return off_set_dict



@GateNlpPr
class GateMLTest:
    def __init__(self):
        self.script_path = os.path.abspath(__file__)
        self.processorLoger = getLogger('processorLoger')

    def start(self, **kwargs):
        self.readerPostProcessor = BertPostProcessor(x_fields=['text'], y_field='target')
        #self.train_dataIter = GateReader(postProcessor=readerPostProcessor)

        self.workingSet = kwargs.get('workingSet', '')
        self.instanceType = kwargs.get('instanceType', None)
        #self.targetType = kwargs.get('targetType', None)
        self.targetFeature = kwargs.get('targetFeature', 'target')
        self.gpu = str_to_bool(kwargs.get('gpu', 'False'))
        self.model_path = kwargs.get('model_path')
        self.targetFile = kwargs.get('targetFile', None)
        self.resultsExportFile = kwargs.get('resultsExportFile', None)
        self.target2GateType = kwargs.get('target2GateType', None)

        self.mm = ModelManager(gpu=self.gpu)
        self.mm.load_model(self.model_path)

        if self.targetFile:
            self.target_dict = {}
            with open(self.targetFile, 'r') as ft:
                for each_line_id, each_line in enumerate(ft):
                    if each_line_id == 0:
                        self.file_suffix = each_line.strip()
                    else:
                        line_tok = each_line.split('\t')
                        self.target_dict[line_tok[0]] = line_tok[1].strip()

        if self.resultsExportFile:
            self.f_results_export = open(self.resultsExportFile, 'w')



    def finish(self, **kwargs):
        if self.resultsExportFile:
            self.f_results_export.close()



    def __call__(self, doc, **kwargs):
        output_set_name = kwargs.get("outputASName", "GATEML")
        doc_text = doc.text
        current_gate_file_name = doc.features['gate.SourceURL']
        current_gate_file_base_name = os.path.basename(current_gate_file_name)

        #print(doc._name)
        workingSet = doc.annset(self.workingSet)
        config = {'TARGET':{'labels':self.mm.target_labels}}

        test_dataIter = GateReader(postProcessor=self.readerPostProcessor, config=config)

        outputType = 'MLpred'

        if self.instanceType:
            outputType = self.instanceType

            instanceSet = workingSet.with_type([self.instanceType])
            for instanceAnno in instanceSet:
                current_instance_text = doc_text[instanceAnno.start:instanceAnno.end]
                if self.targetFeature:
                    current_instance_target_feature = instanceAnno.features[self.targetFeature]
                else:
                    ### add a dummy target
                    current_instance_target_feature = self.mm.target_labels[0]
                test_dataIter.addSample(current_instance_text, current_instance_target_feature, anno_start=instanceAnno.start, anno_end=instanceAnno.end)
        else:
            current_instance_text = doc_text
            current_instance_target_feature = self.mm.target_labels[0]
            if self.targetFile:
                if current_gate_file_base_name in self.target_dict:
                    current_instance_target_feature = self.target_dict[current_gate_file_base_name]
            test_dataIter.addSample(current_instance_text, current_instance_target_feature, anno_start=0, anno_end=len(current_instance_text))



        test_dataIter._reset_iter()


        apply_output_dict = self.mm.apply(test_dataIter)
        output_set = doc.annset(output_set_name)
        output_set.clear()
        #print(apply_output_dict['all_cls_att'])


        test_dataIter.postProcessor.postProcessMethod = 'postProcess4GATEapply'
        for each_sample_id, dataIterItem in enumerate(test_dataIter):
            each_sample = dataIterItem[0]
            bert_tokenized = dataIterItem[1]
            pred_score = apply_output_dict['all_prediction'][each_sample_id]
            pred_label_string = apply_output_dict['all_pred_label_string'][each_sample_id]
            cls_att = apply_output_dict['all_cls_att'][each_sample_id]
            anno_start = each_sample['anno_start']
            anno_end = each_sample['anno_end']
            output_feature_map = {'pred_score':pred_score, self.targetFeature:pred_label_string}
            output_set.add(anno_start, anno_end, outputType, output_feature_map)

            #recon_token_list, topn_indices, topn_values = single_att_reconstruction(bert_tokenized, cls_att)
            #off_set_dict = construct_offset_id(doc_text, recon_token_list)

            #print(len(recon_token_list), len(token_offset_list))



            if self.resultsExportFile:
                result_export_line = current_gate_file_base_name + '\t' + str(anno_start) + '\t' + str(anno_end) + '\t' + pred_label_string + '\t' + doc_text[anno_start:anno_end] + '\n'
                self.f_results_export.write(result_export_line)

            if not self.instanceType and self.targetFile and self.target2GateType:
                output_feature_map = {self.targetFeature:each_sample['target']}
                output_set.add(anno_start, anno_end, self.target2GateType, output_feature_map)

            ###export attention
            #for att_id, att_word_index in enumerate(topn_indices):
            #    att_score = topn_values[att_id]
            #    if att_word_index in off_set_dict:
            #        att_feature_map = {'score':str(att_score)}
            #        print(off_set_dict[att_word_index][0], off_set_dict[att_word_index][1], len(doc_text))
            #        output_set.add(off_set_dict[att_word_index][0], off_set_dict[att_word_index][1], 'attentions', att_feature_map)


        test_dataIter.postProcessor.postProcessMethod = 'postProcess4Model'





    





if __name__ == '__main__':
  interact()

