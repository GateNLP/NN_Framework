import sys
import os
from gatenlp import interact, GateNlpPr, Document
from XSModelManager.ModelManager import ModelManager
from XSNLPReader import GateReader
import logging
from XSNLPReader.readerPostProcessor import BertPostProcessor

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
        self.targetType = kwargs.get('targetType', None)
        self.targetFeature = kwargs.get('targetFeature', 'target')
        self.gpu = str_to_bool(kwargs.get('gpu', 'False'))
        self.model_path = kwargs.get('model_path')
        self.targetFile = kwargs.get('targetFile', None)
        self.resultsExportFile = kwargs.get('resultsExportFile', None)

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

        test_dataIter.goPoseprocessor = False
        for each_sample_id, each_sample in enumerate(test_dataIter):
            pred_score = apply_output_dict['all_prediction'][each_sample_id]
            pred_label_string = apply_output_dict['all_pred_label_string'][each_sample_id]
            anno_start = each_sample['anno_start']
            anno_end = each_sample['anno_end']
            output_feature_map = {'pred_score':pred_score, self.targetFeature:pred_label_string}
            output_set.add(anno_start, anno_end, outputType, output_feature_map)
            if self.resultsExportFile:
                result_export_line = current_gate_file_base_name + '\t' + str(anno_start) + '\t' + str(anno_end) + '\t' + pred_label_string + '\t' + doc_text[anno_start:anno_end] + '\n'
                self.f_results_export.write(result_export_line)





    





if __name__ == '__main__':
  interact()

