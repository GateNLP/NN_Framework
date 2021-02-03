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
class GateMLTrain:
    def __init__(self):
        self.script_path = os.path.abspath(__file__)
        self.processorLoger = getLogger('processorLoger')

    def start(self, **kwargs):
        #print(kwargs)
        #self.all_doc = []
        readerPostProcessor = BertPostProcessor(x_fields=['text'], y_field='target')
        self.train_dataIter = GateReader(postProcessor=readerPostProcessor, shuffle=True)

        self.workingSet = kwargs.get('workingSet', '')
        self.instanceType = kwargs.get('instanceType', None)
        self.targetType = kwargs.get('targetType', None)
        self.targetFeature = kwargs.get('targetFeature', None)
        self.targetFile = kwargs.get('targetFile', None)
        self.gpu = str_to_bool(kwargs.get('gpu', 'False'))
        self.model_path = kwargs.get('model_path')

        if self.targetFile:
            self.target_dict = {}
            with open(self.targetFile, 'r') as ft:
                for each_line_id, each_line in enumerate(ft):
                    if each_line_id == 0:
                        self.file_suffix = each_line.strip()
                    else:
                        line_tok = each_line.split('\t')
                        self.target_dict[line_tok[0]] = line_tok[1].strip()

    def finish(self, **kwargs):
        self.train_dataIter.finaliseReader()
        print(len(self.train_dataIter))

        val_dataIter = None
        print(self.train_dataIter.target_labels)
        
        dummy_config = {'MODEL':{'n_classes':len(self.train_dataIter.target_labels)}}

        self.mm = ModelManager(gpu=self.gpu, config=dummy_config)
        self.mm.genPreBuildModel()


        if 'splitValidation' in kwargs:
            self.train_dataIter, val_dataIter = self.mm.splitValidation(self.train_dataIter, val_split=float(kwargs.get('splitValidation')))

        self.mm.train(self.train_dataIter, save_path=self.model_path, valDataIter=val_dataIter, earlyStopping=True, patience=5)

    def __call__(self, doc, **kwargs):
        doc_text = doc.text
        #print(doc._name)
        #print(doc.features)
        current_gate_file_name = doc.features['gate.SourceURL']
        current_gate_file_base_name = os.path.basename(current_gate_file_name)
        #print(current_gate_file_base_name)

        workingSet = doc.annset(self.workingSet)

        if self.instanceType:
            instanceSet = workingSet.with_type([self.instanceType])
            for instanceAnno in instanceSet:
                #print(instanceAnno)
                #print(instanceAnno.start)
                #print(instanceAnno.end)
                current_instance_text = doc_text[instanceAnno.start:instanceAnno.end]
                current_instance_target_feature = instanceAnno.features[self.targetFeature]
                #print(current_instance_text, current_instance_target_feature)
                self.train_dataIter.addSample(current_instance_text, current_instance_target_feature)
        elif self.targetFile:
            current_instance_text = doc_text
            if current_gate_file_base_name in self.target_dict:
                current_instance_target_feature = self.target_dict[current_gate_file_base_name] 
                self.train_dataIter.addSample(current_instance_text, current_instance_target_feature)
            else:
                infomessage = 'no target found discard '+current_gate_file_name
                self.processorLoger.info(infomessage)




if __name__ == '__main__':
  interact()

