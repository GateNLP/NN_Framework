import sys
import os
from pathlib import Path
script_path = os.path.abspath(__file__)
dir_path = os.path.dirname(script_path)
parent = Path(dir_path).parent.parent
#print(parent)
sys.path.insert(0, str(parent))
from gatenlp import interact, GateNlpPr, Document
from XSModelManager.ModelManager import ModelManager
from XSModelManager.ModelManager_CANTM import ModelManager_CANTM as ModelManager
from XSNLPReader import GateReader
import logging
from XSNLPReader.readerPostProcessor import CANTMpostProcessor

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
    elif s == 'None':
        return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was


@GateNlpPr
class GateMLTrain:
    def __init__(self):
        self.script_path = os.path.abspath(__file__)
        self.processorLoger = getLogger('processorLoger')

    def start(self, **kwargs):
        print(kwargs)
        #self.all_doc = []
        readerPostProcessor = CANTMpostProcessor(x_fields=['text'], y_field='target')
        self.train_dataIter = GateReader(postProcessor=readerPostProcessor, shuffle=True)

        self.workingSet = kwargs.get('workingSet', '')
        self.instanceType = kwargs.get('instanceType', None)
        self.targetType = kwargs.get('targetType', None)
        self.targetFeature = kwargs.get('targetFeature', None)
        self.targetFile = kwargs.get('targetFile', None)
        self.gpu = str_to_bool(kwargs.get('gpu', 'False'))
        self.model_path = kwargs.get('model_path', None)
        self.splitValidation = kwargs.get('splitValidation', None)
        self.batch_size = int(kwargs.get('batch_size', 16))

        if not self.model_path:
            print('model_path is required\n')
            sys.exit()

        if not ((self.instanceType and self.targetFeature) or self.targetFile):
            print('(instanceType and targetFeature) or targetFile is requried\n')
            sys.exit()

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
        #print(len(self.train_dataIter))
        self.train_dataIter.buildDict()
        print(len(self.train_dataIter.postProcessor.gensim_dict))

        val_dataIter = None
        print(self.train_dataIter.target_labels)
        
        dummy_config = {'MODEL':{'n_classes':len(self.train_dataIter.target_labels), 'vocab_dim':len(self.train_dataIter.postProcessor.gensim_dict)}}
        #print(next(self.train_dataIter))

        self.mm = ModelManager(gpu=self.gpu, config=dummy_config)
        self.mm.genPreBuildModel('CANTM')


        if self.splitValidation:
            self.train_dataIter, val_dataIter = self.mm.splitValidation(self.train_dataIter, val_split=float(self.splitValidation))

        self.mm.train(self.train_dataIter, save_path=self.model_path, valDataIter=val_dataIter, earlyStopping=True, patience=5, batch_size=self.batch_size, warm_up=15)

        self.mm.getTopics(self.train_dataIter.postProcessor.gensim_dict)

    def __call__(self, doc, **kwargs):
        doc_text = doc.text
        current_gate_file_name = doc.features['gate.SourceURL']
        current_gate_file_base_name = os.path.basename(current_gate_file_name)

        workingSet = doc.annset(self.workingSet)

        if self.instanceType:
            instanceSet = workingSet.with_type([self.instanceType])
            for instanceAnno in instanceSet:
                current_instance_text = doc_text[instanceAnno.start:instanceAnno.end]
                current_instance_target_feature = instanceAnno.features[self.targetFeature]
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

