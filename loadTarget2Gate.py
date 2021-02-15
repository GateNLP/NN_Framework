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


@GateNlpPr
class LoadTarget2Gate:
    def __init__(self):
        self.script_path = os.path.abspath(__file__)
        self.processorLoger = getLogger('processorLoger')

    def start(self, **kwargs):
        self.workingSet = kwargs.get('workingSet', '')
        self.targetFeature = kwargs.get('targetFeature', 'target')
        self.targetFile = kwargs.get('targetFile', None)
        self.target2GateType = kwargs.get('target2GateType', None)

        if self.targetFile:
            self.target_dict = {}
            with open(self.targetFile, 'r') as ft:
                for each_line_id, each_line in enumerate(ft):
                    if each_line_id == 0:
                        self.file_suffix = each_line.strip()
                    else:
                        line_tok = each_line.split('\t')
                        self.target_dict[line_tok[0]] = line_tok[1].strip()

    def __call__(self, doc, **kwargs):
        output_set_name = kwargs.get("outputASName", self.workingSet)
        output_set = doc.annset(output_set_name)
        doc_text = doc.text
        current_gate_file_name = doc.features['gate.SourceURL']
        current_gate_file_base_name = os.path.basename(current_gate_file_name)

        current_instance_text = doc_text
        if current_gate_file_base_name in self.target_dict:
            current_instance_target_feature = self.target_dict[current_gate_file_base_name]
            output_feature_map = {self.targetFeature:current_instance_target_feature}
            output_set.add(0, len(current_instance_text), self.target2GateType, output_feature_map)


if __name__ == '__main__':
  interact()

