from .CLSReaderBase import CLSReaderBase
import re
import pandas as pd
import os
import glob


class GateReader(CLSReaderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_Reader()

    def _init_Reader(self):
        self.all_ids = []
        self.data_dict = {}
        self.global_data_id = 0
        #self.target_labels = []

    def finaliseReader(self):
        self.updateTargetLabels2PostProcessor()
        self._reset_iter()


    def addSample(self, text, target, anno_start=0, anno_end=1):
        self.global_data_id += 1
        self.all_ids.append(self.global_data_id)
        tmp_dict = {
                'text': text,
                'target': target,
                'anno_start':anno_start,
                'anno_end':anno_end,
                }
        self.data_dict[self.global_data_id] = tmp_dict

        if target not in self.target_labels:
            self.target_labels.append(target)



