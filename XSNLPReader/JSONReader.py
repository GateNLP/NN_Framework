from .CLSReaderBase import CLSReaderBase
import re
import pandas as pd
import os
import glob
import json

class JSONReader(CLSReaderBase):
    def __init__(self, json_input, updateTarget=False, **kwargs):
        super().__init__(**kwargs)
        self.updateTarget = updateTarget

        if self.postProcessor:
            self.target_field = self.postProcessor.y_field

        self.candidate_labels = set()
        self._init_Reader(json_input)
        if self.updateTarget:
            self.target_labels += list(self.candidate_labels)
            self.updateTargetLabels2PostProcessor()
        self._reset_iter()

    def _init_Reader(self, json_input):
        self.all_ids = []
        self.data_dict = {}
        self.global_data_id = 0

        self._read_file(json_input)

    def _read_file(self, json_input):
        with open(json_input, 'r') as f_json:
            json_data = json.load(f_json)
        for each_json_data in json_data:
            self.all_ids.append(self.global_data_id)
            self.data_dict[self.global_data_id] = {}
            self.data_dict[self.global_data_id] = each_json_data
            self.global_data_id += 1

            if self.updateTarget and self.postProcessor:
                self.candidate_labels.add(each_json_data[self.target_field])


