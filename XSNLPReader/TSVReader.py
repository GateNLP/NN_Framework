from .CLSReaderBase import CLSReaderBase
import re
import pandas as pd
import os
import glob

class TSVReader(CLSReaderBase):
    def __init__(self, tsv_input, active_fields=None, quotechar='"', error_bad_lines=False, field_delimiter='\t', file_suf='tsv', header=0, skip_blank_lines=True, updateTarget=False, ignoreNoneTarget=False, **kwargs):
        super().__init__(**kwargs)
        self.error_bad_lines = error_bad_lines
        self.quotechar = quotechar
        self.field_delimiter = field_delimiter
        self.active_fields = active_fields
        self.file_suf = file_suf
        self.header = header
        self.skip_blank_lines = skip_blank_lines
        self.updateTarget = updateTarget
        self.ignoreNoneTarget = ignoreNoneTarget

        if self.postProcessor:
            self.target_field = self.postProcessor.y_field

        self.candidate_labels = set()
        self._init_Reader(tsv_input)
        if self.updateTarget:
            self.target_labels += list(self.candidate_labels)
            self.updateTargetLabels2PostProcessor()
        self._reset_iter()

    def _init_Reader(self, tsv_input):
        self.all_ids = []
        self.data_dict = {}
        self.global_data_id = 0

        if os.path.isdir(tsv_input):
            self._read_Folder(tsv_input)
        else:
            self._read_file(tsv_input)

    def _read_file(self, tsv_input):
        tsv_pd_df = pd.read_csv(tsv_input, quotechar=self.quotechar, error_bad_lines=self.error_bad_lines, delimiter=self.field_delimiter, skip_blank_lines=self.skip_blank_lines, header=self.header)
        fileds2read = []
        if self.active_fields:
            fileds2read = self.active_fields
        else:
            fileds2read = list(tsv_pd_df.columns)

        for index, row in tsv_pd_df.iterrows():
            if self.ignoreNoneTarget:
                if row.get(self.target_field) not in self.target_labels:
                    continue

            self.all_ids.append(self.global_data_id)
            self.data_dict[self.global_data_id] = {}
            for each_field in fileds2read:
                self.data_dict[self.global_data_id][each_field] = row[each_field]
            self.global_data_id += 1

            if self.updateTarget and self.postProcessor:
                self.candidate_labels.add(row[self.target_field])


    def _read_Folder(self, tsv_folder_input):
        search_fieds = os.path.join(tsv_folder_input, '*.'+self.file_suf)
        all_file_list = glob.glob(search_fieds)

        for each_file in all_file_list:
            self._read_file(each_file)




