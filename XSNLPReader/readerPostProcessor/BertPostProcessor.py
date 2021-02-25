from .PostprocessorBase import ReaderPostProcessorBase

class BertPostProcessor(ReaderPostProcessorBase):
    def __init__(self, x_fields, y_field, word2id=True, label2id=True, **kwargs):
        super().__init__(**kwargs)
        self.y_field = y_field
        self.x_fields = x_fields
        self.word2id = word2id
        self.label2id = label2id
        self.postProcessMethod = 'postProcess4Model'


    def postProcess(self, sample):
        if self.postProcessMethod == 'postProcess4Model':
            return self.postProcess4Model(sample)
        elif self.postProcessMethod == 'postProcess4Dict':
            return self.postProcess4Dict(sample)
        elif self.postProcessMethod == 'postProcess4GATEapply':
            return self.postProcess4GATEapply(sample)
        else:
            return sample



    def postProcess4Model(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(str(current_rawx))
        current_rawx = ' '.join(split_x)
        current_cleand_rawx = self.scholarTextClean(current_rawx, lower=True)
        current_bert_tokenized = self.bertTokenizer(current_cleand_rawx)
        current_bert_ided = self.bertWord2id(current_bert_tokenized)

        x = current_bert_ided
        y = sample[self.y_field]

        if self.label2id:
            y = self.label2ids(y)

        return x,y


    def postProcess4GATEapply(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(str(current_rawx))
        current_rawx = ' '.join(split_x)
        current_cleand_rawx = self.scholarTextClean(current_rawx, lower=True)
        current_bert_tokenized = self.bertTokenizer(current_cleand_rawx)

        y = sample[self.y_field]

        if self.label2id:
            y = self.label2ids(y)
        return sample, current_bert_tokenized, y 








