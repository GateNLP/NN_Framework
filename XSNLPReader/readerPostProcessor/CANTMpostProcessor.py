from .PostprocessorBase import ReaderPostProcessorBase

class CANTMpostProcessor(ReaderPostProcessorBase):
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

    def doc2bow(self, tokened):
        return self.gensim_dict.doc2bow(tokened)

    def doc2countHot(self, tokened):
        gensim_bow_doc = self.doc2bow(tokened)
        num_vocab = len(self.gensim_dict)
        doc_vec = [0] * num_vocab
        for item in gensim_bow_doc:
            vocab_idx = item[0]
            vovab_counts = item[1]
            doc_vec[vocab_idx] = vovab_counts
        return doc_vec

    def postProcess4Model(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(current_rawx)

        current_rawx = ' '.join(split_x)
        current_cleand_rawx = self.scholarTextClean(current_rawx, lower=True, strip_html=True, keep_at_mentions=False)
        current_nltk_tokened_rawx = self.nltkTokenizer(current_cleand_rawx)
        current_nltk_tokened_rawx = self.scholarTokenClean(current_nltk_tokened_rawx, stopwords=self.stop_words)
        current_nltk_tokened_rawx = [t for t in current_nltk_tokened_rawx if t != '_']
        if self.gensim_dict:
            current_count_matrix = self.doc2countHot(current_nltk_tokened_rawx)
        else:
            current_count_matrix = None

        current_bert_tokenized = self.bertTokenizer(current_cleand_rawx)
        current_bert_ided = self.bertWord2id(current_bert_tokenized)
        y = sample[self.y_field]
        if self.label2id:
            y = self.label2ids(y)

        output_dict = {}
        output_dict['bert_tokenized'] = current_bert_tokenized
        output_dict['bert_ided'] = current_bert_ided
        output_dict['count_matrix'] = current_count_matrix
        output_dict['nltk_tokened_rawx'] = current_nltk_tokened_rawx
        output_dict['target_label'] = y
        return output_dict



























