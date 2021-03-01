import nltk
import os
import re
import string
from transformers import BertTokenizer
from pathlib import Path

class ReaderPostProcessorBase:
    def __init__(self, config=None, stopwords_source=['nltk','snowball', 'mallet'], return_mask=False, max_sent_len=510, gensim_dict=None):
        self.config = config
        self.max_sent_len = max_sent_len
        if 'READER_POSTPROCESSOR' in self.config:
            self.max_sent_len = int(config['READER_POSTPROCESSOR'].get('max_sent_len', 510))

        self.return_mask = return_mask
        punct_chars = list(set(string.punctuation)-set("_"))
        punctuation = ''.join(punct_chars)
        self.pun_replace = re.compile('[%s]' % re.escape(punctuation))
        self.alpha = re.compile('^[a-zA-Z_]+$')
        self.alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
        self.alphanum = re.compile('^[a-zA-Z0-9_]+$')
        self.stopwords_source = stopwords_source
        self.script_path = os.path.abspath(__file__)
        self.parent = os.path.dirname(self.script_path)
        stop_list_dir = os.path.join(self.parent, 'stopwords')
        self._get_stop_words(stop_list_dir)
        self.gensim_dict = None
        self._initPostProcessor()


    def _initPostProcessor(self):
        bert_tokenizer_folder = os.path.join(self.parent, 'bert-base-uncased-tokenizer')
        if not os.path.exists(bert_tokenizer_folder):
            print('bert tokenizer not downloaded, downloading to '+bert_tokenizer_folder)
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_tokenizer_path = Path(bert_tokenizer_folder)
            bert_tokenizer_path.mkdir(parents=True, exist_ok=True)
            self.bert_tokenizer.save_pretrained(bert_tokenizer_folder)
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_folder)

    def _get_stop_words(self, stop_list_dir):
        self.stop_words = set()
        snowball_stopwords_list_file = os.path.join(stop_list_dir, 'snowball_stopwords.txt')
        mallet_stopwords_list_file = os.path.join(stop_list_dir, 'mallet_stopwords.txt')
        scholar_stopwords_list_file = os.path.join(stop_list_dir, 'scholar_stopwords.txt')

        if 'nltk' in self.stopwords_source:
            try:
                from nltk.corpus import stopwords
                self.stop_words = set(stopwords.words('english'))
            except:
                nltk.download('stopwords')
                nltk.download('punkt')
                from nltk.corpus import stopwords
                self.stop_words = set(stopwords.words('english'))
        if 'snowball' in self.stopwords_source:
            with open(snowball_stopwords_list_file, 'r') as fin:
                for line in fin:
                    stop_word = line.strip()
                    self.stop_words.add(stop_word)
        if 'mallet' in self.stopwords_source:
            with open(mallet_stopwords_list_file, 'r') as fin:
                for line in fin:
                    stop_word = line.strip()
                    self.stop_words.add(stop_word)
        if 'scholar' in self.stopwords_source:
            with open(scholar_stopwords_list_file, 'r') as fin:
                for line in fin:
                    stop_word = line.strip()
                    self.stop_words.add(stop_word)



    def scholarTokenClean(self, tokens, keep_numbers=False, keep_alphanum=False, min_length=3, stopwords=None):
        #text = self.clean_text(text, strip_html, lower, keep_emails, keep_at_mentions, keep_pun=keep_pun)
        #if useNLTKTokenizer:
        #    tokens = self.nltkTokenizer(text)
        #else:
        #    tokens = text.split()

        if stopwords is not None:
            tokens = ['_' if t in stopwords else t for t in tokens]

        # remove tokens that contain numbers
        if not keep_alphanum and not keep_numbers:
            tokens = [t if self.alpha.match(t) else '_' for t in tokens]

        # or just remove tokens that contain a combination of letters and numbers
        elif not keep_alphanum:
            tokens = [t if self.alpha_or_num.match(t) else '_' for t in tokens]

        # drop short tokens
        if min_length > 0:
            tokens = [t if len(t) >= min_length else '_' for t in tokens]

        return tokens

    def scholarTextClean(self, text, strip_html=False, lower=False, keep_emails=True, keep_at_mentions=True, keep_pun=True):
        # remove html tags
        if strip_html:
            text = re.sub(r'<[^>]+>', '', text)
        else:
            # replace angle brackets
            text = re.sub(r'<', '(', text)
            text = re.sub(r'>', ')', text)
        # lower case
        if lower:
            text = text.lower()
        # eliminate email addresses
        if not keep_emails:
            text = re.sub(r'\S+@\S+', ' _ ', text)
        # eliminate @mentions
        if not keep_at_mentions:
            text = re.sub(r'\s@\S+', ' _ ', text)
        # break off single quotes at the ends of words
        text = re.sub(r'\s\'', ' _ ', text)
        text = re.sub(r'\'\s', ' _ ', text)
        if not keep_pun:
            # remove periods
            text = re.sub(r'\.', ' _ ', text)
            text = self.pun_replace.sub(' _ ', text)
        text = text.strip()
        return text

    def gen_ngram(self, tokens, n, n_gram_only=False):
        n_grams = []
        sentLen = len(tokens)
        s = n-1
        if sentLen > s:
            for i in range(0,sentLen-s):
                current_n_gram = []
                #stop_signal = False
                for j in range(n):
                    current_token = tokens[i+j]
                    if current_token == '_':
                        break
                    else:
                        current_n_gram.append(current_token)
                if n_gram_only:
                    if len(current_n_gram) == n:
                       n_grams.append('_'.join(current_n_gram))
                else:
                    if len(current_n_gram) > 1:
                        n_grams.append('_'.join(current_n_gram))
                    elif len(current_n_gram) == 1:
                        n_grams += current_n_gram
        return n_grams

    def postProcess(self, sample):#required override in inherited class
        pass

    def _removeSingleList(self, y):
        if len(y) == 1:
            return y[0]
        else:
            return y

    def _get_sample(self, sample, sample_field):
        current_rawx = sample[sample_field]
        #if self.keep_case == False:
        #    current_rawx = current_rawx.lower()
        return current_rawx


    def label2ids(self, label):
        label_index = self.labelsFields.index(label)
        return label_index


    def spaceTokenizer(self, text):
        return text.split()

    def nltkTokenizer(self, text):
        return nltk.word_tokenize(text)

    def bertTokenizer(self, text):
        tokened = self.bert_tokenizer.tokenize(text)
        #print(tokened)
        #ided = self.bert_tokenizer.encode_plus(tokened, max_length=100, pad_to_max_length=True, is_pretokenized=True, add_special_tokens=True)['input_ids']
        #print(ided)
        return tokened

    def bertWord2id(self,tokened, add_special_tokens=True, max_sent_len=None):
        if max_sent_len:
            encoded = self.bert_tokenizer.encode_plus(tokened, max_length=max_sent_len, padding='max_length', is_pretokenized=True, add_special_tokens=add_special_tokens, truncation=True)
        else:
        #encoded = self.bert_tokenizer.encode_plus(tokened, max_length=self.max_sent_len, pad_to_max_length=True, is_pretokenized=True, add_special_tokens=add_special_tokens, truncation=True)
            encoded = self.bert_tokenizer.encode_plus(tokened, max_length=self.max_sent_len, padding='max_length', is_pretokenized=True, add_special_tokens=add_special_tokens, truncation=True)
        #print(encoded)
        ided = encoded['input_ids']
        if self.return_mask:
            mask = encoded['attention_mask']
            return ided, mask
        else:
            return ided

    def postProcess4Dict(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(current_rawx)

        current_rawx = ' '.join(split_x)
        current_cleand_rawx = self.scholarTextClean(current_rawx, lower=True, strip_html=True, keep_at_mentions=False)
        current_nltk_tokened_rawx = self.nltkTokenizer(current_cleand_rawx)
        current_nltk_tokened_rawx = self.scholarTokenClean(current_nltk_tokened_rawx, stopwords=self.stop_words)
        current_nltk_tokened_rawx = [t for t in current_nltk_tokened_rawx if t != '_']

        return current_nltk_tokened_rawx




