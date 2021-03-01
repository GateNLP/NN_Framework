from transformers import BertModel
from .miscLayer import SimpleHidden
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from .BERT_Simple import BERT_Embedding


class BERT_Rel_Simple(nn.Module):
    def __init__(self, config={}, **kwargs):
        super().__init__()
        self._init_params()
        self._read_config(config)

        self.bert_embedding = BERT_Embedding(config)
        bert_dim = self.bert_embedding.bert_dim
        self.hidden_layer = SimpleHidden(bert_dim, self.hidden_dim)
        self.layer_output = torch.nn.Linear(self.hidden_dim, self.n_classes)


    def _init_params(self):
        self.hidden_dim = 300
        self.n_classes = 2
        self.sep_idx = 201

    def _read_config(self, config):
        if 'MODEL' in config:
            if 'hidden_dim' in config['MODEL']:
                self.hidden_dim = int(config['MODEL'].get('hidden_dim'))
            if 'n_classes' in config['MODEL']:
                self.n_classes = int(config['MODEL'].get('n_classes'))
            if 'sep_idx' in config['MODEL']:
                self.sep_idx = int(config['MODEL'].get('sep_idx'))


    def forward(self, batchItem, mask=None):
        x = batchItem[0]
        sent_sep_idx = (x==102).nonzero(as_tuple=True)[1]#[0][1]#[0][1]
        #print(sent_sep_idx)
        bert_rep = self.bert_embedding(x, mask)
        cls_att = torch.sum(bert_rep[2][11][:,:,0], 1)
        bert_rep_raw = bert_rep[0]
        #sent_sep_idx = (bert_rep==102).nonzero()[0][1]
        #sent_sep_idx = (bert_rep==101)#.nonzero(as_tuple=True)#[0][1]
        #print(sent_sep_idx)
        #print(bert_rep_raw.shape)
        each_sep_out = []
        for each_bert_id, each_bert_rep in enumerate(bert_rep_raw):
            current_sep_rep = each_bert_rep[sent_sep_idx[each_bert_id]]
            #print(current_sep_rep.shape)
            each_sep_out.append(current_sep_rep.unsqueeze(0))
            #each_sep_out.append()

        #bert_rep = torch.index_select(bert_rep, 1, sent_sep_idx)
        bert_rep = torch.cat(each_sep_out, 0)
        #print(bert_rep.shape)

        hidden = self.hidden_layer(bert_rep)
        out = self.layer_output(hidden)

        y = {
            'y_hat':out,
            'cls_att':cls_att
        }

        return y

