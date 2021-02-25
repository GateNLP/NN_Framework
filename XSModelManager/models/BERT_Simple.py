from transformers import BertModel
from .miscLayer import SimpleHidden
import os
import torch.nn.functional as F
import torch
import torch.nn as nn


class BERT_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.trainable_layers = None
        if 'BERT' in config:
            bert_model_path = config['BERT'].get('bert_path')
            self.bert_dim = int(config['BERT'].get('bert_dim', 768))
            self.trainable_layers = config['BERT'].get('trainable_layers')

        try:
            self.bert = BertModel.from_pretrained(bert_model_path, output_attentions=True,output_hidden_states=True)
        except Exception as e:
            print(e)
            print('load from web')
            #self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True,output_hidden_states=True)
            self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
            #self.trainable_layers = None
            self.bert_dim = 768
            

        if self.trainable_layers:
            print(self.trainable_layers)
            #self.bert = BertModel.from_pretrained(bert_model_path)
            for name, param in self.bert.named_parameters():
                if name in self.trainable_layers:
                    param.requires_grad = True
                    #print(name, param)
                else:
                    param.requires_grad = False
        else:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, x, mask=None):
        if mask == None:
            mask = x != 0
            mask.type(x.type())
        bert_rep = self.bert(x, attention_mask=mask)
        return bert_rep


class BERT_Simple(nn.Module):
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

    def _read_config(self, config):
        if 'MODEL' in config:
            if 'hidden_dim' in config['MODEL']:
                self.hidden_dim = int(config['MODEL'].get('hidden_dim'))
            if 'n_classes' in config['MODEL']:
                self.n_classes = int(config['MODEL'].get('n_classes'))

    def forward(self, batchItem, mask=None):
        x = batchItem[0]
        bert_rep = self.bert_embedding(x, mask)
        cls_att = torch.sum(bert_rep[2][11][:,:,0], 1)
        bert_rep = bert_rep[0]
        bert_rep = bert_rep[:,0]

        hidden = self.hidden_layer(bert_rep)
        out = self.layer_output(hidden)

        y = {
            'y_hat':out,
            'cls_att':cls_att
        }

        return y

