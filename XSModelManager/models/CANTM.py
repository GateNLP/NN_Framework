from transformers import BertModel
from torch.nn import init
from .miscLayer import SimpleHidden
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from .BERT_Simple import BERT_Embedding
import math
import copy

def kld(mu, log_sigma):
    """log q(z) || log p(z).
    mu: batch_size x dim
    log_sigma: batch_size x dim
    """
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        if len(input) == 1:
            return input[0]
        return input


class LinearClassifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super().__init__()
        self.layer_output = torch.nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        out = self.layer_output(x)
        return out

class NonLinHidden(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        hidden = F.leaky_relu(self.hidden1(x))
        return hidden

class Topics(nn.Module):
    def __init__(self, k, vocab_size, bias=True):
        super(Topics, self).__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.topic = nn.Linear(k, vocab_size, bias=bias)

    def forward(self, logit):
        # return the log_prob of vocab distribution
        return torch.log_softmax(self.topic(logit), dim=-1)

    def get_topics(self):
        #print('hey')
        #print(self.topic.weight)
        return torch.softmax(self.topic.weight.data.transpose(0, 1), dim=-1)

    def get_topic_word_logit(self):
        """topic x V.
        Return the logits instead of probability distribution
        """
        return self.topic.weight.transpose(0, 1)




class CANTM(nn.Module):
    def __init__(self, config={}, **kwargs):
        super().__init__()
        self._init_params()
        self._read_config(config)
        
        vocab_dim = int(config['MODEL'].get('vocab_dim', 2000))

        self.banlance_lambda = float(math.ceil(vocab_dim/self.n_classes))

        self.bert_embedding = BERT_Embedding(config)
        bert_dim = self.bert_embedding.bert_dim

        ##############M1###########################################
        self.mu_z1 = nn.Linear(bert_dim, self.z_dim)
        self.log_sigma_z1 = nn.Linear(bert_dim, self.z_dim)
        self.x_only_topics = Topics(self.z_dim, vocab_dim)
        self.xy_classifier = LinearClassifier(self.z_dim, self.n_classes)
        self.class_topics = Topics(self.n_classes, vocab_dim)
        self.class_criterion = nn.CrossEntropyLoss()

        #############M2############################################
        self.x_y_dim = bert_dim + self.n_classes
        #self.z_y_dim = self.z_dim + self.n_classes
        self.x_y_hidden = NonLinHidden(self.x_y_dim, self.hidden_dim)
        #self.z_y_hidden = NonLinHidden(self.z_y_dim, self.ntopics)

        self.mu_z2 = nn.Linear(self.hidden_dim, self.z_dim)
        self.log_sigma_z2 = nn.Linear(self.hidden_dim, self.z_dim)
        self.xy_topics = Topics(self.z_dim, vocab_dim)
        self.z2y_classifier = LinearClassifier(self.z_dim, self.n_classes)

        ############################################################
        self.h_to_z = Identity()
        self.reset_parameters()



        #self.hidden_layer = SimpleHidden(bert_dim, self.hidden_dim)
        #self.layer_output = torch.nn.Linear(self.hidden_dim, self.n_classes)


    def _init_params(self):
        self.mode = 'train' # 'apply, update_catopic'
        self.hidden_dim = 300
        self.n_classes = 2
        self.z_dim = 100
        self.ntopics = 50
        self.n_samples = 1
        self.classification_loss_lambda = 1
        self.dynamic_sample = False

    @staticmethod
    def config_as_bool(config_item):
        if config_item == 'yes':
            return True
        elif config_item == True:
            return True
        elif config_item == 'True':
            return True
        else:
            return False

    def _read_config(self, config):
        if 'MODEL' in config:
            self.hidden_dim = int(config['MODEL'].get('hidden_dim', 300))
            self.n_classes = int(config['MODEL'].get('n_classes', 2))
            self.z_dim = int(config['MODEL'].get('z_dim', 100))
            self.ntopics = int(config['MODEL'].get('ntopics', 50))
            self.n_samples = int(config['MODEL'].get('n_samples', 1))
            self.classification_loss_lambda = int(config['MODEL'].get('classification_loss_lambda', 1))
            self.sample_weights = config['MODEL'].get('sample_weights', None)
            self.dynamic_sample = self.config_as_bool(config['MODEL'].get('dynamic_sample', 'no'))

    def reset_parameters(self):
        init.zeros_(self.log_sigma_z1.weight)
        init.zeros_(self.log_sigma_z1.bias)
        init.zeros_(self.log_sigma_z2.weight)
        init.zeros_(self.log_sigma_z2.bias)




    def forward(self, batchItem, mask=None):
        #print(batchItem)
        bert_token_ided = batchItem[0]
        true_y_ids = batchItem[1]
        true_y = self.y2onehot(true_y_ids)
        bow = batchItem[2]

        bert_rep = self.bert_embedding(bert_token_ided, mask)
        cls_att = torch.sum(bert_rep[2][11][:,:,0], 1)
        bert_rep = bert_rep[0]
        bert_rep = bert_rep[:,0]

        mu_z1 = self.mu_z1(bert_rep)
        log_sigma_z1 = self.log_sigma_z1(bert_rep)


        kldz1 = kld(mu_z1, log_sigma_z1)
        rec_loss_z1 = 0
        classifier_loss = 0
        kldz2 = 0
        rec_loss_z2 = 0
        log_y_hat_rec_loss = 0
        class_topic_rec_loss = 0

        n_samples = copy.deepcopy(self.n_samples)
        if self.sample_weights and self.dynamic_sample:
            n_samples = self.get_weighted_num_samples(true_y_ids)
        #n_samples = 10
        #print(n_samples)



        for i in range(n_samples):
            z1 = torch.zeros_like(mu_z1).normal_() * torch.exp(log_sigma_z1) + mu_z1
            z1 = self.h_to_z(z1)
            log_probz_1 = self.x_only_topics(z1)

            if self.mode == 'train' or self.mode == 'update_catopic':
                y_hat_logis = self.xy_classifier(z1)
                y_hat = torch.softmax(y_hat_logis, dim=-1)
                if self.mode == 'train':
                    #true_y_ids = self.y2onehot(true_y)
                    classifier_loss += self.class_criterion(y_hat_logis, true_y_ids)
            else:
                y_hat_logis = self.xy_classifier(mu_z1)
                y_hat = torch.softmax(y_hat_logis, dim=-1)

            rec_loss_z1 = rec_loss_z1 - (log_probz_1 * bow).sum(dim=-1)
            log_prob_class_topic = self.class_topics(y_hat)
            class_topic_rec_loss = class_topic_rec_loss - (log_prob_class_topic*bow).sum(dim=-1)


            y_hat_x = torch.cat((bert_rep, y_hat), dim=-1)
            x_y_hidden = self.x_y_hidden(y_hat_x)
            mu_z2 = self.mu_z2(x_y_hidden)
            log_sigma_z2 = self.log_sigma_z2(x_y_hidden)
            z2 = torch.zeros_like(mu_z2).normal_() * torch.exp(log_sigma_z2) + mu_z2
            log_prob_z2 = self.xy_topics(z2)
            y_hat_rec = self.z2y_classifier(z2)
            log_y_hat_rec = torch.log_softmax(y_hat_rec, dim=-1)

            kldz2 += kld(mu_z2, log_sigma_z2)
            rec_loss_z2 = rec_loss_z2 - (log_prob_z2 * bow).sum(dim=-1)
            if self.mode == 'train':
                log_y_hat_rec_loss = log_y_hat_rec_loss - (log_y_hat_rec*true_y).sum(dim=-1)
            else:
                log_y_hat_rec_loss = log_y_hat_rec_loss - (log_y_hat_rec*y_hat).sum(dim=-1)

        rec_loss_z1 = rec_loss_z1/n_samples
        classifier_loss = classifier_loss/n_samples
        kldz2 = kldz2/n_samples
        rec_loss_z2 = rec_loss_z2/n_samples
        log_y_hat_rec_loss = log_y_hat_rec_loss/n_samples
        class_topic_rec_loss = class_topic_rec_loss/n_samples

        elbo_z1 = kldz1 + rec_loss_z1
        elbo_z2 = kldz2 + rec_loss_z2 + log_y_hat_rec_loss

        total_loss = elbo_z1.sum() + elbo_z2.sum() + class_topic_rec_loss.sum() + classifier_loss*self.banlance_lambda*self.classification_loss_lambda
        
        if self.mode == 'update_catopic':
            total_loss = elbo_z2.sum()

        y = {
            'loss': total_loss,
            'elbo_xy': elbo_z2,
            'rec_loss': rec_loss_z2,
            'kldz2': kldz2,
            'kldz1': kldz1,
            'cls_loss': classifier_loss,
            'class_topic_loss': class_topic_rec_loss,
            'y_hat': y_hat_logis,
            'elbo_x': elbo_z1
        }

        return y


    def get_class_aware_topics_weights(self):
        class_aware_weights = torch.softmax(self.z2y_classifier.layer_output.weight.data, dim=-1)
        #print(topic_weights[class_id].shape)
        return class_aware_weights

    def get_class_regularize_topics_weights(self):
        class_regularize_weights = torch.softmax(self.xy_classifier.layer_output.weight.data, dim=-1)
        return class_regularize_weights

    def get_topics(self):
        return self.xy_topics.get_topics()

    def get_class_topics(self):
        return self.class_topics.get_topics()

    def get_x_only_topics(self):
        return self.x_only_topics.get_topics()


    def get_weighted_num_samples(self, y):
        current_sample_weight = 0
        for each_target_id in y:
            current_sample_weight += self.sample_weights[each_target_id.item()]
        return math.ceil(self.n_samples*current_sample_weight)



    def y2onehot(self, y):
        #device = next(self.parameters()).device
        device = y.get_device()
        #print(device)
        num_class = self.n_classes
        one_hot_y_list = []
        for i in range(len(y)):
            current_one_hot = [0]*num_class
            current_one_hot[y[i].item()] = 1
            one_hot_y_list.append(copy.deepcopy(current_one_hot))
        tensor_one_hot_y = torch.tensor(one_hot_y_list, device=device)
        return tensor_one_hot_y
