import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
#from .BatchIter import BatchIter
import logging
import copy
import math
import random
from pathlib import Path
from sklearn.model_selection import KFold

class CrossValidator:
    def __init__(self):
        self._reset_iter()
        self.n_folds = 0

    def reconstruct_ids(self, each_fold):
        output_ids = [[],[]] #[train_ids, test_ids]
        for sp_id in range(len(each_fold)):
            current_output_ids = output_ids[sp_id]
            current_fold_ids = each_fold[sp_id]
            for doc_id in current_fold_ids:
                current_output_ids.append(self.all_ids[doc_id])
        return output_ids


    def cross_validation(self, dataIter, n_folds=5):
        self._reset_iter()
        self.dataIter = copy.deepcopy(dataIter)
        self.valdataIter = copy.deepcopy(dataIter)
        self.valdataIter.shuffle = False
        self.n_folds = n_folds
        kf = KFold(n_splits=self.n_folds)
        self.all_ids = copy.deepcopy(dataIter.all_ids)
        random.shuffle(self.all_ids)
        self.kfIter = kf.split(self.all_ids)

    def __len__(self):
        return self.n_folds

    def __iter__(self):
        self._reset_iter()
        return self

    def __next__(self):
        if self.current_fold < self.n_folds:
            self.current_fold += 1
            train_ids, test_ids = self.reconstruct_ids(next(self.kfIter))
            #print(train_ids, test_ids)
            self.dataIter.all_ids = copy.deepcopy(train_ids)
            self.valdataIter.all_ids = copy.deepcopy(test_ids)
            return self.dataIter, self.valdataIter

        else:
            self._reset_iter()
            raise StopIteration


    def _reset_iter(self):
        self.current_fold = 0


#def getLogger(name, terminator='\n'):
#    logger = logging.getLogger(name)
#    cHandle = logging.StreamHandler()
#    cHandle.terminator = terminator
#    logger.addHandler(cHandle)
#    return logger


class ModelManager:
    def __init__(self, gpu=False, config={}):
        self.gpu=gpu
        self.config=config
        self.target_labels = []
        self.loggers = {}
        self.trainLoger = self.getLogger('trainLoger')

        if gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.corss_validator = CrossValidator()

    def getLogger(self, name, terminator='\n'):
        if self.loggers.get(name):
            return self.loggers.get(name)
        else:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
            return logger

    def genPreBuildModel(self, model_name=None):
        global BatchIter
        pre_build_name = 'BERT_Simple'
        if model_name:
            pre_build_name = model_name
        elif 'MODEL' in self.config:
            if 'model_name' in self.config['MODEL']:
                pre_build_name = self.config['MODEL']['model_name']

        if pre_build_name == 'BERT_Simple':
            from .models.BERT_Simple import BERT_Simple
            self.net = BERT_Simple(self.config)
            from .BatchIter import BatchIter
            self.batchPostProcessor = self.defaultBatchIterPostProcessor

        if pre_build_name == 'CANTM':
            from .models.CANTM import CANTM
            self.net = CANTM(self.config)
            from .BatchIterCANTM import BatchIterCANTM as BatchIter
            self.batchPostProcessor = self.cantmBatchIterPostProcessor

        if self.gpu:
            self.net.cuda()


    def setOptimiser(self):
        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = nn.CrossEntropyLoss()
        if self.gpu:
            self.criterion.cuda()

    @staticmethod
    def splitValidation(train_dataIter, val_split=0.1):
        ### Need to update better split method to speed up and save memory
        all_ids = copy.deepcopy(train_dataIter.all_ids)
        num_all_ids = len(all_ids)
        split_num = math.floor(num_all_ids*val_split)
        random.shuffle(all_ids)
        val_dataIter = copy.deepcopy(train_dataIter)
        val_dataIter.shuffle = False
        train_dataIter.all_ids = copy.deepcopy(all_ids[split_num:])
        val_dataIter.all_ids = copy.deepcopy(all_ids[:split_num])
        return train_dataIter, val_dataIter

    def train(self, trainDataIter, **kwargs):
        return self.train_default(trainDataIter,**kwargs)

    def train_default(self, trainDataIter, num_epoches=100, valDataIter=None, save_path=None, patience=5, earlyStopping=False, earlyStoppingFunction=None, batch_size=32, batchIterPostProcessor=None, warm_up=1, class_weight=None):
        self.target_labels = trainDataIter.target_labels 
        if not earlyStoppingFunction:
            earlyStoppingFunction = self.default_early_stopping
        else:
            earlyStopping=True
        #trainLoger = getLogger('trainLoger')

        output_dict = {}
        output_dict['accuracy'] = 'no val iter'
        self.setOptimiser()
        best_score = 0
        best_passed = 0

        for epoch in range(num_epoches):
            all_prediction = []
            all_pred_label = []
            all_gold_label = []
            all_loss = []
            infomessage = 'processing epoch: '+str(epoch)
            self.trainLoger.info(infomessage)
            predTrainIter = self.pred(trainDataIter, batch_size=batch_size, batchIterPostProcessor=batchIterPostProcessor, train=True)
            for each_batch_output in predTrainIter:
                loss_value = self.optimiseNet(each_batch_output)
                pred, label_pred, gold_target, cls_att = self.evalItem2CPU(each_batch_output) 
                all_loss.append(loss_value)
                all_prediction.append(pred)
                all_pred_label.append(label_pred)
                all_gold_label.append(gold_target)

            all_prediction = np.concatenate(all_prediction)
            all_pred_label = np.concatenate(all_pred_label)
            all_gold_label = np.concatenate(all_gold_label)
            current_epoch_loss = sum(all_loss)/len(all_loss)
            current_epoch_accuracy = self.get_accuracy(all_pred_label, all_gold_label)
            if valDataIter:
                val_eval_output = self.eval(valDataIter, batch_size=batch_size)
                debugmessage = 'epoch: '+str(epoch)+' finished. loss: '+str(current_epoch_loss)+' train_accuracy: '+str(current_epoch_accuracy)+' val_accuracy: '+str(val_eval_output['accuracy'])+' val_fmeasure: '+str(val_eval_output['f-measure'])
                infomessage = infomessage = 'epoch: '+str(epoch)+' finished. loss: '+str(current_epoch_loss)+' train_accuracy: '+str(current_epoch_accuracy)+' val_accuracy: '+str(val_eval_output['accuracy'])+' val_fmeasure: '+str(val_eval_output['f1-avg']) 
            else:
                infomessage = 'epoch: '+str(epoch)+' finished. loss: '+str(current_epoch_loss)+' train_accuracy: '+str(current_epoch_accuracy)
                debugmessage = ''

                val_eval_output = None
            self.trainLoger.info(infomessage)
            self.trainLoger.debug(debugmessage)

            if earlyStopping and (epoch > warm_up):
                train_output = {'loss': current_epoch_loss, 'accuracy': current_epoch_accuracy}
                best_score, stopping_signal = earlyStoppingFunction(train_output, val_eval_output, best_score)
                if stopping_signal:
                    best_passed += 1
                else:
                    self.trainLoger.info('save checkpoint')
                    self.save_checkpoint(save_path, best_score, epoch)
                    best_passed = 0
                if best_passed > patience:
                    epoch, best_score = self.load_checkpoint(save_path, load_optimiser=True)
                    infomessage = 'early stopping, load epoch: '+str(epoch)+'  with stopping metric score: '+str(best_score)
                    self.trainLoger.info(infomessage)
                    break

        self.save_checkpoint(save_path, best_score, epoch, save_entire=True)

    @staticmethod
    def default_early_stopping(train_ouput, eval_output, best_saved):
        if eval_output:
            score2compare = eval_output['accuracy']
            if score2compare > best_saved:
                return score2compare, False
            else:
                return best_saved, True
        else:
            score2compare = train_ouput['accuracy']
            if score2compare > best_saved:
                return score2compare, False
            else:
                return best_saved, True

    def save_checkpoint_default(self, save_path, best_score, epoch, save_entire=False):
        model_save_path = Path(save_path)
        model_save_path.mkdir(parents=True, exist_ok=True)


        save_dict = {
                'epoch': epoch,
                'best_score': best_score,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'target_labels':self.target_labels,
                }
        check_point_save_path = os.path.join(save_path, 'check_point.pt')
        torch.save(save_dict, check_point_save_path)

        if save_entire:
            entrie_save_path = os.path.join(save_path, 'model.net')
            torch.save(self.net, entrie_save_path)

    def save_checkpoint(self, save_path, best_score, epoch, save_entire=False):
        return self.save_checkpoint_default(save_path, best_score, epoch, save_entire=False)

    def load_model(self, load_path):
        entrie_load_path = os.path.join(load_path, 'model.net')
        self.net = torch.load(entrie_load_path)
        self.load_checkpoint(load_path)

        self.net.to(self.device)




    def load_checkpoint(self, load_path, load_optimiser=False):
        check_point_load_path = os.path.join(load_path, 'check_point.pt')
        checkpoint = torch.load(check_point_load_path)

        self.net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']
        self.target_labels = checkpoint['target_labels']
        
        if load_optimiser:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return epoch, best_score
                
                
    def optimiseNet(self, each_batch_output):
        self.optimizer.zero_grad()
        model_pred = each_batch_output['model_output']['y_hat']
        gold_target = each_batch_output['processed_batch_item'][1]

        loss = self.criterion(model_pred, gold_target)
        loss.backward()
        self.optimizer.step()

        loss_value = float(loss.data.item())

        return loss_value


    def eval(self, dataIter, batch_size=32, batchIterPostProcessor=None):
        predEvalIter = self.pred(dataIter, batch_size=batch_size, batchIterPostProcessor=batchIterPostProcessor, train=False)
        output_dict = {}
        all_prediction = []
        all_pred_label = []
        all_gold_label = []

        for each_batch_output in predEvalIter:
            pred, label_pred, gold_target, cls_att = self.evalItem2CPU(each_batch_output)
            all_prediction.append(pred)
            all_pred_label.append(label_pred)
            all_gold_label.append(gold_target)

        all_prediction = np.concatenate(all_prediction)
        all_pred_label = np.concatenate(all_pred_label)
        all_gold_label = np.concatenate(all_gold_label)
        #print(all_prediction[0])
        #print(all_pred_label)

        num_classes = len(self.target_labels)
        output_dict['f-measure'] = {}
        all_f1 = 0
        for class_id in list(range(num_classes)):
            f_measure_score = self.fMeasure(all_pred_label, all_gold_label, class_id)
            output_dict['f-measure']['class '+self.target_labels[class_id]] = {}
            output_dict['f-measure']['class '+self.target_labels[class_id]]['precision'] = f_measure_score[0]
            output_dict['f-measure']['class '+self.target_labels[class_id]]['recall'] = f_measure_score[1]
            output_dict['f-measure']['class '+self.target_labels[class_id]]['f1'] = f_measure_score[2]
            all_f1 += f_measure_score[2]

        output_dict['f1-avg'] = all_f1/num_classes

        accuracy = self.get_accuracy(all_pred_label, all_gold_label)
        output_dict['accuracy'] = accuracy
        return output_dict


    @staticmethod
    def get_accuracy(all_pred_label_npy, all_gold_label_npy):
        num_correct = (all_pred_label_npy == all_gold_label_npy).sum()
        accuracy = num_correct / len(all_pred_label_npy)
        return accuracy

    @staticmethod
    def evalItem2CPU(each_batch_output):
        pred = each_batch_output['model_output']['y_hat']
        softmax_pred = F.softmax(pred, dim=-1)
        label_pred = torch.max(softmax_pred, -1)[1]
        cls_att = None
        if 'cls_att' in each_batch_output['model_output']:
            cls_att = each_batch_output['model_output']['cls_att']
            cls_att = cls_att.to('cpu').detach().numpy()



        gold_target = each_batch_output['processed_batch_item'][1]
        pred = pred.to('cpu').detach().numpy()
        gold_target = gold_target.to('cpu').detach().numpy()
        label_pred = label_pred.to('cpu').detach().numpy()
        return pred, label_pred, gold_target, cls_att


    def apply(self, dataIter, batch_size=32, batchIterPostProcessor=None):
        applyIter = self.pred(dataIter, batch_size=batch_size, batchIterPostProcessor=batchIterPostProcessor, train=False)
        output_dict = {}
        all_prediction = []
        all_pred_label = []
        all_gold_label = []
        all_cls_att = []

        for each_batch_output in applyIter:
            pred, label_pred, gold_target, cls_att = self.evalItem2CPU(each_batch_output)
            all_prediction.append(pred)
            all_pred_label.append(label_pred)
            all_cls_att.append(cls_att)

        all_prediction = np.concatenate(all_prediction).tolist()
        all_pred_label = np.concatenate(all_pred_label).tolist()
        all_pred_label_string = self.labelID2labelString(all_pred_label)
        if cls_att is not None:
            all_cls_att = np.concatenate(all_cls_att).tolist()

        output_dict['all_prediction'] = all_prediction
        output_dict['all_pred_label'] = all_pred_label
        output_dict['all_pred_label_string'] = all_pred_label_string
        output_dict['all_cls_att'] = all_cls_att
        return output_dict

    def labelID2labelString(self, all_pred_label):
        all_pred_label_string = []
        for each_label_id in all_pred_label:
            all_pred_label_string.append(self.target_labels[each_label_id])
        return all_pred_label_string




    def pred(self, dataIter, batch_size=32, batchIterPostProcessor=None, train=False):
        dataIter._reset_iter()
        if train:
            self.net.train()
            filling_last_batch = True
        else:
            self.net.eval()
            filling_last_batch = False

        if not batchIterPostProcessor:
            batchIterPostProcessor = self.batchPostProcessor


        batchIter = BatchIter(dataIter, batch_size=batch_size, filling_last_batch=filling_last_batch)
        #predLoger = getLogger('predLoger')
        #print(logging.root.level)

        for batch_item in batchIter:
            #print(batch_item[1])
            batch_output = {}
            infomessage = 'processing batch '+str(batchIter.current_batch_idx)+'/'+str(len(batchIter))
            #predLoger.info(infomessage)
            if logging.root.level <= 20:
                print(infomessage, end='\r')
                #print(infomessage)
            processed_batch_item = batchIterPostProcessor(batch_item, device=self.device)
            #print(processed_batch_item)
            if train:
                model_output = self.net(processed_batch_item)
            else:
                with torch.no_grad():
                    model_output = self.net(processed_batch_item)

            batch_output['processed_batch_item'] = processed_batch_item
            batch_output['model_output'] = model_output
            yield batch_output

    @staticmethod
    def defaultBatchIterPostProcessor(batch_item, device=torch.device('cpu')):
        #print(device)
        text_tensor = torch.tensor(batch_item[0], device=device)
        target_tensor = torch.tensor(batch_item[1], device=device)
        return [text_tensor, target_tensor]

    @staticmethod
    def cantmBatchIterPostProcessor(batch_item, device=torch.device('cpu')):
        bert_input_tensor = torch.tensor(batch_item['bert_ided'], device=device)
        count_matrix = torch.tensor(batch_item['count_matrix'], device=device)
        target_labels = torch.tensor(batch_item['target_label'], device=device)
        #target_labels = self.y2onehot(target_labels, device)
        return [bert_input_tensor, target_labels, count_matrix]



    def fMeasure(self, all_prediction, true_label, class_id, ignoreid=None):
        #print(class_id)
        mask = [class_id] * len(all_prediction)
        mask_arrary = np.array(mask)
        pred_mask = np.argwhere(all_prediction==class_id)
        #print(pred_mask)
        true_mask = np.argwhere(true_label==class_id)
        #print(1111)
        #print(true_mask)
        #all_prediction = all_prediction.tolist()
        #true_label = true_label.tolist()
        #print(len(true_mask))

        total_pred = 0
        total_true = 0
        #print(total_pred, total_true)

        #print(all_prediction)
        #print(true_label)


        pc = 0
        for i in pred_mask:
            if all_prediction[i[0]] == true_label[i[0]]:
                pc+=1
            if true_label[i[0]] != ignoreid:
                total_pred += 1

        rc = 0
        for i in true_mask:
            if all_prediction[i[0]] == true_label[i[0]]:
                rc+=1
            if true_label[i[0]] != ignoreid:
                total_true += 1

        if total_pred == 0:
            precision = 0
        else:
            precision = float(pc)/total_pred
        if total_true == 0:
            recall = 0
        else:
            recall = float(rc)/total_true
        if (precision+recall)==0:
            f_measure = 0
        else:
            f_measure = 2*((precision*recall)/(precision+recall))
        #print(total_true)
        return precision, recall, f_measure, total_pred, total_true, pc, rc
