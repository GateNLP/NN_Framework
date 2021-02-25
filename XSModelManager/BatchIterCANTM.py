import math
import random


class BatchIterCANTM:
    def __init__(self, dataIter, batch_size=32, filling_last_batch=False, postProcessor=None):
        self.dataIter = dataIter
        self.batch_size = batch_size
        self.num_batches = self._get_num_batches()
        self.filling_last_batch = filling_last_batch
        self.postProcessor = postProcessor
        self.fillter = []
        self._reset_iter()

    def _get_num_batches(self):
        num_batches = math.ceil(len(self.dataIter)/self.batch_size)
        return num_batches

    def _reset_iter(self):
        self.current_batch_idx = 0

    def __iter__(self):
        self._reset_iter()
        return self

    def __next__(self):
        if self.current_batch_idx < self.num_batches:
            current_batch = self._readNextBatch()
            self.current_batch_idx += 1
            if self.postProcessor:
                return self.postProcessor(current_batch)
            else:
                return current_batch

        else:
            self._reset_iter()
            raise StopIteration
    def __len__(self):
        return self.num_batches

    def _readNextBatch(self):
        i = 0
        batch_dict = {}
        while i < self.batch_size:
            try:
                each_item_dict = next(self.dataIter)
                if self.filling_last_batch:
                    self._update_fillter(each_item_dict)
                for reader_item_key in each_item_dict:
                    if reader_item_key in batch_dict:
                        batch_dict[reader_item_key].append(each_item_dict[reader_item_key])
                    else:
                        batch_dict[reader_item_key] = [each_item_dict[reader_item_key]]
                i+=1
            except StopIteration:
                if self.filling_last_batch:
                    batch_dict = self._filling_last_batch(batch_dict, i)
                i = self.batch_size
        return batch_dict


    def _filling_last_batch(self, batch_dict, num_current_batch):
        num_filling = self.batch_size - num_current_batch
        random.shuffle(self.fillter)
        for filler_id in range(num_filling):
            each_item_dict = self.fillter[filler_id]
            for reader_item_key in each_item_dict:
                if reader_item_key in batch_dict:
                    batch_dict[reader_item_key].append(each_item_dict[reader_item_key])
                else:
                    batch_dict[reader_item_key] = [each_item_dict[reader_item_key]]
        return batch_dict

    def _update_fillter(self, each_item_dict):
        r = random.random()
        if len(self.fillter) < self.batch_size:
            self.fillter.append(each_item_dict)
        elif r>0.9:
            self.fillter.pop(0)
            self.fillter.append(each_item_dict)

