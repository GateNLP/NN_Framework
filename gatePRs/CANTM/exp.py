import sys
import argparse
import os
from pathlib import Path
script_path = os.path.abspath(__file__)
dir_path = os.path.dirname(script_path)
parent = Path(dir_path).parent.parent
sys.path.insert(0, str(parent))
from XSModelManager.ModelManager import ModelManager
from XSModelManager.ModelManager_CANTM import ModelManager_CANTM as ModelManager
#from XSNLPReader import TSVReader, JSONReader
import logging
from XSNLPReader.readerPostProcessor import CANTMpostProcessor
from configobj import ConfigObj
logging.basicConfig(level=logging.INFO)


def train_loss_early_stopping(train_ouput, eval_output, best_saved):
        if best_saved == 0:
            best_saved = 999
        score2compare = train_ouput['loss']
        if score2compare < best_saved:
            return score2compare, False
        else:
            return best_saved, True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainInput", help="training file input path")
    parser.add_argument("--testInput", help="testing file input path")
    parser.add_argument("--readerType", help="supported readerType: tsv, json", default='json')
    parser.add_argument("--splitValidation", type=float, help="split data from training for validation")
    parser.add_argument("--nFold", type=int, default=5, help="n fold")
    parser.add_argument("--savePath", help="model save path")
    parser.add_argument("--configFile", help="config files if needed")
    parser.add_argument("--x_fields", help="x fileds", default='text')
    parser.add_argument("--y_field", help="y filed", default='target')
    parser.add_argument("--noUpdateTarget", help="not update targets when reading the training input", default=False, action='store_true')
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--gpu", help="use gpu", default=False, action='store_true')
    parser.add_argument("--num_epoches", type=int, default=100, help="number of training epoches")
    args = parser.parse_args()

    global DataReader
    config = {}
    if args.configFile:
        config = ConfigObj(args.configFile)

    if args.readerType == 'tsv':
        from XSNLPReader import TSVReader as DataReader
    elif args.readerType == 'json':
        from XSNLPReader import JSONReader as DataReader

    x_fields = args.x_fields.split(',')


    readerPostProcessor = CANTMpostProcessor(x_fields=x_fields, y_field=args.y_field)

    updateTarget = not args.noUpdateTarget

    test_dataIter = None

    if args.trainInput:
        train_dataIter = DataReader(args.trainInput, postProcessor=readerPostProcessor, shuffle=True, updateTarget=updateTarget)
        train_dataIter.buildDict()
        train_dataIter.cal_sample_weights()

        dummy_config = {'MODEL':{'n_classes':len(train_dataIter.target_labels), 'vocab_dim':len(train_dataIter.postProcessor.gensim_dict), 'sample_weights':train_dataIter.label_weights_list}}
        dummy_config.update(config)
        print(dummy_config)

        mm = ModelManager(gpu=args.gpu, config=dummy_config)
        mm.genPreBuildModel('CANTM')

        if args.splitValidation:
            train_dataIter, test_dataIter = mm.splitValidation(train_dataIter, val_split=float(args.splitValidation))

        mm.train(train_dataIter, save_path=args.savePath, valDataIter=test_dataIter, earlyStopping=True, patience=10, batch_size=args.batch_size, warm_up=15, earlyStoppingFunction=train_loss_early_stopping, num_epoches=args.num_epoches)
        mm.getTopics(train_dataIter.postProcessor.gensim_dict)
















