import torch
import numpy as np
import os
import config
import logging

from dataset import NERDataset
from model import Bert
from train import optimization
from evaluate import evaluate
from preprocessing import prepare_conll_data_format


if __name__ == "__main__":
    logging.basicConfig(filename=config.LOGFILE, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.GPU_ID}")
        logging.info(f"Using CUDA device: {device}")

    else:
        device = torch.device('cpu')
        logging.warning("CUDA is not available. Using CPU.")


    train_sents , train_labels = prepare_conll_data_format(config.ARMAN_TRAIN_FOLD1,sep=' ',verbose=False)
    test_sents , test_labels = prepare_conll_data_format(config.ARMAN_TEST_FOLD1,sep=' ',verbose=False)

    train_sents.extend(prepare_conll_data_format(config.ARMAN_TRAIN_FOLD2,sep=' ',verbose=False)[0])
    train_labels.extend(prepare_conll_data_format(config.ARMAN_TRAIN_FOLD2,sep=' ',verbose=False)[1])
    train_sents.extend(prepare_conll_data_format(config.ARMAN_TRAIN_FOLD3,sep=' ',verbose=False)[0])
    train_labels.extend(prepare_conll_data_format(config.ARMAN_TRAIN_FOLD3,sep=' ',verbose=False)[1])

    test_sents.extend(prepare_conll_data_format(config.ARMAN_TEST_FOLD2,sep=' ',verbose=False)[0])
    test_labels.extend(prepare_conll_data_format(config.ARMAN_TEST_FOLD2,sep=' ',verbose=False)[1])
    test_sents.extend(prepare_conll_data_format(config.ARMAN_TEST_FOLD3,sep=' ',verbose=False)[0])
    test_labels.extend(prepare_conll_data_format(config.ARMAN_TEST_FOLD3,sep=' ',verbose=False)[1])


    trainset = NERDataset(train_sents,train_labels)
    testset = NERDataset(test_sents,test_labels)
    logging.info("train dataset and test dataset created successfully")

    
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=config.TRAIN_BATCH_SIZE,shuffle=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size=config.VAL_BATCH_SIZE,shuffle=False)
    logging.info(f"""
                 train dataloader and test dataloader created with train batch_size of {config.TRAIN_BATCH_SIZE}
                 and test batch_size of {config.VAL_BATCH_SIZE}
    """)

    model = Bert().to(device)
    logging.info(f"Model initialized: {model}")

    optimizer = torch.optim.AdamW(model.parameters(),lr=config.LEARNING_RATE)
    logging.info(f"Optimizer initialized: AdamW with learning rate {config.LEARNING_RATE}")


    criterion = torch.nn.CrossEntropyLoss()
    logging.info(f"Loss criterion initialized: CrossEntropyLoss")

    scheduler = None

    optimization(model,trainloader,criterion,optimizer,device,scheduler=scheduler)
    eval_loss , eval_precision , eval_recall , eval_f1 , eval_accuracy = evaluate(model,testloader,criterion,device)
