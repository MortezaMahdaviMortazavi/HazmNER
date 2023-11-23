import torch
import numpy as np
import os
import config

from dataset import NERDataset
from model import Bert
from train import optimization
from evaluate import evaluate
from preprocessing import prepare_conll_data_format

def read_data(path):
    pass


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.GPU_ID}")
    else:
        device = torch.device('cpu')

    train_sents , train_labels = prepare_conll_data_format(config.ARMAN_TRAIN_FOLD1,sep=' ',verbose=False)
    test_sents , test_labels = prepare_conll_data_format(config.ARMAN_TEST_FOLD1,sep=' ',verbose=False)

    trainset = NERDataset(train_sents,train_labels)
    testset = NERDataset(test_sents,test_labels)
    
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=config.TRAIN_BATCH_SIZE,shuffle=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size=config.VAL_BATCH_SIZE,shuffle=False)

    model = Bert().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = None

    optimization(model,trainloader,criterion,optimizer,device,scheduler=scheduler)
# def optimization(model,dataloader,criterion,optimizer,device,scheduler=None):
