import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import time

from tqdm import tqdm
import os

import Utils
from models.LLAModel import LLAModel
from Dataloader import Dataloader
    
def train_epoch(model, dataloader, optimizer, criterion):
    pass
    
def train(model, files, optimizer, scheduler, paras):
    pass
    
    
if __name__ == '__main__':
    """ Main function. """
    
    # Parameters
    paras = {
        "vocab_size": 10000,  
        "embedding_dim": 300,
        "hidden_size": 128,
        "num_topics": 50,
        "batch_size": 64,
        "lr": 0.001,
        "epochs": 25,
        "device": "cpu" # cuda
    }

    # Data
    train_data = [torch.randint(0, paras["vocab_size"], (paras["batch_size"],), dtype=torch.long) for _ in range(10)]
    
    # Create the LLA model to be model
    model = LLAModel(paras["vocab_size"], paras["embedding_dim"], paras["hidden_size"], paras["num_topics"])
    
    model.to(torch.device(paras["device"]))
    
    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           paras["lr"], betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ prepare dataloader """
    data_path = {
        'train': 'C:\\Users\\Winston\\Desktop\\document-similarity-main\\document-similarity-main\\validation\\validation\\documents',
        'test': 'C:\\Users\\Winston\\Desktop\\document-similarity-main\\document-similarity-main\\test\\test\\documents'
    }
    dataloader = Dataloader(data_path)
    files = dataloader.load()
    
    """ train the model """
    train(model, files, optimizer, scheduler, paras)

    # Run the EM algorithm
    # model.em_algorithm(train_data, num_iterations=10, batch_size=batch_size, lr=lr)
    
    print(files)
    
    