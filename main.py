import torch
import torch.nn as nn
import torch.optim as optim

import logging

import numpy as np
import pandas as pd
import time

from tqdm import tqdm
import os

import Utils
from models.LLAModel import LLAModel
from Dataloader import Dataloader
from CompareFiles import CompareFiles
from distancecompare.DistanceCompare import DistanceCompare

from gensim import corpora, models
from gensim.test.utils import datapath
from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
    
def train_epoch(model, dataloader, optimizer, criterion):
    pass
    
def train(model, filepath, optimizer, scheduler, paras):
    pass
    
def main(data, paras):
    
    train_data = data['train']

    '''
    LDA模型训练
    '''
    # 构建Dictionary
    text_ls = []
    for key, value in train_data.items():
        text_ls.append(value["file_content"])
    
    dictionary = corpora.Dictionary(text_ls)
    
    # 转换文档为词袋模型
    corpus = [dictionary.doc2bow(text) for text in text_ls]
    
    random_seed = 42

    # 构建 LDA 模型
    if paras["LDA_isload"] == False:
        print("[LDA模型训练中]")
        start_time = time.time()
        lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=20, passes=40, random_state=random_seed)
        end_time = time.time()
        print("[LDA模型训练结束, 用时: " + str(round(end_time - start_time, 2)) + "s]")
        lda_model.save(paras["LDA_load_path"])
        
    else:
        # 加载模型
        print("[LDA模型 loading ... Finished]")
        lda_model = models.LdaModel.load(paras["LDA_load_path"])
    
    # 训练word2Vec模型
    print("[Word2Vec模型训练中]")
    start_time = time.time()
    word2vec_model = Word2Vec(text_ls, vector_size=50, window=10, min_count=1, workers=4, seed=random_seed)
    end_time = time.time()
    print("[Word2Vec模型训练结束, 用时: " + str(round(end_time - start_time, 2)) + "s]")
    
    """
        update data = 
        {
            "train": {
                file_name: {
                    "file_path" : file_path, 
                    "file_content" : file_content, 
                    "file_token_topic_list" : file_token_topic_list [list](file_lenth, topic_numbers)
                    "file_token_topic_list_max" : file_token_topic_list_max [list](file_lenth)
                }
            }
            "test": {
                file_name: {
                    "file_path" : file_path, 
                    "file_content" : file_content
                }
            }
        }
    """
        
    for key, value in train_data.items():
        file_copus_sequence = []
        for token in value["file_content"]:
            file_copus_sequence.append(dictionary.doc2bow([token])[0])
            
        file_token_topic = lda_model.get_document_topics(file_copus_sequence, minimum_probability=0, minimum_phi_value=0, per_word_topics=True)
        file_token_topic_list = file_token_topic[2]
        file_token_topic_list_max = file_token_topic[1]
            
        # 对 file_token_topic_list 进行简单化
        value["file_token_topic_list"] = [ [topic[1] for topic in token[1]] for token in file_token_topic_list ]
        value["file_token_topic_list_max"] = [ word[1][0] for word in file_token_topic_list_max ]

    # print(train_data["test.txt"])
    
    # to do: 嵌入比较
    train_compare_path = os.path.join(paras["file_path"], "validation\\validation\\similarity_scores.csv")
    test_compare_path = os.path.join(paras["file_path"], "validation\\validation\\similarity_scores.csv")
    
    if paras["Debug"] == False:
        comparefiles = CompareFiles(lda_model, word2vec_model, train_data, train_compare_path, paras)
        compare_result = comparefiles.compare()
        compare_result.to_csv('train_output.csv', index=False)
    else:
        compare_result = pd.read_csv('train_output.csv')
    
    # 指标展示
    Utils.compute_rmse(compare_result, paras)
    Utils.compute_correlation(compare_result, paras)
    Utils.compute_f1(compare_result, paras)
    
    
if __name__ == '__main__':
    print("-----------------------------------------------------------------------------------------------------------------")
    """ Main function. """
    
    # Parameters
    paras = {
        "file_path": "C:\\Users\\Winston\\Desktop\\document-similarity-main\\document-similarity-main",
        "topic_distance_matrix_iscomputed": True,
        "MySimilarityCompute": True,
        "LDA_isload": True,
        "LDA_load_path": "C:\\Users\\Winston\\Desktop\\Repository\\TextSimilarity\\modelsave\\lda_model.model",
        "Debug": False,
        "only_compute_this_similarity": 'Similarity_doc_topic',
    #     "vocab_size": 10000,
    #     "embedding_dim": 300,
    #     "hidden_size": 128,
    #     "num_topics": 50,
    #     "batch_size": 64,
    #     "lr": 0.001,
    #     "epochs": 25,
    #     "device": "cpu" # cuda
    }

    # Data
    # train_data = [torch.randint(0, paras["vocab_size"], (paras["batch_size"],), dtype=torch.long) for _ in range(10)]
    
    # Create the LLA model to be model
    # model = LLAModel(paras["vocab_size"], paras["embedding_dim"], paras["hidden_size"], paras["num_topics"])
    
    # model.to(torch.device(paras["device"]))
    
    """ optimizer and scheduler """
    # optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
    #                        paras["lr"], betas=(0.9, 0.999), eps=1e-05)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ number of parameters """
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('[Info] Number of parameters: {}'.format(num_params))

    """ prepare dataloader """
    data_path = {
        'train': os.path.join(paras["file_path"], "validation\\validation\\documents"),
        'test': os.path.join(paras["file_path"], "test\\test\\documents")
    }
    dataloader = Dataloader(data_path)
    data = dataloader.load()
    
    """ train the model """
    # train(model, filepath, optimizer, scheduler, paras)

    # Run the EM algorithm
    # model.em_algorithm(train_data, num_iterations=10, batch_size=batch_size, lr=lr)

    main(data, paras)
