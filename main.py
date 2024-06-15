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
import spacy

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
                    "file_sentences": file_sentences, [list] (sentence_num, sentence_length)
                    "file_token_topic_list" : file_token_topic_list [list](file_lenth, topic_numbers)
                    "file_token_topic_list_max" : file_token_topic_list_max [list](file_lenth)
                    "file_sentence_token_topic_list_max": file_sentence_token_topic_list [list](sentence_number)
                }
            }
            "test": {
                file_name: {
                    "file_path" : file_path, 
                    "file_content" : file_content, 
                    "file_sentences": file_sentences, [list] (sentence_num, sentence_length)
                }
            }
        }
    """
    
    # 得到文档每个token的topic概率
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
    
    # 得到文档每个sentence的topic概率 - 两种方式：词topic分布的平均 或者 从get_document_topics中获得句子的topic，现在采用第二种
    
    for key, value in train_data.items():
        file_sentence_topic_list = []
        for sentence in value["file_sentences"]:
            file_sentence_topic = lda_model.get_document_topics(dictionary.doc2bow(sentence), minimum_probability=0, minimum_phi_value=0)
            
            # 对 file_sentence_token_topic_list 进行简单化
            max_index = max(file_sentence_topic, key=lambda x: x[1])[0]
                
            file_sentence_topic_list.append(max_index)
            value["file_sentence_token_topic_list_max"] = file_sentence_topic_list
        
    # print(train_data["test.txt"])
    
    # 计算文件相似度
    train_compare_path = os.path.join(paras["file_path"], "validation\\validation\\similarity_scores.csv")
    test_compare_path = os.path.join(paras["file_path"], "validation\\validation\\similarity_scores.csv")
    
    if paras["Debug"] == False:
        comparefiles = CompareFiles(lda_model, word2vec_model, train_data, train_compare_path, paras)
        compare_result = comparefiles.compare()
        compare_result.to_csv('output\\train_output.csv', index=False)
    else:
        compare_result = pd.read_csv('output\\train_output.csv')
    
    # 指标展示
    Utils.compute_rmse(compare_result, paras)
    Utils.compute_correlation(compare_result, paras)
    Utils.compute_f1(compare_result, paras)
    
    
if __name__ == '__main__':
    print("-----------------------------------------------------------------------------------------------------------------")
    """ Main function. """
    
    # Parameters
    paras = {
        # 文件路径
        "file_path": "C:\\Users\\Winston\\Desktop\\document-similarity-main\\document-similarity-main", # 数据集路径
        "LDA_load_path": "C:\\Users\\Winston\\Desktop\\Repository\\TextSimilarity\\modelsave\\lda_model.model", # LDA模型保存路径
        
        # 开关
        "LDA_isload": True, # LDA模型是否已经保存下来，保存了就不用再次训练
        "Debug": False, # Debug模式下不进行相似度的计算 即CompareFiles()
        "topic_distance_matrix_iscomputed": True, # topic距离矩阵是否已经计算完成
        "MySimilarityCompute": True, # 是否计算我们的主要相似度度量方式
        "open_theta": False # 暂时关闭theta的逻辑
    }

    """ prepare dataloader """
    data_path = {
        'train': os.path.join(paras["file_path"], "validation\\validation\\documents"),
        'test': os.path.join(paras["file_path"], "test\\test\\documents")
    }
    dataloader = Dataloader(data_path)
    data = dataloader.load()

    main(data, paras)
