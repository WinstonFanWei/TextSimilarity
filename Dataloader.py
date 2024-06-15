import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import os

from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import spacy

from tqdm import tqdm

class Dataloader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.stop_words = set(stopwords.words('english'))
        self.sentence_split_method = spacy.load('en_core_web_sm')
        
    def load(self):
        """
        return data = 
        {
            "train": {
                file_name: {
                    "file_path" : file_path, 
                    "file_sentences": file_sentences, [list] (sentence_num, sentence_length)
                    "file_content" : file_content
                }
            }
            "test": {
                file_name: {
                    "file_path" : file_path, 
                    "file_sentences": file_sentences, [list] (sentence_num, sentence_length)
                    "file_content" : file_content
                }
            }
        }
        """
        
        print("[Loading Data ...]")
        data = {
            'train': self.preprocess("train"),
            'test': self.preprocess("test")
        }
        print("[Loading Data ... Finished]")
        
        return data
    
    def sentence_preprocess(self, text):
        # 预处理文本
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        return tokens

    def preprocess(self, mode):
        """
        return name_filepath_list = {
            file_name: {
                "file_path": file_path, 
                "file_sentences": file_sentences, # 文件的句子表示
                "file_content": file_content # 文件的单词表示
            }
            ......
        }
        """
        name_filepath_list = {}
        # 遍历文件夹中的每个项
        for filename in tqdm(os.listdir(self.data_path[mode]), desc="Loading " + mode + " Data" ):
            # 构造完整的文件路径
            file_path = os.path.join(self.data_path[mode], filename)
            # 检查这个文件是否是文件而不是文件夹
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 读取文件内容到字符串
                    text = file.read()
                    
                    # 确保已经下载了所需的 NLTK 数据包
                    # nltk.download('punkt')
                    # nltk.download('stopwords')

                    # 分词处理
                    tokens = word_tokenize(text)

                    # 停用词处理
                    filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words and word.isalpha()]
                    
                    # 词干提取
                    # stemmer = PorterStemmer()
                    # stemmed_tokens = [stemmer.stem(token) for token in tokens]
                    
                    # 分句子
                    doc = self.sentence_split_method(text)
                    sentences = [sent.text for sent in doc.sents]
                    
                    # 对每个句子进行预处理
                    processed_sentences = [self.sentence_preprocess(sentence) for sentence in sentences]

                name_filepath_list[filename] = {"file_path": file_path, "file_sentences": processed_sentences, "file_content": filtered_tokens}
                
        return name_filepath_list
        