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

class Dataloader:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load(self):
        """
        return data = 
        {
            "train": {
                file_name: {
                    "file_path" : file_path, 
                    "file_content" : file_content
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
        train_name_filepath = {}
        # 遍历文件夹中的每个项
        for filename in os.listdir(self.data_path["train"]):
            # 构造完整的文件路径
            file_path = os.path.join(self.data_path["train"], filename)
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
                    stop_words = set(stopwords.words('english'))
                    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
                    
                    # 词干提取
                    # stemmer = PorterStemmer()
                    # stemmed_tokens = [stemmer.stem(token) for token in tokens]

                train_name_filepath[filename] = {"file_path" : file_path, "file_content" : filtered_tokens}
        
        test_name_filepath = {}
        # 遍历文件夹中的每个项
        for filename in os.listdir(self.data_path["test"]):
            # 构造完整的文件路径
            file_path = os.path.join(self.data_path["test"], filename)
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
                    stop_words = set(stopwords.words('english'))
                    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
                    
                    # 词干提取
                    # stemmer = PorterStemmer()
                    # stemmed_tokens = [stemmer.stem(token) for token in tokens]
                    
                test_name_filepath[filename] = {"file_path" : file_path, "file_content" : filtered_tokens}
                
        data = {
            'train': train_name_filepath,
            'test': test_name_filepath
        }
        
        return data

# 遍历文件夹中的文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):  # 确保它是txt文件
#         file_path = os.path.join(folder_path, filename)  # 获取文件的完整路径
#         with open(file_path, 'r', encoding='utf-8') as file:  # 打开文件进行读取
#             content = file.read()  # 读取文件内容
#             file_contents.append(content)  # 将内容添加到列表中

# 现在file_contents列表包含了每个txt文件的内容作为字符串
    