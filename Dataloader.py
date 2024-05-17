import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import os

class Dataloader:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load(self):
        ls_train = []
        # 遍历文件夹中的每个项
        for filename in os.listdir(self.data_path["train"]):
            # 构造完整的文件路径
            file_path = os.path.join(self.data_path["train"], filename)
            # 检查这个文件是否是文件而不是文件夹
            if os.path.isfile(file_path):
                ls_train.append(file_path)
        
        ls_test = []
        # 遍历文件夹中的每个项
        for filename in os.listdir(self.data_path["test"]):
            # 构造完整的文件路径
            file_path = os.path.join(self.data_path["test"], filename)
            # 检查这个文件是否是文件而不是文件夹
            if os.path.isfile(file_path):
                ls_test.append(file_path)
                
        files = {
            'train': ls_train,
            'test': ls_test
        }
        
        return files

# 遍历文件夹中的文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):  # 确保它是txt文件
#         file_path = os.path.join(folder_path, filename)  # 获取文件的完整路径
#         with open(file_path, 'r', encoding='utf-8') as file:  # 打开文件进行读取
#             content = file.read()  # 读取文件内容
#             file_contents.append(content)  # 将内容添加到列表中

# 现在file_contents列表包含了每个txt文件的内容作为字符串
    