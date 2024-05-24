import numpy as np
import pandas as pd
import ot
from tqdm import tqdm
from tslearn.metrics import dtw_path_from_metric
import torch

import logging

class DistanceCompare:
    def __init__(self, lda_model, word2vec_model, data, topic_distance_matrix_iscomputed):
        self.lda_model = lda_model
        self.word2vec_model = word2vec_model
        self.data = data
        self.theta = torch.tensor(20.0, dtype=torch.float32, requires_grad=True)
        if topic_distance_matrix_iscomputed == False:
            self.topic_distance_matrix = self.generate_topic_distance_matrix()
        else:
            self.topic_distance_matrix = self.generate_topic_distance_matrix_from_csv()
        
    def word_distance(self, word1, word2):
        """
        Word distance through word2vec model.
        Input: word2vec model
        return: Word distance
        """
        similarity = self.word2vec_model.wv.similarity(word1, word2)
        distance = 1 - similarity
        return distance
    
    def generate_freq_word_list(self):
        # 过滤掉出现频率较低的词 产生新字典
        new_dic = {}
        for word_id, freq in self.lda_model.id2word.dfs.items():
            if freq > 4:
                new_dic[word_id] = self.lda_model.id2word[word_id]

        return new_dic
    def generate_topic_distance_matrix(self):
        
        # 产生新高频字典
        new_dic = self.generate_freq_word_list()
        
        # 对字典进行排序，以key为索引 升序
        sort_dic_to_list = sorted(new_dic.items(), key=lambda x: x[0])
        
        # 过滤之后剩下的word_id
        id_list = [word[0] for word in sort_dic_to_list]
        
        # 计算代价矩阵
        lenth = len(sort_dic_to_list)
        M = np.zeros((lenth, lenth))
        # print("M shape: ", len(M), len(M[0]))
        
        for row_id, row in enumerate(tqdm(sort_dic_to_list, desc='[Generating M matrix by new dic]')):
            for column_id, column in enumerate(sort_dic_to_list):
                M[row_id][column_id] = round(self.word_distance(row[1], column[1]), 4)
                
        print("[Generating M matrix Finished]")
        
        pd.DataFrame(M).to_csv('output\\M.csv', index=False)
        
        topic_num = len(self.lda_model.get_topics())
        topic_distance_matrix = np.zeros((topic_num, topic_num))
        
        for row in tqdm(range(topic_num), desc='[Generating topic distance matrix]'):
            u_distribution = self.lda_model.get_topics()[row]
            u_distribution_freq = []
            for word_id, word_pro in enumerate(u_distribution):
                if word_id in id_list:
                    u_distribution_freq.append(word_pro)
                    
            u_distribution_freq /= np.sum(u_distribution_freq)
            
            for column in range(topic_num):
                
                v_distribution = self.lda_model.get_topics()[column]
                v_distribution_freq = []
                for word_id, word_pro in enumerate(v_distribution):
                    if word_id in id_list:
                        v_distribution_freq.append(word_pro)
                        
                v_distribution_freq /= np.sum(v_distribution_freq)

                # 计算Wasserstein距离
                distance = ot.emd2(u_distribution_freq, v_distribution_freq, M, numItermax=1000000)

                topic_distance_matrix[row][column] = round(distance, 4)
                
        pd.DataFrame(topic_distance_matrix).to_csv('output\\topic_distance_matrix.csv', index=False)
        print("[Generating topic distance matrix Finished]")
        
        return topic_distance_matrix
        
    def generate_topic_distance_matrix_from_csv(self):
        print("[From csv \"output\\topic_distance_matrix\" to get topic distance matrix ... Finished]")
        return pd.read_csv('output\\topic_distance_matrix.csv').to_numpy()
        
    def compare(self, file1, file2, mode):
        dtw = self.DTW(file1, file2, mode)
        return torch.exp(- self.theta * dtw)

    def DTW(self, file1, file2, mode="word"):
        def custom_distance(i, j):
            return self.topic_distance_matrix[int(i[0]), int(j[0])]
        
        if mode == "word":
            x = [ [topic] for topic in self.data[str(file1) + ".txt"]["file_token_topic_list_max"] ]
            y = [ [topic] for topic in self.data[str(file2) + ".txt"]["file_token_topic_list_max"] ]
            
        if mode == "sentence":
            x = [ [topic] for topic in self.data[str(file1) + ".txt"]["file_sentence_token_topic_list_max"] ]
            y = [ [topic] for topic in self.data[str(file2) + ".txt"]["file_sentence_token_topic_list_max"] ]
        
        path, dtw_distance = dtw_path_from_metric(x, y, metric=custom_distance)
        
        DTW_standard = dtw_distance / len(path)
        
        with open("output\\DTW_result_" + mode + ".txt", 'a') as file:
            file.write(str(file1) + ".txt V.S. " + str(file2) + ".txt: \n[dtw_distance]" + str(round(dtw_distance, 4)) + "\n[DTW_standard]" + str(round(DTW_standard,4)) + "\n\n[Path]\n\n " + str(path) + 
                       "\n------------------------------------------------------------------------------------------------------\n\n\n")
        
        return DTW_standard
        