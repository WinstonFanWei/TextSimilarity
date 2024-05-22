import numpy as np
import pandas as pd
import ot
from tqdm import tqdm
from tslearn.metrics import dtw_path_from_metric

class DistanceCompare:
    def __init__(self, lda_model, word2vec_model, data):
        self.lda_model = lda_model
        self.word2vec_model = word2vec_model
        self.data = data
        self.topic_distance_matrix = self.generate_topic_distance_matrix()
        self.doc_distance_matrix = None
        
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
        print("M shape: ", len(M), len(M[0]))
        
        for row_id, row in enumerate(tqdm(sort_dic_to_list, desc='[Generating M matrix by new dic]')):
            for column_id, column in enumerate(sort_dic_to_list):
                M[row_id][column_id] = self.word_distance(row[1], column[1])
                
        print("[Generating M matrix Finished]")
        
        pd.DataFrame(M).to_csv('M.csv', index=False)
        
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

                topic_distance_matrix[row][column] = distance
                
        print("[Generating topic distance matrix Finished]")
        
        return topic_distance_matrix
        
    def compare(self, file1, file2):
        dtw = self.DTW(file1, file2)
        return 1 / (1 + dtw)

    def DTW(self, file1, file2):
        def custom_distance(i, j):
            return self.topic_distance_matrix[int(i[0]), int(j[0])]
        
        x = [ [topic] for topic in self.data[str(file1) + ".txt"]["file_token_topic_list_max"] ]
        y = [ [topic] for topic in self.data[str(file2) + ".txt"]["file_token_topic_list_max"] ]
        
        path, dtw_distance = dtw_path_from_metric(x, y, metric=custom_distance)
        return dtw_distance
        