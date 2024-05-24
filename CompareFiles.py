import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from distancecompare.DistanceCompare import DistanceCompare
from tqdm import tqdm
from datetime import datetime
import torch
import torch.optim as optim

import logging

class CompareFiles:
    def __init__(self, lda_model, word2vec_model, data, compare_path, paras):
        self.paras = paras
        self.MySimilarityCompute = paras["MySimilarityCompute"]
        self.lda_model = lda_model
        self.word2vec_model = word2vec_model
        self.data = data
        self.compare_path = compare_path
        self.result = None
        self.DC = DistanceCompare(self.lda_model, self.word2vec_model, self.data, paras["topic_distance_matrix_iscomputed"])
    def compare(self):
        with open("output\\DTW_result_word.txt", 'w') as file:
            file.write("DTW_result, Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            
        with open("output\\DTW_result_sentence.txt", 'w') as file:
            file.write("DTW_result, Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            
        df_compare = self.read_file()
        
        if self.MySimilarityCompute == True:
            if self.paras["open_theta"] == True:
                print(["Training the word theta ... "])
                self.train_theta(df_compare, "word")
                print(["Training the word theta ... Finished"])
            for index, row in tqdm(df_compare.iterrows(), desc='[MySimilarity compute]', total=len(df_compare)):
                df_compare.loc[index, 'mySimilarity'] = self.compare_file_DTW(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"], "word").item()
            df_compare['mySimilarity'] = round(df_compare['mySimilarity'], 2)
            print('[MySimilarity compute Finished]')
        
        for index, row in tqdm(df_compare.iterrows(), desc='[Similarity_cosine compute]', total=len(df_compare)):
            df_compare.loc[index, 'Similarity_cosine'] = self.compare_file_cosine(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
        print('[Similarity_cosine compute Finished]')
            
        for index, row in tqdm(df_compare.iterrows(), desc='[Similarity_doc_topic compute]', total=len(df_compare)):
            df_compare.loc[index, 'Similarity_doc_topic'] = self.compare_file_doc_topic(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
            df_compare.loc[index, 'Similarity_doc_topic'] = round(df_compare.loc[index, 'Similarity_doc_topic'], 2)
        print('[Similarity_doc_topic compute Finished]')
        
        if self.paras["open_theta"] == True:
            print(["Training the sentence theta ... "])
            self.train_theta(df_compare, "sentence")
            print(["Training the sentence theta ... Finished"])

        for index, row in tqdm(df_compare.iterrows(), desc='[Similarity_sentence_topic compute]', total=len(df_compare)):
            df_compare.loc[index, 'Similarity_sentence_topic'] = self.compare_file_DTW(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"], "sentence").item()
        df_compare['mySimilarity'] = round(df_compare['mySimilarity'], 2)
        print('[Similarity_sentence_topic compute Finished]')
        
        self.result = df_compare
        return self.result
    
    def compare_file_DTW(self, file1, file2, mode):
        result = self.DC.compare(file1, file2, mode)
        return result
    
    def compare_file_cosine(self, file1, file2):
        file1_content_str = ' '.join(self.data[str(file1) + ".txt"]["file_content"])
        file2_content_str = ' '.join(self.data[str(file2) + ".txt"]["file_content"])
        
        # 计算TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([file1_content_str, file2_content_str])
        
        # 计算余弦相似度
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(cosine_sim[0][0], 2)
    
    def compare_file_doc_topic(self, file1, file2):
        bow1 = self.lda_model.id2word.doc2bow(self.data[str(file1) + ".txt"]["file_content"])
        bow2 = self.lda_model.id2word.doc2bow(self.data[str(file2) + ".txt"]["file_content"])
        topic_distribution_1 = self.lda_model.get_document_topics(bow1, minimum_probability=0, minimum_phi_value=0)
        topic_distribution_2 = self.lda_model.get_document_topics(bow2, minimum_probability=0, minimum_phi_value=0)
        topic_dis_list_1 = [topic[1] for topic in topic_distribution_1]
        topic_dis_list_2 = [topic[1] for topic in topic_distribution_2]
        cosine_sim = cosine_similarity([topic_dis_list_1], [topic_dis_list_2])
        # if file1 == '1966_236' and file2 == '1967_267':
        #     print(topic_dis_list_1)
        #     print(topic_dis_list_2)
        # if file1 == '1983_129' and file2 == '1983_27':
        #     print(topic_dis_list_1)
        #     print(topic_dis_list_2)
        return round(cosine_sim[0][0], 2)
        
    def read_file(self):
        df_compare = pd.read_csv(self.compare_path, names=["file1", "file2", "Similarity"])
        return df_compare
    
    def train_step(self, df_compare, optimizer, mode):
        mysimilarty_list = []
        for index, row in df_compare.iterrows():
            mysimilarty_list.append(self.compare_file_DTW(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"], mode))
        
        mysimilarty = torch.tensor(mysimilarty_list, dtype=torch.float32, requires_grad=True)
        truesimilarity = torch.tensor(df_compare['Similarity'].values, dtype=torch.float32, requires_grad=True)
        def loss_fn(y_pred, y):
            return torch.mean(((y_pred - y) ** 2).mean()) ** 0.5

        loss = loss_fn(mysimilarty, truesimilarity)
        loss.backward()
        optimizer.step()
        return loss

    def train_theta(self, df_compare, mode):
        self.DC.theta = torch.tensor(20.0, dtype=torch.float32, requires_grad=True)
        learning_rate = 0.001
        epochs = 10
        
        optimizer = optim.SGD([self.DC.theta], lr=learning_rate)
        optimizer.zero_grad()
        
        for epoch in range(epochs):
            loss = self.train_step(df_compare, optimizer, mode)
            print(f"Epoch {epoch}: Loss = {loss.item()}, w = {self.DC.theta.item()}")

        print(f"Final parameter: theta = {self.DC.theta.item()}")
        
        return self.DC.theta.item()
        
        