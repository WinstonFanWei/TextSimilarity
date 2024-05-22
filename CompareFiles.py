import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from distancecompare.DistanceCompare import DistanceCompare
from tqdm import tqdm

class CompareFiles:
    def __init__(self, lda_model, word2vec_model, data, compare_path, paras):
        self.MySimilarityCompute = paras["MySimilarityCompute"]
        self.lda_model = lda_model
        self.word2vec_model = word2vec_model
        self.data = data
        self.compare_path = compare_path
        self.result = None
        self.DC = DistanceCompare(self.lda_model, self.word2vec_model, self.data, paras["topic_distance_matrix_iscomputed"])
    def compare(self):
        df_compare = self.read_file()
        
        if self.MySimilarityCompute == True:
            for index, row in tqdm(df_compare.iterrows(), desc='[MySimilarity compute]', total=len(df_compare)):
                df_compare.loc[index, 'mySimilarity'] = self.compare_file_DTW(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
            print('[MySimilarity compute Finished]')
        
        for index, row in tqdm(df_compare.iterrows(), desc='[Similarity_cosine compute]', total=len(df_compare)):
            df_compare.loc[index, 'Similarity_cosine'] = self.compare_file_cosine(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
        print('[Similarity_cosine compute Finished]')
            
        for index, row in tqdm(df_compare.iterrows(), desc='[Similarity_doc_topic compute]', total=len(df_compare)):
            df_compare.loc[index, 'Similarity_doc_topic'] = self.compare_file_doc_topic(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
        print('[Similarity_doc_topic compute Finished]')
            
        for index, row in tqdm(df_compare.iterrows(), desc='[Similarity_half compute]', total=len(df_compare)):
            df_compare.loc[index, 'Similarity_half'] = self.compare_file_half(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
        print('[Similarity_half compute Finished]')
        
        self.result = df_compare
        return self.result
    
    def compare_file_DTW(self, file1, file2):
        result = self.DC.compare(file1, file2)
        return round(result, 2)
    
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
        return round(cosine_sim[0][0], 2)
    
    def compare_file_half(self, file1, file2):
        return round(0.5, 2)
        
    def read_file(self):
        df_compare = pd.read_csv(self.compare_path, names=["file1", "file2", "Similarity"])
        return df_compare