import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from distancecompare.DistanceCompare import DistanceCompare
from tqdm import tqdm

class CompareFiles:
    def __init__(self, lda_model, word2vec_model, data, compare_path):
        self.lda_model = lda_model
        self.word2vec_model = word2vec_model
        self.data = data
        self.compare_path = compare_path
        self.result = None
        self.DC = DistanceCompare(self.lda_model, self.word2vec_model, self.data)
    def compare(self):
        df_compare = self.read_file()
        
        for index, row in tqdm(df_compare.iterrows(), desc='[MySimilarity compute]', total=len(df_compare)):
            df_compare.loc[index, 'mySimilarity'] = self.compare_file_DTW(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
        print('[MySimilarity compute Finished]')
        
        for index, row in tqdm(df_compare.iterrows(), desc='[Similarity_cosine compute]', total=len(df_compare)):
            df_compare.loc[index, 'Similarity_cosine'] = self.compare_file_cosine(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
        print('[Similarity_cosine compute Finished]')
            
        for index, row in tqdm(df_compare.iterrows(), desc='[Similarity_2 compute]', total=len(df_compare)):
            df_compare.loc[index, 'Similarity_2'] = self.compare_file_type2(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
        print('[Similarity_2 compute Finished]')
            
        for index, row in tqdm(df_compare.iterrows(), desc='[Similarity_half compute]', total=len(df_compare)):
            df_compare.loc[index, 'Similarity_half'] = self.compare_file_half(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
        print('[Similarity_half compute Finished]')
        
        self.result = df_compare
        return self.result
    
    def compare_file_DTW(self, file1, file2):
        result = self.DC.compare(file1, file2)
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
    
    def compare_file_type2(self, file1, file2):
        return 0
    
    def compare_file_half(self, file1, file2):
        return 0.5
        
    def read_file(self):
        df_compare = pd.read_csv(self.compare_path, names=["file1", "file2", "Similarity"])
        return df_compare