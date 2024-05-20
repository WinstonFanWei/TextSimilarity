import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class CompareFiles:
    def __init__(self, data, compare_path):
        self.data = data
        self.compare_path = compare_path
        self.result = None
    def compare(self):
        df_compare = self.read_file()
        
        for index, row in df_compare.iterrows():
            df_compare.loc[index, 'mySimilarity'] = self.compare_file_DTW(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
        
        for index, row in df_compare.iterrows():
            df_compare.loc[index, 'Similarity_cosine'] = self.compare_file_cosine(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
            
        for index, row in df_compare.iterrows():
            df_compare.loc[index, 'Similarity_2'] = self.compare_file_type2(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])
            
        for index, row in df_compare.iterrows():
            df_compare.loc[index, 'Similarity_3'] = self.compare_file_type3(df_compare.loc[index, "file1"], df_compare.loc[index, "file2"])

        self.result = df_compare
        self.write_file()
        return self.result
    
    def compare_file_DTW(self, file1, file2):
        # to do
        return 1
    
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
        return 1
    
    def compare_file_type3(self, file1, file2):
        return 1
    
    def write_file(self):
        self.result.to_csv('output.csv', index=False)  # 不包含索引
        
    def read_file(self):
        df_compare = pd.read_csv(self.compare_path, names=["file1", "file2", "similiarity"])
        return df_compare