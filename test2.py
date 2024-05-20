import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# 确保已下载必要的数据包
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # 分词
    words = nltk.word_tokenize(text)
    # 转换为小写
    words = [word.lower() for word in words]
    # 去除标点符号
    words = [word for word in words if word.isalnum()]
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def calculate_cosine_similarity(doc1, doc2):
    # 预处理文档
    doc1_processed = preprocess_text(doc1)
    doc2_processed = preprocess_text(doc2)
    
    # 计算TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1_processed, doc2_processed])
    
    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    print(cosine_sim)
    return cosine_sim[0][0]

# 示例文档
doc1 = "This is a sample document."
doc2 = "This document is a sample fanwei."

similarity = calculate_cosine_similarity(doc1, doc2)
print(f"Document similarity (cosine): {similarity:.4f}")

print(cosine_similarity([[1,2],[1,2,3]]))
