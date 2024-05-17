from gensim import corpora, models
from gensim.test.utils import datapath

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import torch

# 示例文档数据集
texts = [
    "Kimi is an AI developed by Moonshot Moonshot AI. Kimi.",
    "Kimi can read documents and answer questions.",
    "Moonshot AI is a company based in China.",
    "Kimi can process multiple files and search the internet."
]

text_ls = []

for text in texts:
    # 分词处理
    tokens = word_tokenize(text)

    # 停用词处理
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    
    # 词干提取
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    # print(stemmed_tokens)
    text_ls.append(stemmed_tokens)

# 输入 [[],[],[],[],[]] - (text number, text lenth) 输出一个gensim.corpora.dictionary.Dictionary类型
dictionary = corpora.Dictionary(text_ls)

# 转换文档为词袋模型
corpus = [dictionary.doc2bow(text) for text in text_ls]

# 构建 LDA 模型 passes 为遍历所有文档的次数
lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=4, passes=100)

# Save model to disk.
# temp_file = datapath("C:\\Users\\Winston\\Desktop\\Repository\\TextSimilarity\\ldadiskmodel\\lda_model")
# lda_model.save(temp_file)

text_topics_embedded_ls = [] # (text numbers, text length, topic numbers)
for text in text_ls:
    print("text: ", text)
    text_topics_embedded = []
    for token in text:
        word_id = dictionary.token2id[token]
        token_topics_distribution = [topic[1] for topic in lda_model.get_term_topics(word_id, minimum_probability=0)]
        print(token_topics_distribution)
        text_topics_embedded.append(token_topics_distribution)
    text_topics_embedded_ls.append(text_topics_embedded)
    print("distribution: ", text_topics_embedded)
    '''
    text_topics_embedded eg.[0] 每次训练dis不同
    [
        [0.12858568, 0.004175563, 0.093703076, 0.004175563], 
        [0.12861314, 0.004166925, 0.0017183346, 0.004166925], 
        [0.037034746, 0.0041671097, 0.0017184492, 0.0041671097], 
        [0.12861314, 0.004166925, 0.0017183346, 0.004166925], 
        [0.12861314, 0.004166925, 0.0017183346, 0.004166925], 
        [0.12861314, 0.004166925, 0.0017183346, 0.004166925], 
        [0.12858568, 0.004175563, 0.093703076, 0.004175563]
    ]
    ''' 

# ls =[[[[1,2,3],[3,2,5]],[[1,2,3],[3,2,5]],[[1,2,3],[3,2,5]],[[1,2,3],[3,2,5]],[[1,2,3],[3,2,5]]]]

# a = torch.tensor(ls)

# print(a.shape)