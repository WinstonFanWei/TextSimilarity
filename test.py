from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 示例文档数据集
texts = [
    "Kimi is an AI developed by Moonshot Moonshot AI. Kimi.",
    "Kimi can read documents and answer questions.",
    "Moonshot AI is a company based in China.",
    "Kimi can process multiple files and search the internet."
]

# 分词和构建词典
texts = [[word for word in document.lower().split()] for document in texts]
dictionary = corpora.Dictionary(texts)

# 转换文档为词袋模型
corpus = [dictionary.doc2bow(text) for text in texts]

# 构建 LDA 模型
lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=2, passes=15)

# 打印主题
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# 确保已经下载了所需的 NLTK 数据包
# nltk.download('punkt')
# nltk.download('stopwords')

# 示例文本
text = "This is an example sentence. It illustrates how to use NLTK to process text."

# 分词处理
tokens = word_tokenize(text)

# 停用词处理
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]

print("Original tokens:", tokens)
print("Filtered tokens:", filtered_tokens)
