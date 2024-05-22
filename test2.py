import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pprint import pprint
from gensim.models import Word2Vec
from CompareFiles import CompareFiles

# 创建示例文档
documents = [
    ['human', 'interface', 'computer'],
    ['survey', 'computer', 'system'],
    ['interface', 'system'],
]

# 创建词典
dictionary = Dictionary(documents)

# 创建语料库
corpus = [dictionary.doc2bow(doc) for doc in documents]

# 训练LDA模型
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# 打印词频字典
print("词频字典 (id2word.dfs):")
pprint(dictionary.dfs)

# 筛选词频大于5的词
filtered_word_list = [lda_model.id2word[word_id] for word_id, freq in lda_model.id2word.dfs.items() if freq > 1]

filtered_word_list_ = [word_id for word_id, freq in lda_model.id2word.dfs.items() if freq > 1]

# 打印筛选后的词
print("\n筛选后词频大于5的词:")
print(filtered_word_list)
print(filtered_word_list_)
print(dictionary)
print(lda_model.get_topics())

print(lda_model.id2word.dfs.items())
print(lda_model.id2word.dfs)
