# TextSimilarity

<a href="https://github.com/WinstonFanWei/TextSimilarity">Here is the code link: https://github.com/WinstonFanWei/TextSimilarity</a>

## Use git

```bash
git clone git@github.com:WinstonFanWei/TextSimilarity.git
```

## For teacher

Get the files from bb, and config as introduction.

## Introduction

Please install the required environment, and config ```main.py``` as following, then you can run the ```main.py``` to reproduce the results in the paper.

```python
paras = {
    "file_path": your path + "\\document-similarity-main\\document-similarity-main", # 数据集路径
    "LDA_load_path": your path + "\\modelsave\\lda_model.model", # LDA模型保存路径
    
    "LDA_isload": False, # LDA模型是否已经保存下来，保存了就不用再次训练
    "Debug": False, # Debug模式下不进行相似度的计算 即CompareFiles()
    "topic_distance_matrix_iscomputed": False, # topic距离矩阵是否已经计算完成
    "MySimilarityCompute": True, # 是否计算我们的主要相似度度量方式
    "open_theta": False # 暂时关闭theta的逻辑
}
```

## Environment requirements

```
pip install -r requirements.txt
```