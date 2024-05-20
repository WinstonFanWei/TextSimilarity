import pandas as pd

class CompareFiles:
    def __init__(self, data, compare_path):
        self.data = data
        self.compare_path = compare_path
        self.result = None
    def compare(self):
        df_compare = self.read_file()
        df_compare["mySimilarity"] = self.compare_file(df_compare["file1"], df_compare["file2"])
        self.result = df_compare
        self.write_file()
        return self.result
    
    def compare_file(self, file1, file2):
        # to do
        return 1
    
    def write_file(self):
        self.result.to_csv('output.csv', index=False)  # 不包含索引
        
    def read_file(self):
        df_compare = pd.read_csv(self.compare_path, names=["file1", "file2", "similiarity"])
        return df_compare