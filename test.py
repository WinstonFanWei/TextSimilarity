import numpy as np
from tslearn.metrics import dtw_path_from_metric

# 定义两个时间序列
x = np.array([[0], [1], [2]])
y = np.array([[3], [4]])

# 定义固定的代价矩阵
fixed_cost_matrix = np.array([[0, 100, 200, 100, 900],
                               [100, 0, 100, 100, 900],
                               [200, 100, 0, 100, 900],
                               [100, 100, 100, 0, 100],
                               [900, 900, 900, 100, 0]])
# 定义自定义的距离度量函数
def custom_distance(i, j):
    print(fixed_cost_matrix[int(i[0]), int(j[0])])
    return fixed_cost_matrix[int(i[0]), int(j[0])]

# 使用自定义距离度量函数计算DTW路径和距离
path, dtw_distance = dtw_path_from_metric(x, y, metric=custom_distance)

print(f"DTW 距离: {dtw_distance}")
print("路径:")
print(path)