import pandas as pd  
import numpy as np

# 读取 .pqt 文件  
data = pd.read_parquet('sp500_ohlcv.pqt')  
data = data.loc['2000-01-03':'2000-11-03']
# 查看数据框的前几行  
# print(data.head(10))
# print(type(data.head(1)))  
# print(data.loc[('2000-01-03','105.AAPL')])
# result = data.groupby('date')['symbol'].unique()
result = data.reset_index().groupby('date')['symbol'].unique()

# print(data.index.get_level_values('date').unique())
# result = data.groupby(level='symbol')
# common_symbols = result[result == len(data.index.get_level_values('date').unique())].index
# print(type(result[0]))
arrays = result.values.tolist()
# print(len(arrays))
# 将每个NumPy数组转换为列表
arrays = [array.tolist() for array in arrays]

# 将所有列表连接成一个大列表
concatenated_list = np.concatenate(arrays)

# 使用numpy.unique函数获取元素和对应的计数
unique_elements, counts = np.unique(concatenated_list, return_counts=True)

# 打印每个元素及其出现次数
for element, count in zip(unique_elements, counts):
    # if count == 214:
    print(f"Element {element} appears {count} times.")