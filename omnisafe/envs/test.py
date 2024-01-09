import pandas as pd
import numpy as np
# Load the Parquet file and read the first 10 rows
df = pd.read_parquet('sp500_ohlcv.pqt', engine='pyarrow')

# df.set_index(['date', 'symbol'], inplace=True)

# specific_date = '2000-01-03'  # 请替换为具体的日期
# specific_symbol = '105.AAPL'      # 请替换为具体的符号

# # 使用多重索引获取行
# specific_row = df.loc[(specific_date, specific_symbol)]

# # 假设您想要获取特定日期的所有数据
# specific_date = '2000-01-03'  # 请替换为具体的日期

# # 使用 .xs 方法来获取cross-section
# date_data = df.xs(specific_date, level='date')

# print(type(date_data))

# # 或者，如果您想要基于特定符号获取所有数据
# specific_symbol = '105.AAPL'  # 请替换为具体的符号

# # 使用 .xs 方法来获取cross-section
# symbol_data = df.xs(specific_symbol, level='symbol')

# print(symbol_data)


# 请替换成您的 .pqt 文件的路径
# file_path = 'path_to_your_file.pqt'

# 加载 .pqt 文件
# df = pd.read_parquet(file_path, engine='pyarrow')

# 设置两个特定的日期
# specific_date1 = '2000-01-03'  # 起始日期
# specific_date2 = '2022-01-05'  # 结束日期

# # 首先，确保索引是日期
# df.reset_index(inplace=True)

# # 然后，选择这个日期范围内的数据
# filtered_df = df[(df['date'] >= specific_date1) & (df['date'] <= specific_date2)]

# sorted_symbols = sorted(filtered_df['symbol'].unique())

# print(len(sorted_symbols))

# # 最后，统计唯一的 'symbol' 数量
# unique_symbols_count = filtered_df['symbol'].nunique()

# print(unique_symbols_count)
# Set two specific dates
specific_date1 = '2000-01-03'  # Starting date
specific_date2 = '2000-01-05'  # Ending date

# Ensure 'date' and 'symbol' are not set as an index
if isinstance(df.index, pd.MultiIndex):
    df.reset_index(inplace=True)

# Filter the DataFrame to only include rows between the specified dates
filtered_df = df[(df['date'] >= specific_date1) & (df['date'] <= specific_date2)]

# Get the unique symbols and sort them
sorted_symbols = sorted(filtered_df['symbol'].unique())
# print(filtered_df.columns)
# Determine the feature dimension - number of columns excluding 'date' and 'symbol'
feature_columns = [col for col in filtered_df.columns if col not in ['date', 'symbol', '日期']]
feature_dim = len(feature_columns)

# Initialize the state variable with zeros
state = np.zeros((len(sorted_symbols), feature_dim))
# print(sorted_symbols)
# For each unique symbol, populate the state variable with the feature data
for i, symbol in enumerate(sorted_symbols):
    # Get the feature data for the current symbol
    symbol_data = filtered_df[filtered_df['symbol'] == symbol][feature_columns]
    
    # If there are multiple rows for the symbol, you may want to aggregate them
    # Here we are taking the mean, but you can use any aggregation method you prefer
    symbol_features = symbol_data.mean().values
    
    # Assign the feature data to the state variable
    state[i, :] = symbol_features

# Now 'state' is ready and contains the feature data for each symbol
print(state)

# 找到从startday到endday每天都有的股票list
# result = self.train_data.reset_index().groupby('date')['symbol'].unique()
# arrays = result.values.tolist()
# stocks_sum = len(arrays)
# arrays = [array.tolist() for array in arrays]
# # 将所有列表连接成一个大列表
# concatenated_list = np.concatenate(arrays)
# self.stock_valid = set()
# # 使用numpy.unique函数获取元素和对应的计数
# unique_elements, counts = np.unique(concatenated_list, return_counts=True)
# for element, count in zip(unique_elements, counts):
#     if count == stocks_sum:
#         self.stock_valid.add(element)