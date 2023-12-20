import os

from typing import Any
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# 数据来源：1. 实际数据，大概30组。 2. 根据生成的数据回归出一个函数
# 约束的设定： 1. 单步约束，即当c_{t1}违反“100%的概率大于0.75”的约束之后，结束当前episode，并更新策略网络；
#            2. 多步约束，即将每个时刻的约束c_{ti}作累计，一整个episode结束之后（或当c违反次数达到上限时结束当前episode），更新策略网络。
#               这里的约束仍是95%的概率大于0.8，但是这里的95%是**通过违反的次数/整个episode**来计算的。
#            3. 单步约束，约束保持95%的概率大于0.8，我们使用分位数拟合一个网络来预测分位点的值：
#                       f(s_t,a_t)=F^{-1}_{D_{t+1}}(0.05)
#               这里D_{t+1}代表t+1时刻的资产负债比，F^{-1}代表累计分布函数的逆函数
#               或者直接拟合整个分布而不止是某一个分位点的值
#                       f(s_t,a_t,\tau)=F^{-1}_{D_{t+1}}(\tau)
# 强化学习的目标：max_{\theta} J_{PPO}
#              s.t. f(s_t,a_t) > 0.8, 任意t 
# 状态obs:shape为(state_space+len(tech_indicator_list), state_space), 比如有30支股票，tech_indicator_list的长度为10
#       那么最终的shape就为(40, 30)，这里的state_space用协方差矩阵来表示。比如收益率等单个指标。
# 动作设定：actions are the portfolio weight
# Reward设定：the reward is the new portfolio value or end portfolo value

class PensionPortfolioEnv(gym.Env):
    """A pension trading environment for OpenAI gym

    Attribute
    ---------
        df: DataFrame
            input data
        stocks_and_bonds_dim: int
            number of unique stocks and bonds
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        lookback: int
            None
        day: int
            an increment number to control date
    
    Methods
    -------
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()

    """

    def __init__(
        self,
        stocks_and_bonds_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        startday='2000-01-03',
        endday='2023-11-14',
        file="sp500_ohlcv.pqt",
        mode=1,
    ) -> None:
        self.startday = startday
        self.endday = endday
        self.lookback = lookback
        self.file = file
        self.df = pd.read_parquet(self.file)  
        self.stocks_and_bonds_dim = stocks_and_bonds_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space + len(self.tech_indicator_list), self.state_space)
        )

        # load all data from a pandas dataframe
        self.train_data = self.df.loc[self.startday:self.endday]
        
        # 找到从startday到endday每天都有的股票list
        result = self.train_data.reset_index().groupby('date')['symbol'].unique()
        arrays = result.values.tolist()
        stocks_sum = len(arrays)
        arrays = [array.tolist() for array in arrays]
        # 将所有列表连接成一个大列表
        concatenated_list = np.concatenate(arrays)
        self.stock_valid = set()
        # 使用numpy.unique函数获取元素和对应的计数
        unique_elements, counts = np.unique(concatenated_list, return_counts=True)
        for element, count in zip(unique_elements, counts):
            if count == stocks_sum:
                self.stock_valid.add(element)

        self.covs = self.data["cov_list"].values[0]
        # self.state = np.append(
        #     np.array(self.covs),
        #     [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
        #     axis=0,
        # )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold

        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]


    def cal_constraint_one(self):
        return False

    def step(self, action):
        self.terminal = False
        if mode == 1:
            self.terminal = self.cal_constraint_one(ß)
        else:
            raise NotImplementedError('Only support mode=1 now.')

        if self.terminal:
            pass
        else:
            pass

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        # self.data = 
        self.portfolio_value = self.initial_amount


    def save_asset_memoty(self):
        pass

    def cal_asset_liability_rate(self):
        pass
    
    def cal_liability(self):
        pass

def make(
    id : str,
    **kwargs: Any,
):
    if id == 'PensionInvestment-v0':
        return PensionPortfolioEnv(id, **kwargs)
    else:
        raise NotImplementedError('Only PensionInvestment-v0 now.')