import os

from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gym
from gym import spaces
import scipy.io as scio
import numpy_financial as npf

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
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        limitage=110,
        initialage=35,
        Nrm=100,
        FRi=1,
        NT=10,
        dt=1/12,
        tech_indicator_list=None,
        turbulence_threshold=None,
        lookback=252,
        startday='2019-01-03',
        endday='2020-01-14',
        file="sp500_ohlcv.pqt",
        # symbol_list=["105.AAL"]
    ) -> None:
        # 1) Pension parameters
        self.limitage = limitage  # limiting age
        self.initialage = initialage  # starting age of pension cohorts
        self.Nrm = Nrm  # number of pension participants in the portfolio
        self.FRi = FRi  # Initial Funding Ratio %1 for fully funded plan

        # 2) simulation specific
        self.NT = NT  # total number of simulated scenarios - 1000 just for now
        self.dt = dt  # per unit time step  #monthly step is 1/12; yearly is 1 - however needs yearly mortality rates
        self.NS = int((limitage - initialage) / dt)  # number of months from age 35 to 110
        
        self.startday = startday
        self.endday = endday
        self.lookback = lookback
        self.file = file
        self.df = pd.read_parquet(self.file, engine='pyarrow')  
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list

        # 从start day到end day的所有股票数量self.stocks_and_bonds_dim
        self.df.reset_index(inplace=True)
        self.filtered_df = self.df[(self.df['date'] >= self.startday) & (self.df['date'] <= self.endday)]
        unique_symbols_count = self.filtered_df['symbol'].nunique()
        self.stocks_and_bonds_dim = unique_symbols_count + 1

        # 所有的symbol，按照字符序排序
        self.sorted_symbols = sorted(self.filtered_df['symbol'].unique())
        self.date = self.filtered_df['date'].unique()
        self.date = self._find_closest_dates(self.startday, self.date)
        # print("date is {}".format(len(self.date)))
        # action_space normalization and shape is self.stock_dim
        self.action_space = self.stocks_and_bonds_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))

        # 所有的特征列
        self.feature_columns = [col for col in self.filtered_df.columns if col not in ['date', 'symbol', '日期']]
        self.state_space = len(self.feature_columns)

        #state shape is (self.stocks_and_bonds_dim, self.state_space)
        self.state = np.zeros((len(self.sorted_symbols), self.state_space))

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold

        # memorize portfolio value each step
        self.asset_memory = []
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = []
        self.date_memory = []

        self.PenPortfolio = np.array(pd.read_csv('Hypothetical_Pension.csv'))
        # (age, gender, Target monthly benefits, current monthly salary, accrued years) Age with 35, 45, 55, 65, 75, and 85 only
        # gender 1=male and 2=female

        # 2) Mortality rates for pension cohort (preprocessed based on pension portfolio)
        self.survivalm = scio.loadmat('PortfolioSurRates.mat')['survivalm']
        self.survivalcum = scio.loadmat('PortfolioSurRates.mat')['survivalcum']
        self.survivalm1 = scio.loadmat('PortfolioSurRates.mat')['survivalm1']
        self.survivalcum1 = scio.loadmat('PortfolioSurRates.mat')['survivalcum1']
        # 3) scenarios of assets returns and pension discount rate
        # 0.004867551 改成aa的月度利率
        self.InF = (1 + 0.004867551) * np.ones((self.NS, self.NT))  # As illustration, assuming monthly portfolio return is 0.07/12;
        self.discpen = 0.06 * np.ones((self.NS, self.NT))  # annual pension discount rate - again here is just an illustration
        # 0.06 改成债券年化利率
        self.discpenM = (1 + self.discpen) ** (self.dt) - 1  # monthly effective pension discount rate
        self.vdiscpenM = (1 + self.discpen) ** (-self.dt)  # Convert annual effective discount rate to monthly discount factor v =1/(1+i)^(dt)

        # 4) scenarios of survival paths of each pension corhorts
        self.survivalmNT = np.tile(self.survivalm[:, :, np.newaxis], (1, 1, self.NT))  # Copy the survival rates of cohorts for NT scenarios
        self.survivalT = np.random.binomial(1, self.survivalmNT)  # Randomly generate the survival status, 1 means alive and 0 means dead in that period
        self.survivalT = np.cumprod(self.survivalT, axis=0)  # 1 if the person is alive by that period and 0 means dead in this period and thereafter
        # Note that the above is the survival status of cohort at the end of each month
        self.AnxP = self._cal_Anxg()
        self.RetT, self.PLEnd, self.NCT = self._cal_RetT_PLEnd_NCT()
        self.Fd = np.zeros((self.NS, NT))
        self.SBeg = np.zeros((self.NS + 1, self.NT))  # Investment account value at the beginning of each period after normal contribution
        self.SEnd = np.zeros((self.NS, self.NT))  # Account value at the end of each period before after pension payments
        # ii) Initital Assets determination
        AccrBen = (self.PenPortfolio[:, 2] * np.minimum((self.PenPortfolio[:, 4] + 1 / 12) / 35, 1))
        AccrBen = AccrBen.reshape(len(AccrBen),1) # Accrued monthly benefits at time 0 plus normal contribution at 0 (for initial asset calculation purpose)
        S0 = np.sum(np.tile(AccrBen,(1,self.NT))*self.AnxP[0, :, :],axis=0)  # sum of initial pension liability (t=0) immediately after normal contribution of each cohort - total pension liability
        self.SBeg[0, :] = S0 * self.FRi
        

    def _find_closest_dates(self, input_date, date_array):
        # 将输入日期转换为Datetime对象
        input_date = pd.to_datetime(input_date)
        
        #将日期数组转换为Datetime对象
        date_array = pd.to_datetime(date_array)

        # 创建一个空的列表来存储结果
        closest_dates = []
        
        # 对于日期数组中的每个年和月，找出与输入日期 "日" 最接近的日期
        for year in set(date_array.year):
            for month in set(date_array[date_array.year == year].month):
                # 找到当前月份的日期
                current_month_dates = date_array[(date_array.year == year) & (date_array.month == month)]
                
                # 如果当前月份没有日期，则跳过
                if len(current_month_dates) == 0:
                    continue

                # 找到与输入日期 "日" 最接近的日期
                closest_date = min(current_month_dates, key=lambda date: abs(date.day - input_date.day))
                
                # 将最接近的日期添加到结果列表中
                closest_dates.append(closest_date)

        # 将结果列表转换为DatetimeIndex
        closest_dates = pd.DatetimeIndex(closest_dates)
        closest_dates = closest_dates.sort_values()
        return closest_dates

    def step(self, actions):
        """
        1.1的时候确定这个月的仓位，1.31的时候计算收益，收益-养老金=send；2.1的时候send+normal contribution=2.1的sbeg
        """
        self.terminal = self.day >= len(self.date) - 1
        truncated = True
        if self.terminal:
            pass
        else:
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.date_data

            self.day += 1
            self.date_data = self.filtered_df[self.filtered_df['date'] == self.date[self.day]]
            mask = np.zeros(len(self.sorted_symbols))
            # calcualte portfolio return
            portfolio_return = 0
            for i, symbol in enumerate(self.sorted_symbols):
                symbol_data = self.date_data[self.date_data['symbol'] == symbol][self.feature_columns]
                last_symbol_data = last_day_memory[last_day_memory['symbol'] == symbol][self.feature_columns]
                if symbol_data.empty:
                    self.state[i,:]=np.zeros(self.state_space)
                else:
                    self.state[i,:]=symbol_data
                    mask[i] = 1
                    portfolio_return += (symbol_data.values[0][1] / last_symbol_data.values[0][1] - 1) * weights[i]
            
            # update portfolio value
            new_portfolio_value = self.SBeg[self.day-1, :] * (1 + portfolio_return)
            print("new_portfolio_value is {}".format(new_portfolio_value))
            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.date[self.day])
            self.asset_memory.append(new_portfolio_value)

            self.reward = new_portfolio_value

            self.SEnd[self.day-1,:] = new_portfolio_value - self.RetT[self.day-1, :]
            # print(self.RetT[self.day-1, :])
            self.Fd[self.day-1,:] = self.SEnd[self.day-1,:] / self.PLEnd[self.day-1, :]
            if  self.Fd[self.day-1][0] < 0.75:
                truncated = False
            self.SBeg[self.day, :] = self.SEnd[self.day-1,:] + self.NCT[self.day-1, :]
            
            return self.state, self.reward, self.terminal, truncated, {"mask":mask}

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):
        self.asset_memory = [self.SBeg[0,:]]
        self.day = 0
        self.date_data = self.filtered_df[self.filtered_df['date'] == self.date[self.day]]
        mask = np.zeros(len(self.sorted_symbols))
        for i, symbol in enumerate(self.sorted_symbols):
            symbol_data = self.date_data[self.date_data['symbol'] == symbol][self.feature_columns]
            if symbol_data.empty:
                self.state[i,:]=np.zeros(self.state_space)
            else:
                self.state[i,:]=symbol_data
                mask[i] = 1
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.date_memory.append(self.date[self.day])

        return self.state, {"mask":mask}

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def render(self):
        print(self.state)
    
    def _cal_RetT_PLEnd_NCT(self):
        # Part III Pension Funding Status Calculation for each scenario
        # 1) Pension liabilities valuation
        # PUC method, assuming the normal contribution is made at the beginning of
        # each period
        # i) Define pension fund quantities for each period
        PLEnd = np.zeros((self.NS, self.NT))  # Pension liability at the end of each period after pension payments
        RetT = np.zeros((self.NS, self.NT))  # total retirement payment at the end of each period [intermediate results]
        NCT = np.zeros((self.NS, self.NT))  # total normal contributions for each period [intermediate results]

        # iii) Accrued Pension liability of each person at the end of each period assuming the cohort is alive
        AccYr = np.arange(1, self.NS + 1)
        AccrBenEnd = (self.PenPortfolio[:, 2].reshape(self.PenPortfolio.shape[0],1) * np.minimum(
            (self.PenPortfolio[:, 4].reshape(self.PenPortfolio.shape[0],1) + AccYr.reshape(1,len(AccYr)) / 12) / 35, 1)).T # Accrued benefits by the end of each period

        # iv) Normal Contribution and Retirement Payments
        NormC = np.zeros((self.NS, self.Nrm, self.NT))  # Normal contribution Matrix
        RetireP = np.zeros((self.NS, self.Nrm, self.NT))  # Pension Payments Matrix

        for i in range(self.Nrm):
            if self.PenPortfolio[i, 0] < 65:  # for working employees at t=0
                NormC[:int((65 - self.PenPortfolio[i, 0]) / self.dt) - 1, i, :] = self.PenPortfolio[i, 2] / 35 * self.dt * self.AnxP[:int((65 - self.PenPortfolio[i, 0]) / self.dt) - 1, i, :]  # Normal contribution made at the beginning of period from the second period
                RetireP[int((65 - self.PenPortfolio[i, 0]) / self.dt):, i, :] = self.PenPortfolio[i, 2] + RetireP[int((65 - self.PenPortfolio[i, 0]) / self.dt):, i, :]  # Retirement benefits paid after they reach 65
            else:
                RetireP[:, i, :] = self.PenPortfolio[i, 2] + RetireP[:, i, :]  # For retirees at t=0, reirement benefits already started to pay

        # v) Pension Funding Ratio process
        for j in range(self.NS - 1):  # j represents each month starting from the first month
            # Ret
            RetT[j, :] = np.sum(self.survivalT[j, :, :] * RetireP[j, :, :], axis=0)  # total retirement payments made at jth period
            # SEnd[j, :] = SBeg[j, :] * self.InF[j, :] - RetT[j, :]  # At the end of each period, payments are made to retirees who are alive
            PLEnd[j, :] = np.sum(self.survivalT[j, :, :] * np.tile(AccrBenEnd[j, :].reshape(len(AccrBenEnd[j,:]),1),(1,self.NT)) * self.AnxP[j + 1, :, :], axis=0)  # Pension liablity at the end of the year
            # Fd[j, :] = SEnd[j, :] / PLEnd[j, :]  # funding status at the end of each period
            NCT[j, :] = np.sum(self.survivalT[j, :, :] * NormC[j, :, :], axis=0)  # total normal contribution made at the beginning of j+1 period
            # SBeg[j + 1, :] = SEnd[j, :] + NCT[j, :]  # asset values at the beginning of each period - add normal contribution
        return RetT, PLEnd, NCT

    def _cal_liability(self, j, SBeg):
        SEnd = SBeg * self.InF[j, :] - self.RetT[j, :]  # At the end of each period, payments are made to retirees who are alive
        Fd = SEnd / self.PLEnd[j, :]
        SBeg[j + 1, :] = SEnd + self.NCT[j, :]  # asset values at the beginning of each period - add normal contribution

    def _cal_Anxg(self):
        # 5) Annuity-immediate values with $1 per month, Anx = sum_t v^t* tpx; v and tpx are based on month unit 
        # If the age x+t i not reach 65, the value is a deferred annuity 65-(x+t)|a65
        Anx = np.zeros((self.NS, 12, self.NT))  # Define the Annuity-immediate Factor for pension liability for age x+t (by row) for each possible age/gender
        # for the middle 12: initial ages 35, 45, 55, 65, 75, 85, male (first 6)
        # /female (last 6)
        for i0 in range(1, 4):
            # j0=1

            cf = np.concatenate([np.zeros(int((65-(25+10*i0))/self.dt+1)),self.survivalcum1[int((65-(25+10*i0))/self.dt):,i0-1]
                    ])
            cash_flow = np.tile(cf.reshape(len(cf),1),(1,self.NT))
            rate_of_return = self.discpenM[0,:]
            Anx[0, i0 - 1, :] = [npf.npv(rate_of_return[i], cash_flow[:,i]) for i in range(len(rate_of_return))]# for male

            cf = np.concatenate([np.zeros(int((65-(25+10*i0))/self.dt+1)),self.survivalcum1[int((65-(25+10*i0))/self.dt):,5+i0]
                    ])
            cash_flow = np.tile(cf.reshape(len(cf),1),(1,self.NT))
            Anx[0, 5 + i0, :] = [npf.npv(rate_of_return[i], cash_flow[:,i]) for i in range(len(rate_of_return))]  # for female
            
            for j0 in range(2, int((65 - (25 + 10 * i0)) / self.dt) + 1):
                rate_of_return = self.discpenM[j0-1,:]
                cf = np.concatenate([np.zeros(int((65-(25+10*i0))/self.dt-j0+2)),self.survivalcum1[int((65-(25+10*i0))/self.dt):,i0-1] / self.survivalcum1[j0 - 2, i0-1]
                    ])
                cash_flow = np.tile(cf.reshape(len(cf),1),(1,self.NT))
                Anx[j0 - 1, i0 - 1, :] = [npf.npv(rate_of_return[i], cash_flow[:,i]) for i in range(len(rate_of_return))]  # for male
                
                cf = np.concatenate([np.zeros(int((65-(25+10*i0))/self.dt-j0+2)),self.survivalcum1[int((65-(25+10*i0))/self.dt):,i0+5] / self.survivalcum1[j0 - 2, i0+5]
                    ])
                cash_flow = np.tile(cf.reshape(len(cf),1),(1,self.NT))
                Anx[j0 - 1, 5 + i0, :] = [npf.npv(rate_of_return[i], cash_flow[:,i]) for i in range(len(rate_of_return))] # for female
            
            for j0 in range(int((65 - (25 + 10 * i0)) / self.dt) + 1, int((self.limitage - (25 + 10 * i0)) / self.dt) + 1):
                rate_of_return = self.discpenM[j0-1,:]
                cf = np.concatenate([[0],self.survivalcum1[j0-1:,i0-1] / self.survivalcum1[j0 - 2, i0-1]
                    ])
                cash_flow = np.tile(cf.reshape(len(cf),1),(1,self.NT))
                Anx[j0 - 1, i0 - 1, :] = [npf.npv(rate_of_return[i], cash_flow[:,i]) for i in range(len(rate_of_return))] # for male

                cf = np.concatenate([[0],self.survivalcum1[j0-1:,i0+5] / self.survivalcum1[j0 - 2, i0+5]
                    ])
                cash_flow = np.tile(cf.reshape(len(cf),1),(1,self.NT))
                Anx[j0 - 1, 5 + i0, :] = [npf.npv(rate_of_return[i], cash_flow[:,i]) for i in range(len(rate_of_return))] # for female


        for i0 in range(4, 7):
            rate_of_return = self.discpenM[0,:]
            cf = np.concatenate([[0],self.survivalcum1[:,i0-1]
                    ])
            cash_flow = np.tile(cf.reshape(len(cf),1),(1,self.NT))
            Anx[0, i0 - 1, :] = [npf.npv(rate_of_return[i], cash_flow[:,i]) for i in range(len(rate_of_return))]  # for male
            
            cf = np.concatenate([[0],self.survivalcum1[:,i0+5]
                    ])
            cash_flow = np.tile(cf.reshape(len(cf),1),(1,self.NT))
            Anx[0, 5 + i0, :] = [npf.npv(rate_of_return[i], cash_flow[:,i]) for i in range(len(rate_of_return))]  # for female
            
            for j0 in range(2, int((self.limitage - (25 + 10 * i0)) / self.dt) + 1):
                rate_of_return = self.discpenM[j0-1,:]
                
                cf = np.concatenate([[0],self.survivalcum1[j0-1:,i0-1] / self.survivalcum1[j0 - 2, i0-1]
                    ])
                cash_flow = np.tile(cf.reshape(len(cf),1),(1,self.NT))
                Anx[j0 - 1, i0 - 1, :] = [npf.npv(rate_of_return[i], cash_flow[:,i]) for i in range(len(rate_of_return))]  # for male
                
                cf = np.concatenate([[0],self.survivalcum1[j0-1:,i0+5] / self.survivalcum1[j0 - 2, i0+5]
                    ])
                cash_flow = np.tile(cf.reshape(len(cf),1),(1,self.NT))
                Anx[j0 - 1, 5 + i0, :] = [npf.npv(rate_of_return[i], cash_flow[:,i]) for i in range(len(rate_of_return))]  # for female


        AnxP = np.zeros((self.NS, self.Nrm, self.NT))
        for i in range(self.Nrm):
            AnxP[:, i, :] = Anx[:, int((self.PenPortfolio[i, 0] - 25) / 10 + (self.PenPortfolio[i, 1] - 1) * 6 - 1), :]  # Find the proper annuity factor for each cohort
        return AnxP

def make(
    id : str,
    **kwargs: Any,
):
    if id == 'PensionInvestment-v0':
        return PensionPortfolioEnv(id, **kwargs)
    else:
        raise NotImplementedError('Only PensionInvestment-v0 now.')