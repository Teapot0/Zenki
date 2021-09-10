import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.basic_funcs_open import *

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



vol_corr_300 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volpct_corr_5m_300.csv',index_col='Unnamed: 0')
vol_corr_300 = vol_corr_300.dropna(how='all', axis=0)

v1 = vol_corr_300.shift(2)

diff1 = v1.diff(1)



corr_df = pd.DataFrame(index=vol_corr_300.index, columns=vol_corr_300.columns)


np.corrcoef(diff1.iloc[3:,0], v1.iloc[3:,0])







