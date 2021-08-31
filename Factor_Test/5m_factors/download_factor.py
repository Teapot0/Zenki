import pandas as pd
import jqfactor_analyzer as ja
import matplotlib.pyplot as plt
from tqdm import tqdm
from jqdatasdk import auth, get_query_count
from jqdatasdk import alpha191
auth('13382017213', 'Aasd120120')
get_query_count()

import warnings
warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')

date_list = list(close.index)
stock_list = list(close.columns)
out = pd.DataFrame(index=date_list, columns = stock_list)
for d in tqdm(date_list):
    out.loc[d] = alpha191.alpha_074(stock_list,d,fq ='pre')
out.to_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_074.csv')







