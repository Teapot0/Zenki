
import pandas as pd
from tqdm import tqdm

data1 = pd.read_excel('/.data_1.xlsx',converters={'股票代码':str})
data2 = pd.read_excel('./data_2.xlsx',converters={'股票代码':str})
data3 = pd.read_excel('./data_3.xlsx',converters={'股票代码':str})


def transform_miaodong(df_list):
    date_list = df_list[0]['日期'].unique()
    df_0 = pd.DataFrame(index=date_list)
    df_1 = pd.DataFrame(index=date_list)
    df_2 = pd.DataFrame(index=date_list)
    df_all = pd.DataFrame(index=date_list)

    for df in df_list:
        tmp_stock = list(df['股票代码'].unique())
        for s in tqdm(tmp_stock):
            df_0[s] = df['0'][df['股票代码']==s].values
            df_1[s] = df['1'][df['股票代码']==s].values
            df_2[s] = df['2'][df['股票代码']==s].values
            df_all[s] = df['all'][df['股票代码']==s].values

    return df_0, df_1, df_2, df_all


tmp_out_0, tmp_out_1, tmp_out_2, tmp_out_3 = transform_miaodong([data1,data2,data3])






