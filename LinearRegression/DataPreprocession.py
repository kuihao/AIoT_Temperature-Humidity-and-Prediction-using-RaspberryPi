import numpy as np 
import matplotlib.pyplot as plt #畫出圖型 
import pandas as pd #資料處理
pd.set_option("display.max_rows", 1000)    #設定最大能顯示1000rows
pd.set_option("display.max_columns", 1000) #設定最大能顯示1000columns

'''
# 匯入資料
'''
# dataset = pd.read_csv('./TrainingData_2019_Pinzhen/CSV_2019PingZhen.csv') # Command Line外部執行
dataset = pd.read_csv('LinearRegression\TrainingData_2019_Pinzhen\CSV_2019PingZhen.csv') # VScode內部執行
'''
# 空值填寫、資料洗清
    原始資料註記說明：# 表示儀器檢核為無效值，* 表示程式檢核為無效值，x 表示人工檢核為無效值，
    A 係指因儀器疑似故障警報所產生的無效值，NR 表示無降雨，空白 表示缺值。風向資料888代表無風，
    999則代表儀器故障。
    自定義：空值轉為'empty'
'''
dataset_full = dataset.fillna('empty')
# dataset_full.info() # 檢查空值皆已填滿
data_temp = dataset_full.iloc[:, 3:] 

for col_name in data_temp.columns:
    series = data_temp[col_name]
    series_pick = series[series.str.contains('#|\*|x|A|888|999')]
    for data in series_pick.index:
        series_pick[data] = 'empty' 
    '''
    #   更新Series資料的方法2: 
    #       for data in series_pick
    #       series_pick.replace(to_replace = data, value = 'e', inplace=True)
    #   但若使用series.replace則會搜尋所有相同值並替換，估計效能比較差
    '''
    dataset_full[col_name].update(series_pick) # 將更新的series更新合併至完整dataset
# 檢查是否將所有錯誤資料都清洗成功
# print(dataset_full[dataset_full['00'].str.contains('#|\*|x|A|888|999')])
# 抽取資料中含empty的列
# print(dataset_full[dataset_full['00'].str.contains('empty')])
    
    #   補充：filter選出特定欄之特定規則的資料
    #       在df中[用df布林作filter]找出符合條件的列(欄都印出)，後綴再用loc選定特定範圍: 
    #       print(dataset_NoNull[dataset_NoNull['日期']=='2019/01/01'].loc[:, '日期'])
    
'''
缺失資料處理：平均值替換缺值
'''
from sklearn.preprocessing import Imputer
#Imputer : 專門處理數據缺失的類別

imputer = Imputer(missing_values = 'empty', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset_full[:, 3:])
dataset_full[:, 1:3] = imputer.transform(dataset_full[:, 1:3])
print(dataset_full[dataset_full['00'].str.contains('empty')])