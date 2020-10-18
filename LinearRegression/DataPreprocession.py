import numpy as np 
import matplotlib.pyplot as plt #畫出圖型 
import pandas as pd #資料處理
#import sklearn
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
    自定義：空值轉為 '666'
'''
dataset_full = dataset.fillna('666')
# dataset_full.info() # 檢查空值皆已填滿
data_temp = dataset_full.iloc[:, 3:] 

for col_name in data_temp.columns:
    series = data_temp[col_name]
    series_pick = series[series.str.contains('#|\*|x|A|888|999')]
    for data in series_pick.index:
        series_pick[data] = '666'
    '''
    #   更新Series資料的方法2: 
    #       for data in series_pick
    #       series_pick.replace(to_replace = data, value = 'e', inplace=True)
    #   但若使用series.replace則會搜尋所有相同值並替換，估計效能比較差
    '''
    # 將更新的series更新合併至完整dataset
    dataset_full[col_name].update(series_pick) 
# 檢查是否將所有錯誤資料都清洗成功
# print(dataset_full[dataset_full['00'].str.contains('#|\*|x|A|888|999')])
# 抽取資料中含'666'的列
# print(dataset_full[dataset_full['00'].str.contains('666')])

    #   補充：filter選出特定欄之特定規則的資料
    #       在df中[用df布林作filter]找出符合條件的列(欄都印出)，後綴再用loc選定特定範圍: 
    #       print(dataset_NoNull[dataset_NoNull['日期']=='2019/01/01'].loc[:, '日期'])

# 將NR(No Rain)都改成0.0
dataset_full = dataset_full.replace(['NR'], [0.0])
# 檢查是否將所有'NR'都更正成0.0
# print(dataset_full[dataset_full['01'].str.contains('NR')])
'''
缺失資料處理：平均值替換缺值
'''
from sklearn.impute import SimpleImputer

# 從dataset_full切下數據資料部分進行清洗
temp_slice = dataset_full.iloc[:, 3:]
# print('前測有666\n', temp_slice[temp_slice['10'] == '666'].loc[:, '10'])
imputer_mean = SimpleImputer(missing_values=666, strategy='mean')
Fit_exec = imputer_mean.fit(temp_slice[:])
# print(Fit_exec) # 執行imputer後的回傳值
temp_slice = pd.DataFrame(imputer_mean.transform(temp_slice[:]))
temp_slice.set_axis(['00','01','02','03','04','05',
                    '06','07','08','09','10','11','12',
                    '13','14','15','16','17','18','19',
                    '20','21','22','23'], axis='columns', inplace=True)
# SimpleImputer直接將DataFrame的值從str轉成float
# print('後測無666\n', temp_slice[temp_slice['10'] >= 666].loc[:, '10'])
# 全部掃描確認資料清理完畢 # for col in temp_slice.columns:print(temp_slice[temp_slice[col] >= 666].loc[:, col])

#print('dataset_full\n', dataset_full)
#print('temp_slice\n', temp_slice)
for col_name in temp_slice.columns:
    dataset_full[col_name].update(temp_slice[col_name])
#print('Merged\n', dataset_full)
# 確認DataFrame正確合併 # for col in temp_slice.columns:print(dataset_full[dataset_full[col] >= 666].loc[:, col])
# 資料清洗完畢
def OutputCSV():   
    Path = 'LinearRegression\TrainingData_2019_Pinzhen\CleanData_2019_PingZhen.csv'
    dataset_full.to_csv( Path, index=False )
    print( '成功產出: ' + Path )
# 匯出查看 # OutputCSV()
