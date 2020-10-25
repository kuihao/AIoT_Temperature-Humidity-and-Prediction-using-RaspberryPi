import gc
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 1000)    #設定最大能顯示1000rows
pd.set_option("display.max_columns", 1000) #設定最大能顯示1000columns
'''
# [load data]
# 將原始資料整理成 row_testdata:
#   列: id，15 個監測項目 * 每月剩餘天數 (不固定)
#   欄: 每小時監測一次，共 24 (Hours)
'''
read_testdata = pd.read_csv(r'LinearRegression\TestingData\TestData2_2019_PingZhen.csv', encoding = 'utf-8')
id_col = read_testdata.iloc[:, 0]
row_testdata = read_testdata.iloc[:, 4:]
row_testdata = pd.concat([id_col, row_testdata], axis=1, ignore_index=True)
del(read_testdata, id_col)
gc.collect()
# print(row_testdata)

'''
# [Extract Features]
# 萃取出測試資料並標籤 (Label) 對應的真實解答 (Ground Truth)
#   萃取方法是連續 9 小時的監測數據個別視為一項 Feature，共有 9*15=135 項 Features
#   由於測試集是取每月前 20 天以後的剩餘天數製成，因此缺乏次月月初資料，
#   導致每個月月底的 9 個小時無法萃取成測試資料，並且需要依照月份分別萃取
# 用月份前後半個月切割成訓練集與測試集的好處是抽取的資料保證能平均取得四季、每個月的資料。

# 萬一儲存空間爆了，需要使用逐列存取
# 研究一
# def RowByRow(OneMounth, d, OneDay):
#     for i in range(15):
#         OneMounth[i:(i+1), 24*d:24*(d+1)] = np.array(OneDay.iloc[i:(i+1), :].values.reshape(1, 24))
#     return OneMounth
# 研究二
#    x = 7
#    A = np.zeros([15,24])
#    # print('2'+str(x+1)+'\n' ,row_testdata[id_filter].iloc[x*15:(x+1)*15, 1:].to_numpy())
#    #print(A)
#    for i in range(12):
#        #print(i, '||', (row_testdata[id_filter].iloc[(x*15)+i:(x*15)+(i+1), 1:].to_numpy()).reshape(1,24))
#        A[ i:(i+1),:] = (row_testdata[id_filter].iloc[(x*15)+i:(x*15)+(i+1), 1:].to_numpy()).reshape(1,24)
#    print(A)
'''
months_testdata = {}
list_remain_day = np.zeros(12, int)
for month in range(12):
    id_filter = row_testdata[0]==('id_'+str(month+1)) # a boolean of pd.Series
    remain_days = int(len(row_testdata[id_filter].iloc[:, 0:1])/15) # an integer
    list_remain_day[month]=remain_days
    one_month_15Fx24nH = np.empty([15, remain_days*24], dtype=float) # np.array 15*remain-hours
    for day in range(remain_days):
        # sequence: 0 to 15 => x*15 to x*15+15 => 15x to 15(x+1)
        one_day_15Fx24H = row_testdata[id_filter].iloc[day*15:(day+1)*15, 1:] # pd.DataFrame 15*24 
        one_month_15Fx24nH[0:, day*24:(day+1)*24] = one_day_15Fx24H.to_numpy()
        #one_month_15Fx24nH[:, day*24:(day+1)*24] = RowByRow(one_month_15Fx24nH, day, one_day_15Fx24H)
    months_testdata[month] = one_month_15Fx24nH
del(row_testdata)
gc.collect()
#print(months_testdata)
'''
[Extract Features (2)] 萃取測試資料
'''
items = 0 # 總筆數
for m in range(12):
    items += (list_remain_day[m]*24-9)
# print(items) # 總筆數為 2,892

test_x = np.empty([items, 15*9], dtype = float)
test_y = np.empty([items, 1], dtype = float)
id_item = 0
for month in range(12):
    remain_days = list_remain_day[month]
    for day in range(remain_days):
        for hour in range(24):
            if (day == (remain_days-1)) and (hour > 14):
                continue
            test_x[id_item,:] = months_testdata[month][:,(day*24+hour):(day*24+(hour+9))].reshape(1, -1)
            test_y[id_item,0] = months_testdata[month][7, day*24+(hour+9)] # 第 10 小時的 PM2.5
            id_item += 1
del(id_item, months_testdata, list_remain_day)
gc.collect()
# print(test_x, '\n', pd.DataFrame(test_y))
'''
[Shuffle] 將資料打散隨機排序
'''
box = np.concatenate((test_x, test_y), axis=1)
box = pd.DataFrame(box)
# print(box[0][0])
box = box.sample(frac=1).reset_index(drop=True)
# print(box[0][0])
getItems = 240 # 隨機排序後抽取幾項資料
test_x_shuffle = np.array(box.iloc[0:getItems, 0:135])
test_y_shuffle = np.array(box.iloc[0:getItems, 135:])
np.save(r'LinearRegression\TestingData\x_test_shuffle.npy', test_x_shuffle)
np.save(r'LinearRegression\TestingData\y_test_shuffle.npy', test_y_shuffle)
#del(box, test_x, test_y)
#gc.collect()