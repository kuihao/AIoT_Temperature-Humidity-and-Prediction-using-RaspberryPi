'''
題目：桃園平鎮 PM 2.5 預測
'''
import sys
import numpy as np 
import matplotlib.pyplot as plt #畫出圖型 
import pandas as pd #資料處理
pd.set_option("display.max_rows", 1000)    #設定最大能顯示1000rows
pd.set_option("display.max_columns", 1000) #設定最大能顯示1000columns
'''
[Load Train Data(匯入訓練資料)]
TrainData_2019_PingZhen.csv 的資料為 12 個月中，每個月取 20 天，每天 24 小時的資料(每小時資料有 18 個 features)
'''
data = pd.read_csv('LinearRegression\TrainingData_2019_Pinzhen\TrainData_2019_PingZhen.csv', encoding = 'utf-8')
'''
[Preprocessing(資料預處理)]
Column Name 'RAINFALL' 的 NR(No Rain) 全部改成 0 ......[已完成]
'''
data = data.iloc[:, 3:]
# data[data == 'NR'] = 0
raw_data = data.to_numpy()
'''
[Extract Features (1)] 以 12 個月的 15 個Features (共180項) 為列，小時為欄
將原始數據 3600 (筆) * 24 (hours) 依照月份重新分組成 12 個 15 (features) * 480 (hours) 的資料
'''
month_data = {}
for month in range(12):
    sample = np.empty([15, 480])
    for day in range(20):
        # sample是容器，15 features為列，一日24小時為一欄，要裝20天；row_data每次選15筆feature(含24欄)，一個月有20天的資料*目前第幾個月+已算到第幾天
        sample[:, day * 24 : (day + 1) * 24] = raw_data[15 * (20 * month + day) : 15 * (20 * month + day + 1), :]
    month_data[month] = sample
'''
[Extract Features (2)]抽出訓練資料
1. 每個月有 15 (features) * 480 (hours) 的數據。
2. 從中「產生訓練資料」的規則是：
   每 10 個小時的資料組成一筆訓練資料 (意思是午夜12點至早上10點為第一筆，凌晨1點至早上11點為第二筆，依此類推)，
   其中，前 9 小時的資料作為 Input Datas，最後 1 小時的 PM 2.5 數值作為 Output Data，
   因此，Input Datas 為 15 (features) * 9 (hours)，Output Data 為 1 (features) * 1 (hours)。
3. 每個月扣除最後9小時資料不拿來預測(為何不是一年為計算？)，則可預測資料量是 480 - 9 = 471 個小時的 PM 2.5，
   也就是每個月可從中提取 471 筆訓練資料，換算成整年就是 12 * 471 = 8,892 筆訓練資料。
'''
# x 為訓練資料的輸入，列是訓練資料個數(一個月471筆，乘上12個月)，欄是9個小時的Features循環
x = np.empty([12 * 471, 15 * 9], dtype = float)
# y 為訓練資料的輸出，列是訓練資料個數(一個月471筆，乘上12個月)，欄是下一小時的PM 2.5
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            # 由於沒有隔年年初的資料，因此年底剩餘的9個小時皆無法構成訓練資料
            if day == 19 and hour > 14:
                continue
            # 每一筆訓練資料是新增列(鉛直軸新增)
            # x 是輸入端，取 month_data [月][ 15個features全要, 遞增的9個小時(欄的總長度單位是20天換算小時計)]
            # 經過reshape，numphy可以廣播至x矩陣，且同類的feature會接續排列
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            # y 是輸入端，取 month_data [月][ PM 2.5的項, 第10個小時(index起於0，故第10小時以9起)]
            y[month * 471 + day * 24 + hour, 0] = month_data[month][7, day * 24 + hour + 9] #value
# print(x)
# print(y)
'''
[Normalize (1)]
標準化：測量一組數值的離散程度使用 mean(x) 平均值、Standard Deviation 標準差
正歸化：x_normalized = x - min(x) / max(x) - min(x)
'''
mean_x = np.mean(x, axis = 0) #15 * 9: axis = 0 是取欄方向12 * 471的資料算 Mean，因此會有 15 * 9 個值 
std_x = np.std(x, axis = 0) #15 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #15 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
x
# print(x)
'''
[Split Training Data Into "train_set" and "validation_set"]
train_set用來訓練，validation_set不會被放入訓練、只是用來驗證
'''
import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
#print(x_train_set)
#print(y_train_set)
#print(x_validation)
#print(y_validation)
#print(len(x_train_set))
#print(len(y_train_set))
#print(len(x_validation))
#print(len(y_validation))
'''
[Training]
1. 造Linear Model: weight, bias
2. 做一個Loss Function來選Function
3. Loss Function搭配Gradient descent，計算weight, bias對Function的微分來找到最好的參數值

下面的 code 採用 Root Mean Square Error
因為常數項的存在，所以 dimension (dim) 需要多加一欄；
eps 項是避免 adagrad 的分母為 0 而加的極小數值。
每一個 dimension (dim) 會對應到各自的 gradient, weight (w)，
透過一次次的 iteration (iter_time) 學習。

# -----以下筆記----- #

# 假設 Function Model
假設預測值為 Y_p，輸入的 9小時 * 15個 Features 為 X_i，
每個時空的 Feature 都獨立，故 i 為 9 * 15 = 135
Y_p = bias + w1 * X1 + w2 * X2 + ... + w15 * X15
    = bias + Summation_i^135(w_i * X_i)

# Loss Function (損失函數) 的製作：可用任何合理的方式評斷Function預測效果的優劣
  課堂教導的方法是用類似統計學的 Variance (變異數) 做評估
  * Variance 變異數 就是每一個觀察值與平均數的的距離平方和的平均
  * 此將 Variance 的平均離均差特性視為「估計誤差」，輸入資料的個數就是 Training datas 的總數，
    特別之處在於這裡 Variance 的「均數」也就是指估計誤差的Label值是要隨著不同組 Training data 而
    變動的，並非傳統離均差有一個固定的均值。
  * 總結此 Loss Function 輸入 (w, b) 即 Function 的一組係數、Training datas 的 Input 及 Output，
    輸出的 Loss 值為「Label事實和預測結果之間的均方差」
  * 目標是調整輸入的參數 (w, b) ，找到夠小的 Loss 值，就是最佳 Function參數，
    調整參數這件事情不用手動調，也不是暴力法直接開，此題可以用線性代數解出 Close Form，
    也能使用 Gradient descent 來做。

# Gradient descent (梯度下降法)
  * 目的：為 Loss Function 找到比較好的參數 W，使 Loss(W) 值最小。
  * 限制：所定義的 Loss Function 必需可微分
  * 實作：隨機令 W 初值，計算 W 對 Loss Function 的微分值 d，
    若 d 是正數 (斜率為正) 下一輪就減少 W，反之 d 為負數，則在下一輪增加 W 的值。
  * 公式：賦予一個 weight 值 (稱為Learning rate，代號是 η /Eta/) 來調整下一輪該
    增加/減少 W 多少值 (以此提升調整參數的效率)。令 W' 為下一輪參數值，公式為 {W' = W - Eta * d}
'''
dim = 15 * 9 + 1 # 因為常數項的存在，所以 dimension (dim) 需要多加一欄
w = np.zeros([dim, 1]) 
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float) # 因為常數項的存在，所以 dimension (dim) 需要多加一欄
learning_rate = 100 # K:gradient descent
iter_time = 1000 # K:gradient descent
adagrad = np.zeros([dim, 1]) # K:就是每次更新的𝜂就是等於前一次的𝜂再除以𝜎^t，而 σ^t則代表的是第 t 次以前的所有梯度更新值之平方和開根號(root mean square)
eps = 0.0000000001
#print('iter_time & Loss\n')
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    #if(t%100==0):
    #    print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
w
'''
[Testing]
# 這240項資料應該是萃取完後隨機取出的，資料的萃取工作暫時跳過，目前用Excel隨機選取資料代替
載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，
使 test data 形成 240 個維度為 15 * 9 + 1 的資料。
'''
testdata = pd.read_csv('LinearRegression\TrainingData_2019_Pinzhen\EASY_TEST.csv', header = None, encoding = 'utf-8')
test_data = testdata.iloc[:, 1:10]
# test_data[test_data == 'NR'] = 0 # 已完成
test_data = test_data.to_numpy()
test_x = np.empty([240, 15*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[15 * i: 15* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
test_x
#print(test_x)
'''
[Prediction]
現在我們已定出Model(預測模型, Functuon set)、
找到自認完美Function的係數組合、選好了測試資料集，
那就能進行預測了～
'''
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
ans_y
#print(ans_y)
'''
[Save Prediction to CSV File]
'''
import csv
with open('PredictionResult.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
'''
[Function的改善]
用各式 Model 比較輸入 validation_set 預估結果的 Average Error 來選擇更好的 Model
切勿用 Testing data 來做篩選，否則會導致 Model 預測 Private Test data 的結果變差
'''