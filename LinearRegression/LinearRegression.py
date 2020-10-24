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
data = pd.read_csv(r'LinearRegression\TrainingData_2019_Pinzhen\TrainData_2019_PingZhen.csv', encoding = 'utf-8')
'''
[Preprocessing(資料預處理)]
Column Name 'RAINFALL' 的 NR(No Rain) 全部改成 0 ......[已完成]
'''
data = data.iloc[:, 3:]
# data[data == 'NR'] = 0
raw_data = data.to_numpy()
'''
[Extract Features (1)] 以 12 個月的 15 個 Features (共180項) 為列，小時為欄
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
            # y 是輸出端，取 month_data [月][ PM 2.5的項, 第10個小時(index起於0，故第10小時以9起)]
            y[month * 471 + day * 24 + hour, 0] = month_data[month][7, day * 24 + hour + 9] #value
# print(x)
# print(y)

'''
shuffle 將資料打散隨機排序
目的：使後續切割 train_set、validation_set 資料可以較為平均
'''
# box = np.concatenate((x, y), axis=1)
# box = pd.DataFrame(box)
# #print(box[0][0])
# box = box.sample(frac=1).reset_index(drop=True) # n=len(box.index)
# #print(box[0][0])
# x = np.array(box.iloc[:,0:135])
# y = np.array(box.iloc[:,135:])
# np.save(r'LinearRegression\TrainingData_2019_Pinzhen\x_tran_shuffle.npy', x)
# np.save(r'LinearRegression\TrainingData_2019_Pinzhen\y_tran_shuffle.npy', y)
'''
[Normalize (1)]
正歸化：當資料不是數值，或是雖為數值卻是類別資料(屬性資料、名義尺度)，則需經過編碼處理
       (本專題因為不用作敘述、推論統計分析，因此不需要)
標準化：測量一組數值的離散程度使用 mean(x) 平均值、Standard Deviation 標準差
    方法1 Max-Min 常見的資料標準化方法，簡單來說，將原始資料的最大、最小值mapping至區間[0,1]
          公式：x_normalized = x - min(x) / max(x) - min(x)
    方法2 採用Z-Score (Z分數) 利用原始資料的均值（mean）和標準差（standard deviation）進行
          資料的標準化，適用於資料的最大值和最小值未知的情況，或有超出取值範圍的離群資料的情況。
          公式：新資料 =（原始資料-均值）/ 標準差
'''
x = np.load(r'LinearRegression\TrainingData_2019_Pinzhen\x_tran_shuffle.npy')
y = np.load(r'LinearRegression\TrainingData_2019_Pinzhen\y_tran_shuffle.npy')
mean_x = np.mean(x, axis = 0) #15 * 9: axis = 0 是取鉛直方向，對12 * 471的資料算 Mean，因此會有 15 * 9 個Mean值 
std_x = np.std(x, axis = 0) #15 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #15 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
            # 目前是將 15 * 9 個 Features都視為獨立不同種類的 Feature
            # z-score (又稱Standard Score) 的標準化方法就是將個別 Feature 都從目前的資料集裡面，計算自己這種 Feature 的
            # 平均數及標準差，公式「離均差與標準差的比值」就是每組資料與平均數的標準化後「相對距離」
            # 依照中央極限定理，世界上所有資料的分布99.9936%都會包在四個標準差之內，因此
            # z-score 公式可以達到不同種類資料統一標準化呈現的效果
x
# print(x)
'''
[Split Training Data]
train_set用來訓練，validation_set不會被放入"train_set" and "validation_set"訓練、只是用來驗證
'''
import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
# print(len(x_train_set))
# print(len(y_train_set))
# print(len(x_validation))
# print(len(y_validation))

features = 15 * 9 + 1
item = len(x_train_set)
w = np.zeros([features, 1]) # features*1 
x_train_set = np.concatenate((np.ones([item, 1]), x_train_set), axis=1).astype(float) # 常數項水平合上 x_train_set 
x_validation = np.concatenate((np.ones([len(x_validation), 1]), x_validation), axis=1).astype(float)
learning_rate = 0.000001
iter_time = 10000
adagrad = np.zeros([features, 1])
eps = 0.0000000001 # /epsilon/
loss_array = []
for t in range(iter_time):
    gradient = (-2) * np.dot(x_train_set.transpose(), (y_train_set - np.dot(x_train_set, w))) # features*1
    # adagrad += gradient ** 2
    # w = w - learning_rate / np.sqrt(adagrad + eps) * gradient    
    w = w - learning_rate * gradient    
    loss = np.sqrt(np.sum(np.power(y_train_set - np.dot(x_train_set, w), 2))/item) # rmse (root-mean-square error)
    if loss > 100:
        loss_array.append(20) # 數值天花板，繪圖用
    else:
        loss_array.append(loss)
    if (not(t%1000)) | (t==iter_time-1):
        print('Iter_time = ', t, "Loss(error) = ", loss)
print('Training error rate: '+str(round(loss/np.mean(y_train_set)*100, 2))+'%')
'''
[Training]
1. 創造Linear Model: weight, bias
2. 製作Loss Function來衡量Function的預測準度
3. 使用Gradient descent計算weight, bias對Loss Function的微分，以此方式有效率地調整參數，
   快速找到最好的參數值組合

範例Code說明：
   下面的 code 採用 Root Mean Square Error (均方根誤差)
   因為常數項的存在，所以 dimension (dim) 需要多加一欄；
   eps 項是避免 adagrad 的分母為 0 而加的極小數值。
   每一個 dimension (dim) 會對應到各自的 gradient, weight (w)，
   透過迭代 (iter_time, iteration) 調整參數。

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
  * 實作：隨機令 W 初值，計算 W 對 Loss Function 的微分值 gd，
    若 gd 是正數 (斜率為正) 下一輪就減少 W，反之 gd 為負數，則在下一輪增加 W 的值。
  * 公式：賦予一個 weight 值 (稱為Learning rate，代號是 η /Eta/) 來調整下一輪該
    增加/減少 W 多少值 (以此提升調整參數的效率)。令 W' 為下一輪參數值，公式為 {W' = W - Eta * gd}
'''
'''
# 總 Train
dim = 15 * 9 + 1 # 因為有常數項參數 bias，所以 dimension (dim) 需要多加一欄
w = np.zeros([dim, 1]) # [dim, 1]和(dim, 1)一樣意思，就是存成 dim * 1 的二維零矩陣
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float) # 因為常數項的存在，所以 dimension (dim) 需要多加一欄
learning_rate = 0.000001 #ada:100 # 就是/Eta/ gradient descent的常係數
iter_time = 1000 #1000 # K:gradient descent的迭代次數
adagrad = np.zeros([dim, 1]) # ??? K:就是每次更新的𝜂就是等於前一次的𝜂再除以𝜎^t，而 σ^t則代表的是第 t 次以前的所有梯度更新值之 root mean square (平方和開根號)
eps = 0.0000000001 # /epsilon/
loss_array = []
# print('iter_time & Loss\n')
for t in range(iter_time):
    # 解釋梯度 (gradient) 的運算：
    #   令 資料的數量為 item = 8,892  
    #   令 features 的數量為 dim = 135
    #   令 訓練資料為 x，二維矩陣 (item*dim)
    #   令 參數(權重值，內涵 Bias)為 w，二維矩陣 (dim*1) 
    #   令 Ground truth 為 y，二維矩陣 (item*1)
    #   令 梯度 ( w 對 SSE (Sum Square Error, 誤差平方和) 趨近零的微分計算結果) 為 D = (np.dot(x, w) - y)，二維矩陣(item*1)
    #   令 轉置矩陣運算子為 ^T
    #   梯度運算的轉置簡化推導：2 * (矩陣D^T * 矩陣x)^T，得到一個 dim*1 的結果，依照轉置運算化簡變成 2*(矩陣x^T * 矩陣D)
    gradient = 2 * np.dot(x.transpose(), (np.dot(x, w)-y)) #dim*1
    # if(t==iter_time-1):
    #     print('Gradient:\n', pd.DataFrame(gradient))
    
    # 使用 AdaGrad
    #adagrad += gradient ** 2
    #w = w - learning_rate / np.sqrt(adagrad + eps) * gradient    
    w = w - learning_rate * gradient    
    # 目前的 Loss 數值
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12) #rmse(root-mean-square deviation) #Loss Function #矩陣相乘: [(12*471)*(1+15*9)] dot [(1+15*9)*1]=[(12*471)*(1)]
    #loss = np.sum(np.power(np.dot(x, w) - y, 2))/(471*12)
    loss_array.append(loss)
    # if(t%100==0)|(t==iter_time-1):
    #     print('Iter_time = '+ str(t), "Loss(error) = " + str(loss))
np.save(r'LinearRegression\TrainingData_2019_Pinzhen\weight.npy', w)
w
# print('Function Parameter:\n', pd.DataFrame(w))
'''
'''
[Testing]
# 這240項資料應該是萃取完後隨機取出的，資料的萃取工作暫時跳過，目前用Excel隨機選取資料代替
# 測試資料也要經過標準化處理才能輸入 Function
載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，
使 test data 形成 240 個維度為 15 * 9 + 1 的資料。
'''
testdata = pd.read_csv(r'LinearRegression\TestingData\EASY_TEST.csv', header = None, encoding = 'utf-8')

test_data = testdata.iloc[:, 1:10]
# test_data[test_data == 'NR'] = 0 # 已完成
test_data = test_data.to_numpy()
test_x = np.empty([240, 15*9], dtype = float)
# std_x, mean_x 要以訓練資料的數值進行標準化，才能將測試資料轉成相同的比例尺進行運算 
for i in range(240):
    test_x[i, :] = test_data[15 * i: 15* (i + 1), :].reshape(1, -1) # test_x 是標準二維陣列 240*135
for i in range(len(test_x)): # 二維陣列的長度是算最外框裡面內涵的一維陣列個數，因此是 240，即輸入的測試資料項目個數
    for j in range(len(test_x[0])): # 135 個 Features
        if std_x[j] != 0: # 根據除法定裡，分母不得為零
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
test_x
#print(test_x)

# Ground truth of testing marks as test_y
test_GroundTruth = testdata.iloc[:, 10:]
test_GroundTruth = test_GroundTruth.to_numpy()
test_y = np.empty([240, 1], dtype = float)
for i in range(240):
    test_y[i][0] = test_GroundTruth[15*i+7, 0]
#print(pd.DataFrame(test_y))
'''
[Prediction]
現在我們已定出 Model (預測模型, Functuon set)、
找到自認完美 Function 的係數組合、選好了測試資料集，
那就能進行預測了～
'''
w = np.load(r'LinearRegression\TrainingData_2019_Pinzhen\weight.npy')
ans_y = np.dot(test_x, w) # [item*dim] * [dim*1] = [item*1] 矩陣相乘的奧祕就是：
                          # 被左乘矩陣，則列運算；被右乘一個矩陣，則行運算。
                          # 矩陣相乘是相對的概念，單看你的主體是誰。
                          # 用矩陣相乘可表示方程式的「係數與未知數相乘再相加」，即線性組合
ans_y
#print(ans_y)
'''
[Save Prediction to CSV File]
'''
import csv
with open(r'LinearRegression\PredictionResult\PredictionResult.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['item_id', 'value']
    # print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        # print(row)
'''
[Function的改善]
用各式 Model 比較輸入 validation_set 預估結果的 Average Error 來選擇更好的 Model
切勿用 Testing data 來做篩選，否則會導致 Model 預測 Private Test data 的結果變差
'''

'''
Loss 變化分析
'''
plt.axis([0, iter_time, 0, max(loss_array)+1])
x_pos = np.linspace(0, iter_time, iter_time)
y_pos = loss_array
# print(pd.DataFrame(y_pos))
plt.plot(x_pos, y_pos, '-', c='blue', markersize=4)
print('Train Ave_err: ', loss_array[-1]) # No AdaGrad, all testing data: 5.158543826472928 iter:1000 ETA:0.000001
print('Validation Ave_err: ', np.sqrt(np.sum((y_validation - np.dot(x_validation, w))**2)/len(y_validation)))
print('Testing Ave_err: ', np.sqrt(np.sum((test_y - np.dot(test_x, w))**2)/len(test_y)))
#plt.show()
