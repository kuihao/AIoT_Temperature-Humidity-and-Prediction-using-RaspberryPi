'''
題目：桃園平鎮 PM 2.5 預測
程式說明：雙井字「##」表示資料預處理之必要程式碼，只是為提升實驗效率而屏蔽
'''
import sys
import numpy as np 
import matplotlib.pyplot as plt #畫出圖型 
import pandas as pd #資料處理
import csv
import gc
# pd.set_option("display.max_rows", 1000)    #設定最大能顯示1000rows
# pd.set_option("display.max_columns", 1000) #設定最大能顯示1000columns

'''
[Load Train Data(匯入訓練資料)]
TrainData_2019_PingZhen.csv 的資料為 12 個月中，每個月取 20 天，每天 24 小時的資料(每小時資料有 15 個 features)
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
   也就是每個月可從中提取 471 筆訓練資料，換算成整年就是 12 * 471 = 5,652 筆訓練資料。
'''
# x 為訓練資料的輸入，列是訓練資料個數(一個月471筆，乘上12個月)，欄是9個小時的 Features 循環
x = np.empty([12 * 471, 15 * 9], dtype = float)
# y 為訓練資料的輸出，列是訓練資料個數(一個月471筆，乘上12個月)，欄是下一小時的PM 2.5
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            # 由於每個月的訓練集資料並無相連，因此最後一天剩餘的9個小時皆無法構成訓練資料
            if day == 19 and hour > 14:
                continue
            # 每一筆訓練資料是新增列(鉛直軸新增)
            # x 是輸入端，取 month_data [月][ 15 個 features 全要, 遞增的 9 個小時(欄的總長度單位是 20 天換算小時計)]
            # 經過reshape，numphy可以廣播至x矩陣，且同類的feature會接續排列
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector feature_weights:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            # y 是輸出端，取 month_data [月][ PM 2.5 的項, 第 10 個小時(index起於 0 ，故第 10 小時以 9 起)]
            y[month * 471 + day * 24 + hour, 0] = month_data[month][7, day * 24 + hour + 9] #value
# print(x)
# print(y)

'''
[Shuffle] 將資料打散隨機排序
目的：使後續切割 train_set、validation_set 資料可以較為平均
'''
box = np.concatenate((x, y), axis=1)
box = pd.DataFrame(box)
#print(box[0][0])
box = box.sample(frac=1).reset_index(drop=True) # n=len(box.index)
#print(box[0][0])
x = np.array(box.iloc[:,0:135])
y = np.array(box.iloc[:,135:])
#np.save(r'LinearRegression\TrainingData_2019_Pinzhen\x_tran_shuffle2.npy', x)
#np.save(r'LinearRegression\TrainingData_2019_Pinzhen\y_tran_shuffle2.npy', y)
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
Normalization_Method = 'Z-Score' # 'Z-Score'
x = np.load(r'LinearRegression\TrainingData_2019_Pinzhen\x_tran_shuffle2.npy')
y = np.load(r'LinearRegression\TrainingData_2019_Pinzhen\y_tran_shuffle2.npy')
# Z-Score
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
# Max-Min
## max_x = np.max(x)
## min_x = np.min(x)
## full_range_x = max_x - min_x
## if full_range_x < 0:
##   full_range_x *= (-1) 
## for i in range(len(x)): #12 * 471
##     for j in range(len(x[0])): #15 * 9 
##       if x[i][j] - min_x < 0:
##         x[i][j] = (x[i][j] - min_x) * (-1)
##       else:
##         x[i][j] = (x[i][j] - min_x) / full_range_x
# print(x)
'''
[Split Training Data]
train_set用來訓練，validation_set不會被放入"train_set" and "validation_set"訓練、只是用來驗證
'''
# 以下為正式全部的訓資
## x_train_set = x
## y_train_set = y
# 以下為 Validation
import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
# print(len(x_train_set)) # 4521
# print(len(y_train_set)) # 4521
# print(len(x_validation)) # 1131
# print(len(y_validation)) # 1131

import random
features = 15 * 9 + 1
item = len(x_train_set)
w = np.zeros([features, 1]) # features*1 
x_train_set = np.concatenate((np.ones([item, 1]), x_train_set), axis=1).astype(float) # 常數項水平合上 x_train_set 
x_validation = np.concatenate((np.ones([len(x_validation), 1]), x_validation), axis=1).astype(float)
# 以下為調整 Gradient 的重要參數 
SavingDialog = True #False
Model = 'one adjusting' 
Gradient_Method = 'AdaGrad'
learning_rate = 1
iter_time = 10000
#iter_time = math.ceil(150000/4521)
# AdaGrad 參數
adagrad_HSS = np.zeros([features, 1]) # [HSS] Historical Sum of Grdient Square 使每個參數的 Learning rate變得客製化
eps = 0.00000000001 # /epsilon/ 1e-11, 1e-8, 1e-6
# SGD
## concat_x_y = np.concatenate((x_train_set, y_train_set), axis=1)
## random.shuffle(concat_x_y)
## x_train_set = concat_x_y[0:, 0:features]
## y_train_set = concat_x_y[0:, features:]
## del(concat_x_y)
## gc.collect()
## stop_loop = False
# Momentum
momentum = np.zeros([features, 1])
Lambda = 0.9 # Attenuation coefficient，為歷史動量的衰退係數，值需小於 1，否則會 monotonic incressing
# RMSProp
ema = np.zeros([features, 1]) # EMA (exponential moving average，指數移動平均) 可能取名為 prop 比較好
Alpha = 0.85
# Adam
Beta_1 = 0.9
Beta_2 = 0.999
eps_adam = 0.0000001
momentum_adam = np.zeros([features, 1])
prop_adam = np.zeros([features, 1])
# 紀錄 Loss 值，繪圖用
loss_array = []
# 紀錄迭代次數
count = 0 
# 刪去部分項目
# 5小時
## L0 = np.zeros([item, 4])
## for k in range(15):
##   x_train_set[:,k*9+1:k*9+4+1] = L0
## L0 = np.zeros([len(x_validation), 4])
## for k in range(15):
##   x_validation[:,k*9+1:k*9+4+1] = L0
# 只有pm2.5+PM10
###L0 = np.zeros([item, 9])
###for k in range(6):
###  x_train_set[:,k*9+1:k*9+9+1] = L0
###for k in range(8,15):
###  x_train_set[:,k*9+1:k*9+9+1] = L0
###L0 = np.zeros([len(x_validation), 9])
###for k in range(6):
###  x_validation[:,k*9+1:k*9+9+1] = L0
###for k in range(8,15):
###  x_validation[:,k*9+1:k*9+9+1] = L0
# 多考慮 PM2.5+? 3 6 7
L0 = np.zeros([item, 9])
for k in range(3):
  x_train_set[:,k*9+1:k*9+9+1] = L0
for k in range(4,6):
  x_train_set[:,k*9+1:k*9+9+1] = L0
for k in range(8,15):
  x_train_set[:,k*9+1:k*9+9+1] = L0
#for k in range(12,15):
#  x_train_set[:,k*9+1:k*9+9+1] = L0

# 迭代
for t in range(iter_time):
  count += 1
  # Vanilla Gradient descenting
  ## gradient = (-2) * np.dot(x_train_set.transpose(), (y_train_set - np.dot(x_train_set, w))) # features*1
  ## w = w - learning_rate * gradient    

  # Momentum
  ## gradient = (-2) * np.dot(x_train_set.transpose(), (y_train_set - np.dot(x_train_set, w))) # features*1
  ## momentum = Lambda * momentum - learning_rate * gradient
  ## w = w + momentum

  # AdaGrad Method [Adaptive learning rate]
  gradient = (-2) * np.dot(x_train_set.transpose(), (y_train_set - np.dot(x_train_set, w))) # features*1
  adagrad_HSS += gradient ** 2
  w = w - learning_rate / np.sqrt(adagrad_HSS + eps) * gradient

  # RMSProp (Root-Mean-Square propagation) [Adaptive learning rate]
  # EMA 和 Momentum 有點類似，都是用迭代小數係數達到「歷史數據影響力指數遞減」
  # propagation 傳播，就是指隨著時間越長、傳播的越遠，gradient**2 的影響力要越小
  ## gradient = (-2) * np.dot(x_train_set.transpose(), (y_train_set - np.dot(x_train_set, w))) # features*1
  ## ema = Alpha * ema + (1-Alpha) * (gradient**2) # Tip:adagrad_HSS += gradient**2
  ## w = w - learning_rate / np.sqrt(ema) * gradient
  
  # Adam (Ada + momentum) SGDM + RMSProp 缺點：真的不太會收斂，最後一直震盪
  ## gradient = (-2) * np.dot(x_train_set.transpose(), (y_train_set - np.dot(x_train_set, w))) # features*1
  ## # 此時的 momentum 直接結合 EMA 的概念，兩者果然很像
  ## momentum_adam = Beta_1 * momentum_adam + (1-Beta_1) * gradient 
  ## # ema_adam 用來自動調整 Learning rate 所以要用除的
  ## prop_adam = Beta_2 * prop_adam + (1-Beta_2) * (gradient**2)
  ## # de-biasing: 由於 Beta_1 跟 Beta_2 的值都是 0.9 (接近1)
  ## # 導致迭代剛開始時，係數 (1-Beta) 值直接影響 gradient 的數值不夠大，
  ## # 因此要隨著時間除上 (1-Beta) 以維持當下 gradient 的影響力不會減少
  ## momentum_hat = momentum_adam/(1-(Beta_1)**count)
  ## prop_hat = prop_adam/(1-(Beta_2)**count)
  ## w = w - (learning_rate/np.sqrt(prop_hat)+eps_adam) * momentum_hat

  # 計算 Loss 值
  #loss_sse = np.sum(np.power(y_train_set - np.dot(x_train_set, w), 2)) # SSE (Sum of squared errors)
  #loss = np.sqrt(loss_sse/item) # RMSE (Root-mean-square error)
  Vali_Ave_Err = np.sqrt(np.sum((y_validation - np.dot(x_validation, w))**2)/len(y_validation))
  loss_sse = np.sum((y_validation - np.dot(x_validation, w))**2)
  loss = np.sqrt(loss_sse/len(y_validation))
  loss_array.append(loss)
  # 文字顯示 Loss 變化
  if (count%100==1) | (count==iter_time-1):
     print('Iter_time = ', count, "validation Loss = ", loss)
  #---------------#
  # SGD
  ## if not(stop_loop):
  ##   for n in range(item):
  ##     count += 1
  ##     if(count>150000):
  ##       count -= 1
  ##       stop_loop = True
  ##       break
  ##     x_n = x_train_set[n,:].reshape(1, features)
  ##     y_n = y_train_set[n].reshape(1, 1)
  ##     gradient = (-2) * np.dot(x_n.transpose(), (y_n - np.dot(x_n, w))) # features*1
  ##     w = w - learning_rate * gradient 
  ##     ## adagrad_HSS += gradient ** 2
  ##     ## w = w - learning_rate/np.sqrt(adagrad_HSS + eps) * gradient
  ##     loss_sse = np.sum(np.power(y_train_set - np.dot(x_train_set, w), 2)) # SSE (Sum of squared errors)
  ##     loss = np.sqrt(loss_sse/item) # RMSE (Root-mean-square error)
  ##     loss_array.append(loss)
  ##     if (not(count%10000)):
  ##        print('Iter_time = ', count, "Loss(error) = ", loss)
  ## else:
  ##   break
  # BGD (Batch GD)
# 將重要的函數權重值存檔
np.save(r'LinearRegression\TrainingData_2019_Pinzhen\weight.npy', w)
with open(r'LinearRegression\PredictionResult\Iter_loss.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    header = ['Iter', 'loss value']
    # print(header)
    csv_writer.writerow(header)
    for i in range(len(loss_array)):
      if(i%100==0)|(i==len(loss_array)-1):
        row = [str(i), loss_array[i]]
        csv_writer.writerow(row)
'''
[Training]
1. 創造Linear Model: weight, bias
2. 製作Loss Function來衡量Function的預測準度
3. 使用Gradient descent計算weight, bias對Loss Function的微分，以此方式有效率地調整參數，
   快速找到最好的參數值組合

範例Code說明：
   下面的 code 採用 Root Mean Square Error (均方根誤差)
   因為常數項的存在，所以 feature_weightsension (feature_weights) 需要多加一欄；
   eps 項是避免 adagrad 的分母為 0 而加的極小數值。
   每一個 feature_weightsension (feature_weights) 會對應到各自的 gradient, weight (w)，
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
  * Code 說明：
    # 解釋梯度 (gradient) 的運算：
    # 令 資料的數量為 item = 8,892  
    # 令 features 的數量為 feature_weights = 135
    # 令 訓練資料為 x，二維矩陣 (item*feature_weights)
    # 令 參數(權重值，內涵 Bias)為 w，二維矩陣 (feature_weights*1) 
    # 令 Ground truth 為 y，二維矩陣 (item*1)
    # 令 梯度 ( w 對 SSE (Sum Square Error, 誤差平方和) 趨近零的微分計算結果) 為 GD = (np.dot(x, w) - y)，二維矩陣(item*1)
    # 令 轉置矩陣運算子為 ^T
    # 梯度運算的轉置簡化推導：2 * (矩陣GD^T * 矩陣x)^T，得到一個 feature_weights*1 的結果，依照轉置運算化簡變成 2*(矩陣x^T * 矩陣GD)
'''
## # Real Train wirh whole training set
## import random
## feature_weights = 15 * 9 + 1 # 因為有常數項參數 bias，所以 feature_weights 需要多加一欄
## item = len(x)
## w = np.zeros([feature_weights, 1]) # [feature_weights, 1]和(feature_weights, 1)一樣意思，就是存成 feature_weights * 1 的二維零矩陣
## x = np.concatenate((np.ones([item, 1]), x), axis = 1).astype(float) # 因為常數項的存在，所以 feature_weightsension (feature_weights) 需要多加一欄
## # 以下為調整 Gradient 的重要參數 
# SavingDialog = True
## Model = 'One' #One_5hour-F12345F10F12F13
## Gradient_Method = 'Vanilla'
## learning_rate = 0.01 # gradient descent 的常係數 𝜂 /Eta/
## iter_time = 15000 # gradient descent 的迭代次數
## # AdaGrad 參數
## adagrad_HSS = np.zeros([feature_weights, 1]) # 代號 𝜎^t 意思是第 t 次迭代以前的所有梯度更新值之平方和 [HSS] Historical Sum of Grdient Square
## eps = 0.0000000001 # /epsilon/ 用途是避免在 Local minima (微分為零時) 停下來
## # RMSProp
## prop = np.zeros([feature_weights, 1]) # EMA (exponential moving average，指數移動平均) 可能取名為 prop 比較好
## Alpha = 0.85
## # SGD
## ## concat_x_y = np.concatenate((x, y), axis=1)
## ## random.shuffle(concat_x_y)
## ## x = concat_x_y[0:, 0:feature_weights]
## ## y = concat_x_y[0:, feature_weights:]
## ## del(concat_x_y)
## ## gc.collect()
## # 紀錄 Loss 值，繪圖用
## loss_array = []
## # 紀錄迭代次數
## count = 0 
## 
## # [改良：篩去部分 Feature]
## ## L0 = np.zeros([item, 18])
## ## x[:, 109:127] = L0 # 'WIND_DIREC', 'WIND_SPEED
## ## L0 = np.zeros([item, 45])
## ## x[:, 9:54] = L0 # 'CO', 'NO', 'NO2', 'NOx', 'O3'
## ## L0 = np.zeros([item, 9])
## ## x[:, 90:99] = L0 # SO2
## ## L0 = np.zeros([item, 4])
## ## for k in range(15):
## ##   x[:,k*9+1:k*9+4+1] = L0
## # Full去掉
## ## L0 = np.zeros([item, 9])
## ## for k in range(6):
## ##   x[:,k*9+1:k*9+9+1] = L0
## ## for k in range(8,15):
## ##   x[:,k*9+1:k*9+9+1] = L0
## 
## for t in range(iter_time):
##   count += 1
##   # Vanilla
##   gradient = (-2) * np.dot(x.transpose(), (y - np.dot(x, w))) #feature_weights*1
##   w = w - learning_rate * gradient    
## 
##   # 使用 AdaGrad
##   ## gradient = (-2) * np.dot(x.transpose(), (y - np.dot(x, w))) #feature_weights*1
##   ## adagrad_HSS += gradient ** 2
##   ## w = w - learning_rate / np.sqrt(adagrad_HSS + eps) * gradient
## 
##   # RMSProp (Root-Mean-Square propagation) [Adaptive learning rate]
##   # EMA 和 Momentum 有點類似，都是用迭代小數係數達到「歷史數據影響力指數遞減」
##   # propagation 傳播，就是指隨著時間越長、傳播的越遠，gradient**2 的影響力要越小
##   ## gradient = (-2) * np.dot(x.transpose(), (y - np.dot(x, w))) # features*1
##   ## prop = Alpha * prop +  (1-Alpha) * (gradient**2) # Tip:adagrad_HSS += gradient**2
##   ## w = w - (learning_rate*gradient) / np.sqrt(prop+eps)     
##     
##   # 計算 Loss 值
##   loss_sse = np.sum(np.power(y - np.dot(x, w), 2)) # Loss Function: SSE (Sum of squared errors)
##   loss = np.sqrt(loss_sse/item) # rmse (Root-mean-square deviation) 
##   loss_array.append(loss) # 紀錄 loss 值，繪圖用
##   # 文字顯示 Loss 變化
##   if (not(t%1000)) | (t==iter_time-1):
##     print('Iter_time = ', t, "Loss(error) = ", loss)
##   # SGD
##   ## for n in range(item):
##   ##   count += 1
##   ##   x_n = x[n,:].reshape(1, feature_weights)
##   ##   y_n = y[n].reshape(1, 1)
##   ##   gradient = (-2) * np.dot(x_n.transpose(), (y_n - np.dot(x_n, w))) # features*1
##   ##   w = w - learning_rate * gradient  
##   ##   #adagrad_HSS += gradient ** 2
##   ##   #w = w - learning_rate/np.sqrt(adagrad_HSS + eps) * gradient
##   ##   loss_sse = np.sum(np.power(y - np.dot(x, w), 2)) # SSE (Sum of squared errors)
##   ##   loss = np.sqrt(loss_sse/item) # RMSE (Root-mean-square error)
##   ##   loss_array.append(loss)
##   ##   if (not(count%10000)):
##   ##      print('Iter_time = ', count, "Loss(error) = ", loss)
## 
## # 將重要的函數權重值存檔
## np.save(r'LinearRegression\TrainingData_2019_Pinzhen\weight.npy', w)
'''
[Testing]
# 測試資料也要經過標準化處理才能輸入 Function
載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，
使 test data 形成 240 個維度為 15 * 9 + 1 的資料。
'''
# [load data]
## test_x = np.load(r'LinearRegression\TestingData\x_test_shuffle.npy')
## test_y = np.load(r'LinearRegression\TestingData\y_test_shuffle.npy')
test_x = np.load(r'LinearRegression\TestingData\x_test_shuffle2_publuc.npy')
test_y = np.load(r'LinearRegression\TestingData\y_test_shuffle2_publuc.npy')

# [Scaling]
# Z-Score
# std_x, mean_x 要以訓練資料的數值進行標準化，才能將測試資料轉成相同的比例尺進行運算 
for i in range(len(test_x)): # 二維陣列的長度是算最外框裡面內涵的一維陣列個數，因此是 240，即輸入的測試資料項目個數
    for j in range(len(test_x[0])): # 135 個 Features
        if std_x[j] != 0: # 根據除法定裡，分母不得為零
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
# min-max
## for i in range(len(test_x)): #12 * 471
##     for j in range(len(test_x[0])): #15 * 9 
##       if test_x[i][j] - min_x < 0:
##         test_x[i][j] = (test_x[i][j] - min_x) * (-1)
##       else:
##         test_x[i][j] = (test_x[i][j] - min_x) / full_range_x
# [增加常數系數]
test_x = np.concatenate((np.ones([len(test_x), 1]), test_x), axis = 1).astype(float)
test_x
# print(test_x)
# 遮罩一些 Features 9:54
## L0 = np.zeros([len(test_x), 18])
## test_x[:,109:127] = L0
## L0 = np.zeros([len(test_x), 45])
## test_x[:,9:54] = L0
## L0 = np.zeros([len(test_x), 9])
## test_x[:,90:99] = L0
## L0 = np.zeros([len(test_x), 4])
## for k in range(15):
##   test_x[:,k*9+1:k*9+4+1] = L0
# 只剩 PM2.5+PM10
###L0 = np.zeros([len(test_x), 9])
###for k in range(6):
###  test_x[:,k*9+1:k*9+9+1] = L0
###for k in range(8,15):
###  test_x[:,k*9+1:k*9+9+1] = L0

# 只剩 PM2.5 + ???
L0 = np.zeros([len(test_x), 9])
for k in range(3):
  test_x[:,k*9+1:k*9+9+1] = L0
for k in range(4,6):
  test_x[:,k*9+1:k*9+9+1] = L0
for k in range(8,15):
  test_x[:,k*9+1:k*9+9+1] = L0
#for k in range(12,15):
#  test_x[:,k*9+1:k*9+9+1] = L0
# best 5.259926965
'''
[Prediction]
現在我們已定出 Model (預測模型, Functuon set)、
找到自認完美 Function 的係數組合、選好了測試資料集，
那就能進行預測了~
'''
w = np.load(r'LinearRegression\TrainingData_2019_Pinzhen\weight.npy')
ans_y = np.dot(test_x, w) # [item*feature_weights] * [feature_weights*1] = [item*1] 矩陣相乘的奧祕就是：
                          # 被左乘矩陣，則列運算；被右乘一個矩陣，則行運算。
                          # 矩陣相乘是相對的概念，單看你的主體是誰。
                          # 用矩陣相乘可表示方程式的「係數與未知數相乘再相加」，即線性組合
ans_y
#print(ans_y)
'''
[Save prediction result to CSV file]
'''
with open(r'LinearRegression\PredictionResult\PredictionResult.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['item_id', 'value']
    # print(header)
    csv_writer.writerow(header)
    for i in range(len(ans_y)):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        # print(row)
'''
[紀錄實驗數據] ---- Function的改善 ----
用各式 Model 比較輸入 validation_set 預估結果的 Average Error 來選擇更好的 Model
切勿用 Testing data 來做篩選，否則會導致 Model 預測 Private Test data 的結果變差
[Validation/ Train] Line change: 300, 301, 328 (326, 323)
'''
# No AdaGrad, all testing data: 5.158543826472928 iter:1000 ETA:0.000001
import datetime
# 顯示實驗數據
Test_Ave_Err = np.sqrt(np.sum((test_y - ans_y)**2)/len(test_y))
# Validation 檢測時設此值
Vali_Ave_Err = np.sqrt(np.sum((y_validation - np.dot(x_validation, w))**2)/len(y_validation))
# 全部訓練集的時候設此值
## y_validation = '' # Validation 檢測時此行要屏蔽
## Vali_Ave_Err = 0 # Validation 檢測時此行要屏蔽

# 計算訓練資料的錯誤率`，公式： 真實值-預測值/真實值 * 100% 
Train_error_rate = round((np.sqrt(loss_sse)/np.sqrt(np.sum(y**2)))*100, 2)
# print('Train Ave_err: ', loss)
# print('Validation Ave_err: ', Vali_Ave_Err)
# print('Testing Ave_err: ', Test_Ave_Err)
time = (datetime.datetime.now()).strftime("%y/%m/%d %H:%M:%S")
row =   [time, Model, item, 
        len(y_validation), len(test_y), 
        Normalization_Method, learning_rate, 
        count, Gradient_Method, 
        str(Train_error_rate)+'%', loss, 
        Vali_Ave_Err, Test_Ave_Err,]
head =  ['Date', 'Model','Train set size', 
         'Validation set size','Test set size', 
         'Normalization', 'Learning rate', 
         'Iteration time', 'Gradient', 
         'Train error rate', 'Train Ave_err', 
         'Validation Ave_err', 'Testing Ave_err',]
Record = pd.DataFrame({time:row[1:]}, index=head[1:])
print('[Record]:\n', Record)

# 存檔
def SaveRecord(a_head, a_row):
    with open(r'LinearRegression\PredictionResult\ExperimentRecord.csv', mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = a_head
        #writer.writerow(header) # 先用 mode w 寫 header，之後用 mode a 新增數據
        writer.writerow(a_row)
        print('Save record successfully!')
# SaveRecord(head, row)
'''
[Loss 變化分析]
'''
# 繪製「函數的 Loss 值時變圖」
plot_len = len(loss_array)
plt.axis([0, plot_len, 0, max(loss_array)+1]) # 二維座標圖 x, y 軸的顯示範圍
x_pos = np.linspace(0, plot_len, plot_len) # 依照迭代次數產生 x 軸座標
y_pos = loss_array # y 軸是 Loss value
plt.plot(x_pos, y_pos, '-', c='blue', markersize=4) # 形成每個點，繪出函數圖形
plt.show() # 顯示 plot 視窗

'''
[實驗紀錄是否存檔？]
'''
while SavingDialog:
  save = False
  c = input('Save record? [y/n]')
  if len(c) == 1:
    if(c == 'y')|(c == 'Y'):
      save = True
      SaveRecord(head, row)
      break
    elif (c == 'n')|(c == 'N'):
      print('unsave.')
      break
    else:
      print('plz enter again, only one character \'y\' or \'n\'.')
      continue
  else:
    print('plz enter again, only one character \'y\' or \'n\'.')
    continue