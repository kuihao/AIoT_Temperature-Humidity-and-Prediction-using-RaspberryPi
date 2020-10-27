'''
# 重新設計預測模型 (預測桃園平鎮 2019 PM 2.5)
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
[Normalize (1)]
標準化：測量一組數值的離散程度使用 mean(x) 平均值、Standard Deviation 標準差
    方法1 【Max-Min】常見的資料標準化方法，簡單來說，將原始資料的最大、最小值mapping至區間[0,1]
          公式：x_normalized = x - min(x) / max(x) - min(x)
    方法2 【Z-Score (Z分數)】利用原始資料的均值（mean）和標準差（standard deviation）進行
          資料的標準化，適用於資料的最大值和最小值未知的情況，或有超出取值範圍的離群資料的情況。
          公式：新資料 =（原始資料-均值）/ 標準差
'''
Normalization_Method = 'Z-Score'
x = np.load(r'LinearRegression\TrainingData_2019_Pinzhen\x_tran_shuffle.npy')
y = np.load(r'LinearRegression\TrainingData_2019_Pinzhen\y_tran_shuffle.npy')
mean_x = np.mean(x, axis = 0) # 15 * 9: axis = 0 是取鉛直方向，對12 * 471的資料算 Mean，因此會有 15 * 9 個 Mean 值 
std_x = np.std(x, axis = 0) # 15 * 9 
for i in range(len(x)): # 12 * 471
    for j in range(len(x[0])): # 15 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
'''
[Split Training Data]
train_set and validation_set
'''
import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
'''
[Partial training and Validation test]
'''
import random
features = 15 * 9 + 1
item = len(x_train_set)
w1 = np.zeros([features, 1]) # features*1 
w2 = np.zeros([features, 1]) # features*1 
x_train_set = np.concatenate((np.ones([item, 1]), x_train_set), axis=1).astype(float) # 常數項水平合上 x_train_set 
x_validation = np.concatenate((np.ones([len(x_validation), 1]), x_validation), axis=1).astype(float)
# 以下為調整 Gradient 的重要參數 
Model = 'Square'
SavingDialog = True
Gradient_Method = 'MBGD+Momentum'
learning_rate = 0.0000001
iter_time = 150000
# AdaGrad 參數
adagrad_HSS = np.zeros([features, 1]) # [HSS] Historical Sum of Grdient Square 使每個參數的 Learning rate變得客製化
eps = 0.00000000001 # /epsilon/ 1e-11, 1e-8, 1e-6
# SGD
concat_x_y = np.concatenate((x_train_set, y_train_set), axis=1)
random.shuffle(concat_x_y)
x_train_set = concat_x_y[0:, 0:features]
y_train_set = concat_x_y[0:, features:]
del(concat_x_y)
gc.collect()
stop_loop = False
epoch = iter_time
#batch_size = item
#iter_time = math.ceil(iter_time/batch_size)
# MBGD
batch_size = int(0.1*item)
iter_time = math.ceil(iter_time/(item/batch_size))
# Momentum
momentum = np.zeros([features, 1])
momentum2 = np.zeros([features, 1])
Lambda = 0.9 # Attenuation coefficient，為歷史動量的衰退係數，值需小於 1，否則會 monotonic incressing
# RMSProp
ema = np.zeros([features, 1]) # EMA (exponential moving average，指數移動平均) 可能取名為 prop 比較好
Alpha = 0.85
# Adam
Beta_1 = 0.9
Beta_2 = 0.999
eps_adam = 0.00000001
momentum_adam = np.zeros([features, 1])
prop_adam = np.zeros([features, 1])
# 紀錄 Loss 值，繪圖用
loss_array = []
# 紀錄迭代次數
count = 0 
for t in range(iter_time):
  ## count += 1
  # Vanilla Gradient descenting
  ## gradient = (-2) * np.dot(x_train_set.transpose(), (y_train_set - np.dot(x_train_set, w))) # features*1
  ## w = w - learning_rate * gradient    

  # Momentum
  ## gradient = (-2) * np.dot(x_train_set.transpose(), (y_train_set - np.dot(x_train_set, w))) # features*1
  ## momentum = Lambda * momentum - learning_rate * gradient
  ## w = w + momentum

  # AdaGrad Method [Adaptive learning rate]
  ## gradient = (-2) * np.dot(x_train_set.transpose(), (y_train_set - np.dot(x_train_set, w))) # features*1
  ## adagrad_HSS += gradient ** 2
  ## w = w - learning_rate / np.sqrt(adagrad_HSS + eps) * gradient

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
  ## loss_sse = np.sum(np.power(y_train_set - (np.dot(x_train_set, w1) + np.dot(x_train_set**2, w2)), 2)) # SSE (Sum of squared errors)
  ## loss = np.sqrt(loss_sse/item) # RMSE (Root-mean-square error)
  ## loss_array.append(loss)
  ## # 文字顯示 Loss 變化
  ## if (not(count%1000)) | (count==iter_time-1):
  ##    print('Iter_time = ', count, "Loss(error) = ", loss)
  #---------------#
  # SGD
  ## if not(stop_loop):
  ##   for n in range(item):
  ##     count += 1
  ##     if(count>epoch):
  ##       count -= 1
  ##       stop_loop = True
  ##       break
  ##     x_n = x_train_set[n,:].reshape(1, features)
  ##     y_n = y_train_set[n].reshape(1, 1)
  ##     gradient1 = 2 * np.dot(x_n.transpose(), (np.dot(x_n, w1)+np.dot(x_n**2, w2) - y_n)) # features*1
  ##     gradient2 = 2 * np.dot((x_n**2).transpose(), (np.dot(x_n, w1)+np.dot(x_n**2, w2) - y_n)) # features*1
  ##     momentum = Lambda * momentum - learning_rate * gradient1
  ##     momentum2 = Lambda * momentum2 - learning_rate * gradient2
  ##     w1 = w1 + momentum
  ##     w2 = w2 + momentum2 
  ##     ## adagrad_HSS += gradient ** 2
  ##     ## w = w - learning_rate/np.sqrt(adagrad_HSS + eps) * gradient
  ##     loss_sse = np.sum(np.power(y_train_set - (np.dot(x_train_set, w1) + np.dot(x_train_set**2, w2)), 2)) # SSE (Sum of squared errors)
  ##     loss = np.sqrt(loss_sse/item) # RMSE (Root-mean-square error)
  ##     loss_array.append(loss)
  ##     if (not(count%10000)):
  ##        print('Iter_time = ', count, "Loss(error) = ", loss)
  ## else:
  ##   break
  # MBGD (Mini-Batch GD)
  if not(stop_loop):
    for n in range(math.ceil(item/batch_size)):
      count += 1
      if(batch_size*(n+1)>item):
        r = item - batch_size*n
        x_n = x_train_set[batch_size*n:batch_size*n+r,:]
        y_n = y_train_set[batch_size*n:batch_size*(n)+r]
      else:
        x_n = x_train_set[batch_size*n:batch_size*(n+1),:]
        y_n = y_train_set[batch_size*n:batch_size*(n+1)]
      gradient1 = 2 * np.dot(x_n.transpose(), (np.dot(x_n, w1)+np.dot(x_n**2, w2) - y_n)) # features*1
      gradient2 = 2 * np.dot((x_n**2).transpose(), (np.dot(x_n, w1)+np.dot(x_n**2, w2) - y_n)) # features*1
      momentum = Lambda * momentum - learning_rate * gradient1
      momentum2 = Lambda * momentum2 - learning_rate * gradient2
      w1 = w1 + momentum
      w2 = w2 + momentum2 
      ## adagrad_HSS += gradient ** 2
      ## w = w - learning_rate/np.sqrt(adagrad_HSS + eps) * gradient
      loss_sse = np.sum(np.power(y_train_set - (np.dot(x_train_set, w1) + np.dot(x_train_set**2, w2)), 2)) # SSE (Sum of squared errors)
      loss = np.sqrt(loss_sse/item) # RMSE (Root-mean-square error)
      loss_array.append(loss)
      if (not(count%10000)):
        print('Iter_time = ', count, "Loss(error) = ", loss)
      if(count>epoch):
        stop_loop = True
        break
  else:
    break
  

# 將重要的函數權重值存檔
np.save(r'LinearRegression\TrainingData_2019_Pinzhen\weight_1.npy', w1)
np.save(r'LinearRegression\TrainingData_2019_Pinzhen\weight_2.npy', w2)
'''
[Whole Training]
'''
## # Real Train wirh whole training set
## import random
## SavingDialog = False
## feature_weights = 15 * 9 + 1 # 因為有常數項參數 bias，所以 feature_weights 需要多加一欄
## item = len(x)
## w = np.zeros([feature_weights, 1]) # [feature_weights, 1]和(feature_weights, 1)一樣意思，就是存成 feature_weights * 1 的二維零矩陣
## w1 = np.zeros([features, 1]) # features*1 
## w2 = np.zeros([features, 1]) # features*1 
## x = np.concatenate((np.ones([item, 1]), x), axis = 1).astype(float) # 因為常數項的存在，所以 feature_weightsension (feature_weights) 需要多加一欄
## # 以下為調整 Gradient 的重要參數 
## Gradient_Method = 'SGD'
## learning_rate = 0.0001 # gradient descent 的常係數 𝜂 /Eta/
## iter_time = 30 # gradient descent 的迭代次數
## # AdaGrad 參數
## adagrad_HSS = np.zeros([feature_weights, 1]) # 代號 𝜎^t 意思是第 t 次迭代以前的所有梯度更新值之平方和 [HSS] Historical Sum of Grdient Square
## eps = 0.0000000001 # /epsilon/ 用途是避免在 Local minima (微分為零時) 停下來
## # SGD
## concat_x_y = np.concatenate((x, y), axis=1)
## random.shuffle(concat_x_y)
## x = concat_x_y[0:, 0:feature_weights]
## y = concat_x_y[0:, feature_weights:]
## del(concat_x_y)
## gc.collect()
## # 紀錄 Loss 值，繪圖用
## loss_array = []
## # 紀錄迭代次數
## count = 0 
## for t in range(iter_time):
##   ## # Vanilla
##   ## gradient = (-2) * np.dot(x.transpose(), (y - np.dot(x, w))) #feature_weights*1
##   ## w = w - learning_rate * gradient    
## 
##   ## # 使用 AdaGrad
##   ## adagrad_HSS += gradient ** 2
##   ## w = w - learning_rate / np.sqrt(adagrad_HSS + eps) * gradient    
##     
##   # SGD
##   for n in range(item):
##     count += 1
##     x_n = x[n,:].reshape(1, feature_weights)
##     y_n = y[n].reshape(1, 1)
##     gradient = (-2) * np.dot(x_n.transpose(), (y_n - np.dot(x_n, w))) # features*1
##     w = w - learning_rate * gradient  
##     #adagrad_HSS += gradient ** 2
##     #w = w - learning_rate/np.sqrt(adagrad_HSS + eps) * gradient
##     loss_sse = np.sum(np.power(y - np.dot(x, w), 2)) # SSE (Sum of squared errors)
##     loss = np.sqrt(loss_sse/item) # RMSE (Root-mean-square error)
##     loss_array.append(loss)
##     if (not(count%10000)):
##        print('Iter_time = ', count, "Loss(error) = ", loss)
## 
##     ## # 計算 Loss 值
##     ##  loss_sse = np.sum(np.power(y - np.dot(x, w), 2)) # Loss Function: SSE (Sum of squared errors)
##     ##  loss = np.sqrt(loss_sse/item) # rmse (Root-mean-square deviation) 
##     ##  loss_array.append(loss) # 紀錄 loss 值，繪圖用
##     ## # 文字顯示 Loss 變化
##     ##  if (not(t%100)) | (t==iter_time-1):
##     ##     print('Iter_time = ', t, "Loss(error) = ", loss)
## # 將重要的函數權重值存檔
## np.save(r'LinearRegression\TrainingData_2019_Pinzhen\weight_1.npy', w1)
## np.save(r'LinearRegression\TrainingData_2019_Pinzhen\weight_2.npy', w2)
'''
[Testing] 匯入測試資料
'''
# [load data]
## test_x = np.load(r'LinearRegression\TestingData\x_test_shuffle.npy')
## test_y = np.load(r'LinearRegression\TestingData\y_test_shuffle.npy')
test_x = np.load(r'LinearRegression\TestingData\x_test_shuffle2_publuc.npy')
test_y = np.load(r'LinearRegression\TestingData\y_test_shuffle2_publuc.npy')

# [Scaling]
# std_x, mean_x 要以訓練資料的數值進行標準化，才能將測試資料轉成相同的比例尺進行運算 
for i in range(len(test_x)): # 二維陣列的長度是算最外框裡面內涵的一維陣列個數，因此是 240，即輸入的測試資料項目個數
    for j in range(len(test_x[0])): # 135 個 Features
        if std_x[j] != 0: # 根據除法定裡，分母不得為零
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([len(test_x), 1]), test_x), axis = 1).astype(float)
test_x
# print(test_x)
'''
[Prediction] 對 Function 輸入測試資料及參數
'''
w1 = np.load(r'LinearRegression\TrainingData_2019_Pinzhen\weight_1.npy')
w2 = np.load(r'LinearRegression\TrainingData_2019_Pinzhen\weight_2.npy')
ans_y = np.dot(test_x, w1) + np.dot((test_x)**2, w2)
ans_y
#print(ans_y)
'''
[Save prediction result to CSV file]
'''
with open(r'LinearRegression\PredictionResult\PredictionResult.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['item_id', 'value']
    csv_writer.writerow(header)
    for i in range(len(ans_y)):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        # print(row)
'''
[紀錄實驗數據] Function的改善
'''
import datetime
# 顯示實驗數據
Vali_Ave_Err = np.sqrt(np.sum((y_validation - (np.dot(x_validation, w1) + np.dot(x_validation**2, w2)))**2)/len(y_validation))
Test_Ave_Err = np.sqrt(np.sum((test_y - ans_y)**2)/len(test_y))
## # 全部訓練集時以下解除，Validation 檢測時以下要屏蔽
## y_validation = '' 
## Vali_Ave_Err = 0 
time = (datetime.datetime.now()).strftime("%y/%m/%d %H:%M:%S")
Train_error_rate = round(np.sqrt((loss_sse)/(np.sum(y**2)))*100, 2)
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
'''
[Loss 變化分析] 繪製「函數的 Loss 值時變圖」
'''
plot_len = len(loss_array)
plt.axis([0, plot_len, 0, max(loss_array)+1]) # 二維座標圖 x, y 軸的顯示範圍
x_pos = np.linspace(0, plot_len, plot_len) # 依照迭代次數產生 x 軸座標
y_pos = loss_array # y 軸是 Loss value
plt.plot(x_pos, y_pos, '-', c='blue', markersize=4) # 形成每個點，繪出函數圖形
plt.show() # 顯示 plot 視窗
'''
[實驗紀錄是否存檔？]
'''
def SaveRecord(run_dialog ,a_head, a_row):
  while run_dialog:
    c = input('Save record? [y/n]')
    if len(c) == 1:
      if(c == 'y')|(c == 'Y'):
        print('Saving...')
        with open(r'LinearRegression\PredictionResult\ExperimentRecord_Redesign.csv', mode='a', newline='') as csvfile:
          writer = csv.writer(csvfile)
          header = a_head
          # writer.writerow(header) # 先用 mode w 寫 header，之後用 mode a 新增數據
          writer.writerow(a_row)
          print('Save record successfully!')
        break
      elif (c == 'n')|(c == 'N'):
        print('Unsave.')
        break
      else:
        print('plz enter again, only one character \'y\' or \'n\'.')
        continue
    else:
      print('plz enter again, only one character \'y\' or \'n\'.')
      continue
SaveRecord(SavingDialog, head, row)

    
