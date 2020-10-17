# import numpy as np 
# import matplotlib.pyplot as plt #畫出圖型 
import pandas as pd #資料處理

dataset = pd.read_csv('LinearRegression\TrainingData_2019_Pinzhen\CSV_2019PingZhen.csv') 
x = dataset.iloc[:, 3: ].values # iloc[ 列起:列終, 行起:行終]
y = x[1, 1]
print( type(y) )

'''
普通測站資料註記說明：# 表示儀器檢核為無效值，* 表示程式檢核為無效值，x 表示人工檢核為無效值，
A 係指因儀器疑似故障警報所產生的無效值，NR 表示無降雨，空白 表示缺值。風向資料888代表無風，
999則代表儀器故障。
'''
'''
from sklearn.preprocessing import Imputer
#Imputer : 專門處理數據缺失的類別
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])#1:3 = 1~2
X[:, 1:3] = imputer.transform(X[:, 1:3])
'''