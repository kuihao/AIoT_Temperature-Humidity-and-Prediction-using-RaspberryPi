import numpy as np 
import matplotlib.pyplot as plt #畫出圖型 
import pandas as pd #資料處理

np1 = np.array([1, 2, 3])
np2 = np.array([3, 4, 5])
# 陣列相加
print(np1 + np2) # [4 6 8]
# 顯示相關資訊
print(np1.ndim, np1.shape, np1.dtype) # 1 (3,) int64 => 一維陣列, 三個元素, 資料型別
np3 = np.array([1, 2, 3, 4, 5, 6])