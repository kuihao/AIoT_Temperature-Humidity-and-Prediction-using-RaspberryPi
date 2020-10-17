# 2020/10/17 資料預處理紀錄
0. 嘗試全程使用Python(不依賴Microsoft Excel)，當未來面對**真正的大數據資料**知道如何善用python有效率地完成工作
1. 使用Python Module: **pandas**
    * *pandas.read_csv* 讀取CSV檔案，以*DataFrame*型態存取
    * *<data>.iloc[:,:].values* 取得*DataFrame*中特定範圍的資料，以*Module: Numpy*的*ndarray*型態存成二維陣列，其中陣列資料是以String型態儲存