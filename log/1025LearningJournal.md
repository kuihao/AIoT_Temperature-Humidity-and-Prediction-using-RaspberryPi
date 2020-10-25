# 2020/10/25
## 檢測 pandas 是 call by reference
---
### pd.iloc
    df = pd.DataFrame(np.ones(12, dtype=int).reshape(3, 4))
    df_slice = df.iloc[1:,2:]
    print('df_slice\n',df_slice)
    print(df_slice.iloc[0:1, 0:1])
1. df.iloc 是 call by reference
2. df.iloc[] 若是切出一維陣列，則會存為 Series 型態
3. df.iloc[] 若是切出二維陣列，則會存為 DataFrame 型態
4. 經過 df.iloc[] 切出來的 DataFrame，雖然 Index(列名) 和 Columns Name(欄名)<br>
   會保持原本被切割的欄名及列名，但實際上 df.iloc[] 並不是用名稱來定位區間 (df.loc[]才是)，<br>
   df.iloc[] 是用元素的「位置序號」來定位，行與列永遠都是從 0 開始編號，就如同 C 語言陣列的 index 是「絕對座標」的概念  
---
### call by reference
    df = pd.DataFrame(np.ones(12, dtype=int).reshape(3, 4))
    df_slice = df.iloc[1:,2:]
    df_slice.iloc[0:1, 0:1] = 10
    print('df\n', df)
    print('df_slice\n',df_slice)
---
改用 nparray存取，則不再參考原本的位址    
    df = pd.DataFrame(np.ones(12, dtype=int).reshape(3, 4))
    df_slice = np.array(df.iloc[1:,2:])
    df_slice[1,1] = 10
    print('df\n', df)
    print('df_slice\n', df_slice)
---
### Bug: dataframe 15*24 全部轉型為 np.array 儲存空間有機會爆掉
    B = row_testdata[id_filter].iloc[0:15, 1:]
    #C = np.array(B.iloc[0:11, :].values.reshape(11, 24))