# 2020/10/17 資料預處理紀錄
0. 嘗試全程使用Python(不依賴Microsoft Excel)，當未來面對**真正的大數據資料**知道如何善用python有效率地完成工作
1. 資料匯入
    * 使用Python Module **pandas**的*pandas.read_csv* 讀取CSV檔案，以*DataFrame*型態存取
        * [補充] *pandas*有兩種資料結構：
            1. DataFrame 就是一張表格(table)
            2. Series 就是單欄(column)或是單列(row)的所有資料
    * *<data>.iloc[:,:].values* 取得*DataFrame*中特定範圍的資料，以*Module: Numpy*的*ndarray*型態存成二維陣列，其中陣列資料是以String型態儲存
2. 資料清洗
    * 檢查空值: 使用Function *info()* 可以觀察每一欄Non-Null(非空值)的數量，進而得知哪幾欄有空值
3. 資料預處理
    * 資料分割
* 補充
    * *pandas*的常用函式
        * .shape →回傳列數與欄數
        * .describe() →回傳一些描述性統計，平均值、標準差、最小最大值..等
        * .head() → 回傳前 5 筆資料內容，()內可自填想回傳從頭數來的「幾」筆資料
        * .tail() → 回傳後 5 筆資料內容，()內可自填想回傳倒數「幾」筆資料
        * .columns → 回傳所有欄位名稱
        * .index → 回傳 index(索引值)有多少
        * .info() → 回傳資料內容的資訊
        * 資料切片
            * .loc[列起:列終, '欄起Label':'欄終Label'] → 取得某列資料，包含列終，選欄可以用實際Label名稱作位置根據，亦可使用Index
            * .iloc[列起:列終, 欄起:欄終] → 取得某列資料，不包含列終，i是指Integer(或Index?說法不一)，根據實際Index位置計算
            * 其他示範
                * df['地震個數'][:3] 利用 [ ] 包住欄位名稱，列與欄的寫法順序不要緊
                * df[['年份', '地震個數']] 只要年份和地震個數的欄位
        * 過濾資料
            * 布林過濾器：df['年份'] == 2013 符合者回傳T, 否則為F
            * df[df['年份'] == 2013] DataFrame []內嵌入過濾器即可作篩
        * 資料分析
            * value_counts() → 可以對 Series(某欄或某列) 裡面內容進行「計數」
            * sum() → 可以對 Series(某欄或某列) 裡面內容進行「加總」
            * mean() → 可以對 Series(某欄或某列) 裡面內容進行「平均值計算」
            * groupby() → 可以將資料依照自己想要的欄位做「分組」
    * *matplotlib*入門
        * 類似R語言的qqplot2是資料視覺化的工具，可產生圖表
        * 匯入語法: from matplotlib import pyplot as plt
        * 常用圖表: plt.plot() 折線圖、plt.bar() 長條圖、plt.scatter() 散布圖，資料預設放在y軸
        * plt.show()、免show: %matplotlib inline

* Reference
    * Numpy: https://ithelp.ithome.com.tw/articles/10195434
    * Pandas, Data Science: https://www.taiwancodeschool.com/freelearning
    * 環保署空氣品質監測網-空氣公開數據: https://airtw.epa.gov.tw/CHT/Themes/LinkOut.aspx
    * Data Science Example: https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC4-1%E8%AC%9B-kaggle%E7%AB%B6%E8%B3%BD-%E9%90%B5%E9%81%94%E5%B0%BC%E8%99%9F%E7%94%9F%E5%AD%98%E9%A0%90%E6%B8%AC-%E5%89%8D16-%E6%8E%92%E5%90%8D-a8842fea7077
    * Data Preprocession: https://medium.com/@doremi31618/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-%E6%95%B8%E6%93%9A%E9%A0%90%E8%99%95%E7%90%8601-ae90853978da