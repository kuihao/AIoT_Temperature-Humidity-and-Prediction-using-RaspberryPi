# 2020/10/20 進度報告
## 摘要
* 目前已完成**資料預處理**、**Linear Regression的程式碼**，程式碼能正確運作且已理解內部原理。<br>
* 之後將進行：
    1. 更深認識Gradient descent中adagrad的原理
    2. learning rate的調整及分析
    3. 嘗試不同的特徵選擇
    4. 測試不同的特徵縮放
## 資料預處理
- 無效值處理
    - 使用**phyhon**的字串比對，替換數據中包含無效值#、*、x、A、888、999的資料，改成特殊值'666'
    - 特殊值 'NR' (No Rain) 替換成 0
- 缺失值處理
    - 利用函式 *.info()* 檢查資料中的空值
    - 利用函式 *.fillna()* 將空值皆填上特殊值'666'
    - 使用 *sklearn.impute 模組*的 *SimpleImputer物件*，針對前設定的特殊值'666'使用**平均值填補法**，同一個小時的資料理論上擁有相近的數值，選用平均值填補空缺資料的好處是對於原先資料的分布影響較低。<br> 另外有 KNN (k-Nearest Neighbors) 填補法，選用鄰近資料進行填補，理論上能得到更佳的資料，是未來可以嘗試的方向
- 資料分割
    - 將資料中的日期轉型成 *Datetime* 型態，利用 *Pandas 模組*中 *DataFrame* 的 *Filter 功能*，依據日期將資料切割成 Training set 及 Testing set。
    - 切割規則是每個月前20天為訓練資料，之後至月底的資料為測試資料。
    - 使用 *csv 模組* 將資料匯出

## 模型訓練
- 將CSV資料匯入並使用 *numpy模組* 存成 *numpy array*，優化對於矩陣的運算操作
- 模型制定
    - 今年環保署測站資料的檢測項目只有 15 項，因此選擇 15 項檢測資料皆作為 Function 的Features，並且將連續9個小時的資料也列入考慮，第十個小時的 PM2.5 數值作為 Ground Truth (Label解答)，此 15 * 9 = 135 項資料皆作為 Features，理論是**不同測項每個小時對 PM2.5 數值的影響會有所不同**
- 資料萃取
    - 依據 Features 的制定原則從資料集中萃取並製作成 471 * 12 = 8,892 份訓練資料，及471份對應的 Ground Truth
    - 471 是指每個月扣除月底最後9個小時之後可製作出 471 份，不過這個分割方式會缺少每個月月底 9 小時的資料。未來可以改成每個月資料相連，唯獨排除年底最後 9 小時的資料，改善缺少月底訓練資料的問題
    - 目前測試資料暫時以 Microsoft Excel 軟體萃取，之後會使用與訓練資料相同的方法進行萃取，並用 python 隨機抽取資料作為測試資料
- Normalize (資料標準化)
    - 目前採用 **Z-Score (Standard Score)** 的方式將 Features 進行標準化，此步驟又稱為資料縮放 (feature scaling)，概念是將不同單位的資料統一成相同的比例尺度，依此Function Space 會較為平均，有助於後續使用梯度下降法時能更有效率地找到最佳的參數組合
- 將訓練資料分割為 Train set、Validation set
    - 此用途為評量 Model 好壞，目前只有一個 Model 因此尚未用到
    - 進行 Model 訓練時，切記不應直接針對 Testing data 進行 Model 改良，這會造成 Public testing data 與 Private testing data 預測能力的不一致，造成對於真實數據預測能力的低估
    - 使用方法是利用 Train set 對不同的 Model 進行訓練，再用 Validation set 篩選出最好的 Model，其中可以切割出多組 Train se t與 Validation set 並與 Model 交叉比對，依據 Average error 選出最佳的 Model
- 制定 Loss Function 評斷 Function 參數的優劣
    - 此採用**Root Mean-Square Deviation (RMSD ,估計誤差)** 的方法來衡量，將 Function 輸出的 PM2.5 預測數值與 Ground Truth 相減並取平方(將數值取正數)，得出預測誤差值，並將 8,892 份訓練資料的誤差值總和再開方得出 Loss 值，Loss 值越小表示預測誤差越小，即 Function 所配對的參數組合是越好的
    - 未來可以針對 Loss Function 可以再加上特定參數的平方值，如此一來考量 Minimal Loss 值時，便會連帶考慮該參數也盡量小，並搭配一個係數，調整 Loss Function 對特定參數的考慮程度，這個方法就稱為 **增加 Function 的平滑程度**，Function 越平滑越減少 Feature 的影響力。當 Function 呈現 Overfitting 時，此方法可以降低 Variance，但也會增加 Bias，需要透過估計誤差值來找尋恰當的平衡
- Gradient descent (梯度下降法)
    - 精隨：將目前參數加總「目前參數對 Loss Function 的微分值」，以此調整參數該變大或變小才能接近 Ground Truth，其中可調整係數 *learning rate (η /Eta/)* 來決定每次調整的幅度，如此一來能更有效率地調整參數，不必窮舉所有參數。*iter_time* 是 Gradient descent 執行的次數。
    - **待理解：adagrad**
## 模型測試
- 輸出正常，目前測試資料 PM2.5 的平均誤差值是 0.0645(測試240組資料，資料的選擇非完全隨機)
## 這星期的學習筆記(10/12-10/19)
- https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/tree/master/log

## 程式碼
* 預處理的程式碼：https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/LinearRegression/DataPreprocession.py

* Linear Regression的程式碼：https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/LinearRegression/LinearRegression.py