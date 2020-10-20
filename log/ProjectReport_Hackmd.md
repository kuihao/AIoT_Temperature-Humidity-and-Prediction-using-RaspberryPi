# Homework 1: PM2.5預測

目錄
[ToC]

## 資料說明
- 資料來源：==氣象網址==
- 細項：整理<br>https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/log/1019LearningJournal.md

## 制定模型
- 模型類別：Machine Learning 中**監督式學習**包含**分類**與**回歸**兩種，由於本資料集是連續性資料，故採用回歸模型做預測是恰當的。
- 選擇Features
- Label Ground Truth
- 定義Function

## 資料預處理(pre-processing)
- 無效值說明：==Line 14-20==<br>https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/LinearRegression/DataPreprocession.py
- 資料清洗：==無效值清理== ==\<code>== ==無降雨NR(No Rain)取代為0==
- 缺失值填補：==使用平均值法== <br>https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/log/1018LearningJournal.md
- 資料分割：==訓練資料集== ==測試資料集==
    - **補值的意義:** 當蒐集的資料有缺失時有兩個做法，方法一是直接把缺失資料全部捨棄，但當數據缺稀或比例資料不均時，直接捨棄資料是相當浪費且失準的；因此方法二是有邏輯地填補缺值並填補值不會過度影響全部數據的分布。

## 訓練模型：Linear Regression
- 資料萃取
    - 訓練資料
    - 測試資料
- 資料縮放 (Z-score 標準化)
- 分割訓練資料 "train_set"、"validation_set"
- 制定Loss Function
- 使用Gradient descent調整參數

## 測試模型 & 分析結果
- 測試資料萃取
- 測試結果
- **嘗試不同的learning rate，分析對於收斂結果的影響**
- **嘗試不同的特徵選擇(feature selection)**
- **嘗試不同的特徵縮放(feature scaling)**
- https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC2-4%E8%AC%9B-%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86-missing-data-one-hot-encoding-feature-scaling-3b70a7839b4a
- 如何讓預測結果更進一步

## HackMD工具
- LaTeX for formulas
$$
x = {-b \pm \sqrt{b^2-4ac} \over 2a}
$$

- Code block with color and line numbers：
```javascript=16
var s = "JavaScript syntax highlighting";
alert(s);
```

- UML diagrams
```sequence
Alice->Bob: Hello Bob, how are you?
Note right of Bob: Bob thinks
Bob-->Alice: I am good thanks!
Note left of Alice: Alice responds
Alice->Bob: Where have you been?
```

