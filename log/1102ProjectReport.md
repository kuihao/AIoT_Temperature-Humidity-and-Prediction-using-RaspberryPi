# 2020/11/02 HOMEWORK 1 - FINAL REPROT
ARCHITECTURE<br>
<center><img src="https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/log/Architecture.png" width="70%" height="70%" alt="Architecture" title="Architecture" /></center>

----
## CONTENT
* [PROBLEM DESPRICTION](#problem-despriction)
* [HIGHLIGHT](#highlight)
    * [PRE-PROCESSING](#pre-processing)
    * [ADJUST LEARNING RATE](#adjust-learning--rate)
    * [FEATURE SELECTION](#feature-selection)
    * [FEATURE SCALING](#feature-scaling)
    * [PERFORMANCE IMPORVEMENT](#performance-imporvement)
* [LISTING](#listing)
* [TEST AND RUN](1102ProjectReport.md#test-and-run)
* [DISCUSSION](#discussion)

-----
## PROBLEM DESPRICTION
#### Through Prof. Hung-Yi Lee’s OCW (Open Course Ware) and the homework **“PM2.5 prediction”** learning the knowledge of Linear Regression.
* **Data source**<br>
“環境保護署 2019 年桃園市平鎮區空氣品質監測資料”
* **Choose Model**<br>
    * Let "**y'**" be the **prediction PM2.5**,<br>
    "**x**" be the **features** of 9 hours sensor types (ex. CO, NO, PM10, PM2.5, Rainfalls),<br>
    "**w**" be the **weights** of features
    * Function1: **y’ = w * x**
    * Function2: **y’ = w1 * x + w2 * x^2**
* **Loss function**<br>
**RMSE** (root-mean-square error)
* **Gradient descent**<br>
Vanilla (basic), SGDM, MBSGD, AdaGrad, RMSProp, Adam

* **Flow chart:**<br>
  ![Linear regression flow chart](https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/log/flowchart_linear_regression.png "Linear regression flow chart")

----
## HIGHLIGHT
### PRE-PROCESSING
* Using **Python** rather than excel
* **Invalid value, null value**<br>
filled with the **mean** of sensor types
* **Split training dataset**<br>
**20** days/month
* **Split testing dataset**<br>Remaining days/month (8~11 days)
* **Data extraction**<br>
Every 9 hours with 15 sensor type/hour be the features and **predict the PM2.5 of 10th hour**

### ADJUST LEARNING  RATE
![plot of adjusting learing rate](https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/log/LearningRateAdjusting.png "plot of adjusting learing rate")
* Model: Power of one
* Gradient: Vanilla g radient descent
* Iteration: 1,000 times

### FEATURE SELECTION
![plot of features selection](https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/log/FeatureSelection.png "plot of features selection")
* Training set number: 4,521
* Validation set number: 1,131
* Iteration: 10,000 times 
* Gradient: AdaGrad

### FEATURE SCALING
![plot of features scaling](https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/log/FeatrueScaling.png "plot of features scaling")
* Z-Score (Standard Score): (x – μ) / σ
* Max-Min: x - min(x) / max(x) - min(x)

### PERFORMANCE IMPORVEMENT
![plot of IMPORVEMENT](https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/log/FinalResult.png "plot of IMPORVEMENT")
* Training set number: 5,652
* Testing set number: 1,446
* Iteration: 150,000 times

----
## LISTING
* [DataPreprocession.py](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/masterLinearRegression/DataPreprocession.py): Data cleaning, Dataset spliting
* [TestDataPreprocessing.py](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/masterLinearRegression/TestDataPreprocessing.py): Feature extraction, Shuffle
* [LinearRegression.py](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/masterLinearRegression/LinearRegression.py): Feature extraction, Shuffle,  Normalization,Split validation set, Training, Prediction
* [ReDesignModel_Seq.py](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/masterLinearRegression/ReDesignModel_Seq.py): Same as [LinearRegression.py](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/masterLinearRegression/LinearRegression.py) but use "squarefunction model"
* [getHT.py](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/getHT.py): Raspberry Pi reads temperature and humidity datas from DHT11 
* [CSVDownloader_GoogleSheet.py](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/masterGoogleSheetTransfer/CSVDownloader_GoogleSheet.py): Automatically download datas frominternet google sheet
* [getUbidotsData.gs](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/masterGoogleSheetTransfer/getUbidotsData.gs): Get datas from Ubidots to Google sheet, "gs"means "Google Apps Script"

----
## TEST AND RUN
<center><img src="https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/log/runcode.gif" width="70%" height="70%" alt="Demo" title="Demo" /></center>

----
## DISCUSSION
1. What's happened to **adjusting learning rate?**<br>
    * If learning rate is **not minimal enough**, loss would not converge (even become greater than greater)<br>
    * If learning rate is **too small**, loss convergence would be inefficient
    * Only learning rate **just fit**, loss convergence would be efficient
2. Why **removing** some features would **improve** the performance?
    * Maybe the other features effect on PM2.5 be not that fast
3. Why **Z-Score is better** in performance of convergence?
    * Z-Score is appropriate in unknowing maximum and minimum
4. How to **improve model**?
    1. Shuffle training and testing data
    2. Using AdaGrad rather than Vanilla gradient descent
    3. Feature only select PM10 and PM2.5
5. **Final improvement of model:**<br> converge loss value descents **0.0225**