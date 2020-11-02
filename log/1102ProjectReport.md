# 2020/11/02 HOMEWORK 1 - FINAL REPROT
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
----
## PROBLEM DESPRICTION
    Through Prof. Hung-Yi Lee’s OCW (Open Course Ware) and the homework “PM2.5 prediction” learning the knowledge of Linear Regression.
* **Data source:** “環境保護署 2019 年桃園市平鎮區空氣品質監測資料”

----
## HIGHLIGHT
### PRE-PROCESSING
### ADJUST LEARNING  RATE
分析對於收斂結果的影響
### FEATURE SELECTION
### FEATURE SCALING
### PERFORMANCE IMPORVEMENT
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
----
## DISCUSSION
