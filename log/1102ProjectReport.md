# 2020/11/02 HOMEWORK 1 - FINAL REPROT
## CONTENT
* <a herf="#anchor0">PROBLEM DESPRICTION</a>
* <a herf="#anchor0">HIGHLIGHT</a>
    * <a herf="#pre-processing">PRE-PROCESSING</a>
    * <a herf="#adjust-learning--rate">ADJUST LEARNING  RATE</a>
    * <a herf="#feature-selection">FEATURE SELECTION</a>
    * <a herf="#feature-scaling">FEATURE SCALING</a>
    * <a herf="#performance-imporvement">PERFORMANCE IMPORVEMENT</a>
* [LISTING](#listing)
* [TEST AND RUN](1102ProjectReport.md#test-and-run)
* DISCUSSION
----
## PROBLEM DESPRICTION
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
* [ReDesignModel_Seq.py](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/masterLinearRegression/ReDesignModel_Seq.py): Same as LinearRegression.py but use "squarefunction model"
* [getHT.py](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/getHT.py):Raspberry Pi read temperature and humidity datas from DHT11 
* [CSVDownloader_GoogleSheet.py](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/masterGoogleSheetTransfer/CSVDownloader_GoogleSheet.py): Automatically download datas frominternet google sheet
* [getUbidotsData.gs](https://github.com/kuihaoAIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/masterGoogleSheetTransfer/getUbidotsData.gs): Get datas from Ubidots to Google sheet, "gs"means "Google Apps Script"
----
## TEST AND RUN
----
## DISCUSSION
