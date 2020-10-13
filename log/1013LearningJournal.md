# 2020/10/13
* 修改/優化getHT程式碼：調整迴圈增加效能、意外狀況處理(網路斷線時，重新嘗試執行，仍未成功則跳過雲段上傳步驟)
* google雲端Excel(csv)程式碼更新：增加每小時(或 每15分鐘)**自動更新google資料**的功能，使雲端資料隨時保持最新資訊，並且使用由google提供服務，降低硬體設備維護成本
* 閱讀google api文件、撰寫**CSVDownloader.py程式**：此python程式可以直接抓取google雲端Excel的資料，並**自動儲存成CSV檔案**
* 學習python數據分析