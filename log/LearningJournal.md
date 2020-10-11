# 2020/10/10 Debug:Linux command line 無法執行python檔案  
	* 狀況描述：確定pip模組都正確按裝於python3，但編譯檔案時卻找不到模組  
	* 解決：初始預設的python版本為python2，需要將預設版本設定成python3才能正確使用指令「phthon <檔案名>.py」，不更改的話需使用指令「python3 <檔案名>.py」才能正確執行 
	* 更改版本的指令： $ sudo alias python='/usr/bin/python3.4'  
	* 註:python3的版本要依所安裝之版本為準*

# 2020/10/11 terminal ssh/scp commands :  
	* step1: 查詢ip  
		linux: $ ifconfig  
		windows: ipconfig  
	* step2: 本地傳輸至遠端樹莓派    
		[ssh] ssh pi@<ip> #<ip>替換成LAN IP    
		[scp] scp <本地檔案存放路徑(含檔名)> <帳號>@<遠端主機位址>:<遠端檔案位置路徑(含檔名)>   
		[scp/batch] scp -r <本地檔案夾路徑> <帳號>@<遠端主機位址>:<遠端檔案俠位置路徑>  
		注意: 樹莓派格式初始為 pi@<IP>  
# 2020/10/11 進度報告:<br>
	* <b>已完成</b>
		* 樹莓派的python程式撰寫
			* 溫溼度感測
			* CSV自動存檔
			* 雲端儲存資料
			* 網頁顯示資料
	* <b>待完成</b>
		* google excel表單儲存及資料定時下載功能(約3小時)
			* ref1: https://help.ubidots.com/en/articles/934604-how-to-export-your-ubidots-data-to-google-sheets
			* ref2: https://help.ubidots.com/en/articles/2331043-import-data-to-ubidots-from-google-sheets
			* ref3: https://www.hackster.io/ubimaker/get-ubidots-data-from-google-sheets-057040
			* ref4: https://stackoverflow.com/questions/11619805/using-the-google-drive-api-to-download-a-spreadsheet-in-csv-format
			* ref5: https://developers.google.com/sheets/api/guides/create
			* ref6: https://www.youtube.com/watch?v=gcQ3P95mEM0
		* linear regression training
			* ref1: https://www.youtube.com/watch?v=CXgbekl66jc&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49 
			* ref2 觀測公開資料CODIS: https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp
		* 異常檢測分析
			* ref: https://www.itread01.com/content/1545816302.html
		* email notify
			* ref 使用python: https://www.learncodewithmike.com/2020/02/python-email.html


