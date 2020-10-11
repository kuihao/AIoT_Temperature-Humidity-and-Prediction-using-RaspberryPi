# 2020/10/10 Debug:Linux command line 無法執行python檔案  
	## 狀況描述：確定pip模組都正確按裝於python3，但編譯檔案時卻找不到模組  
	## 解決：初始預設的python版本為python2，需要將預設版本設定成python3才能正確使用指令「phthon <檔案名>.py」，不更改的話需使用指令「python3 <檔案名>.py」才能正確執行 
	## 更改版本的指令： $ sudo alias python='/usr/bin/python3.4'  
		*註:python3的版本要依所安裝之版本為準*

# 2020/10/11 terminal ssh/scp commands :  
	## step1: 查詢ip  
		linux: $ ifconfig  
		windows: ipconfig  
	## step2: 本地傳輸至遠端樹莓派    
		[ssh] ssh pi@<ip> #<ip>替換成LAN IP    
		[scp] scp <本地檔案存放路徑(含檔名)> <帳號>@<遠端主機位址>:<遠端檔案位置路徑(含檔名)>   
		[scp/batch] scp -r <本地檔案夾路徑> <帳號>@<遠端主機位址>:<遠端檔案俠位置路徑>  
		注意: 樹莓派格式初始為 pi@<IP>  