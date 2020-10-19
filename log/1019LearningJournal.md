# 2020/10/19 預測模型訓練紀錄
## Data說明
* 總共有15種不同的feature，全年每小時記錄一次<br>
  因此數據共有 15(*features*) * 24(*hours*) = **360(筆/日)**
* Feature說明(空氣品質測項)
    * 資料時空及來源：民國108年(西元2019年)環保署
    * **Feature數量：15個**
    * 英文代號解釋
        * **'AMB_TEMP': 溫度(℃)**
        * 'CO': 一氧化碳(ppm)
        * 'NO': 一氧化氮(ppb)
        * 'NO2': 二氧化氮(ppb)
        * 'NOx': 氮氧化物(ppb)
        * 'O3': 臭氧(ppb)
        * 'PM10': 粒徑小於等於10微米懸浮微粒(μg/m3)
        * **'PM2.5': 粒徑小於等於2.5微米細懸浮微粒(μg/m3)**
        * **'RAINFALL': 雨量(㎜)**(有時紀錄以'RAIN_INT'(降雨強度))
        * **'RH': 相對濕度(percent)**
        * 'SO2': 二氧化硫(ppb)
        * 'WD_HR': 小時風向(degrees)
        * 'WIND_DIREC': 風向(degrees)
        * 'WIND_SPEED': 風速(m/sec)
        * 'WS_HR': 小時風速值(m/sec)
    * 本測站未檢測項目
        * CO2: 二氧化碳(ppm)
        * THC: 總碳氫化合物(甲烷及非甲烷碳氫化合物)(ppm)
        * NMHC: 非甲烷碳氫化合物(ppm)
        * CH4: 甲烷(ppm)
        * PH_RAIN: 酸雨(pH)
        * RAIN_COND: 導電度(μmho/cm)
    * 單位說明
        * 風速：以每小時最後10分鐘算術平均
        * 風向：以每小時最後10分鐘向量平均
        * 平均風速：以整個小時的風速算數平均
        * 平均風向：以整個小時的風向向量平均
        * μg/m3為微克/立方公尺
        * ppb為百萬分之一
## Loss Function的製作：
* 可用**任何合理的方式**評斷 Function 預測效果的優劣
* 課堂教導的方法是用類似統計學的 Variance (變異數) 做評估
    * **Variance 變異數** 是「使用全部資料」做離散程度評估的方法，其意義是<br>
      每一個觀察值x_i(數據、樣本)與平均數的x_bar(或Mu)的「距離平方和的平均(平均離均差、平均離差)」<br>
      其中，「取平方」的意義是為了將差值**取正值**，這樣會比取絕對值方便計算。
* 套用在此就是將Label值(真實解答，視其為x_bar的角色)與每個預估值(視為x_i的角色)相差，<br>
  也就是「**估計誤差**」的意思。
* 對於 Simple variance (抽樣變異數) 取平均的數值須為「總數減1 (n-1)」，如此才能不偏估計。
* **Standard deviation 標準差** 就是變異數的平方根，數值單位和觀測值視相同的，因此更便於理解
##　疑問：
* 為何訓練資料要切割成12個月，而非每個月串在一起，用年做分割?
* 為何Line 55要做reshape(1, -1)?
    * What is reshape? https://ithelp.ithome.com.tw/articles/10195830
## 可以參考的相關 Reference :
* Adagrad : https://youtu.be/yKKNr-QKz2Q?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&t=705
* RMSprop : https://www.youtube.com/watch?v=5Yt-obwvMHI
* Adam https://www.youtube.com/watch?v=JXQT_vxqwIs
* 以上 print 的部分主要是為了看一下資料和結果的呈現，拿掉也無妨。另外，在自己的 linux 系統，可以將檔案寫死的的部分換成 sys.argv 的使用 (可在 terminal 自行輸入檔案和檔案位置)。
* 本作業學習自：https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C#scrollTo=Y54yWq9cIPR4

# 助教留言
* 最後，可以藉由調整 learning rate、iter_time (iteration 次數)、取用 features 的多寡(取幾個小時，取哪些特徵欄位)，甚至是不同的 model 來超越 baseline

# 明天的工作
1. 再順一遍 Code 理解每個細節(LR function code 內部構造、Gradient descent微分、最終Average error呈現、輸出檔案的位置擺放)
2. 萃取 Testing Data
3. 嘗試使用 Validation_set 調整新的 Model
4. 觀看 Gradient descent 的影片更加了解原理
