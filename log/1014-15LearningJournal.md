# 2020/10/14 機器學習筆記
0. 學習來源：Hung-yi Lee, Machine Learning 2020: Course Introduction, http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html
1. 機器學習首先要釐清要找「什麼功能、用途」的函式
    * 例如：找出「預測數值」的「函式」，**本專案以預測溫度、相對濕度、降雨機率**
    * 這樣預測功能的函式稱為「Regression」
    * 函式的輸入：過去溫度、相對濕度、(是否降雨、降雨量)的資料
    * 函式的輸出：未來溫度、相對濕度、降雨率
2. 訓練方法分為「Supervised」、「Reinforcement」與「Unsupervised」
    * Supervised Learning
        * Supervised Learning 輸入的 Trainning Data 需要Label預期輸出的解答
        * Supervised Learning 會給目前函式一個Loss值，代表這個函式的好壞程度，Loss值越小越函式越準，使機器自行找出Loss值最小的函式
    * Reinforcement Learning
        * 讓機器直接從經驗中學習，單純從每次測試的結果(reword)回饋給自己調整函式
        * 初版Alpha Go是Supervised(用棋譜訓練好) + Reinforcement
        * 新版Alpha Go是純Reinforcement
    * Unsupervised Learning(Auto-encoder)
        * 輸入沒有Label的Training Data
3. 機器如何找出好的函式？
    * 先給定Function Pool(一堆函式)的Range(範圍)
        * 例如：規定Regression、Classification是Linear Function
        * 例如：規定Neuro Network Architecture為RNN、CNN
    * 使用Learning Algorithm(如：Gradient Descent)從給定Range中找出好的函式
4. 前沿研究
    * 學習何為Explainable AI：具備可解釋性、說明為何如此判斷
    * Adversarial Attack：對資料加入惡意雜訊，使AI判別錯誤
    * Network Compression：如何將龐大的Network縮小使其可安裝在行動裝置或App中
    * Anomaly Detection：使AI足以判斷自己無法進行辨識
    * Transfer Learning：當測試資料與訓練資料落差極大，如何學習
    * Meta Learning：訓練AI學習「如何訓練AI、找出最好的Learning Algorithm」，以減少訓練成本
    * Life-long Learning(終生學習)：一個可以不斷學習的機器(常見的AI無法學新的東西)，此又稱Never Ending Learning, Continous Learning, Incremental Learning
# 2020/10/15 機器學習筆記 Regression
0. 學習來源：Hung-yi Lee, ML Lecture 1: Regression - Case Study, https://youtu.be/fegAeph9UaA
1. **制定Linear Function:**
    * <img src="http://chart.googleapis.com/chart?cht=tx&chl= y = b + \sum w_i  \cdot x_i" style="border:none;" alt="y = b + summaton(w_i * x_i)">
    * <img src="http://chart.googleapis.com/chart?cht=tx&chl= x_i" style="border:none;" alt="x_i"> 稱為**Feature**即Function的輸入屬性(Input Attributes)，例如：預測下雨需要參考的溫度、相對濕度等天氣因子
    * <img src="http://chart.googleapis.com/chart?cht=tx&chl=(w_i, b)" style="border:none;" alt="w_i, b"> 則為Weight和Bias，是Function的係數(可以是任意值)
    * ( Latex數學式線上編輯器： https://www.codecogs.com/latex/eqneditor.php )
2. 判別出好的Function
    1. 蒐集Traning Data
        * 決定Function的輸入及輸出，Output Data需要Label(使用Supervised Learning方法)
    2. 定義Loss Function L
        * Input: A **Function** of *the Linear Function Set*
        * Output: 一個**Loss值**，象徵該Function好壞程度
            * <img src="http://chart.googleapis.com/chart?cht=tx&chl=L(w, b) = \sum_t(f\hat{}(t)-f_i(t))^{2}" style="border:none;" alt="所有Train Data代入「真實輸出值(y_hat或稱Label值)與估計值(Function的輸出值)的差平方」並計算其總合">
            * t的range為所有的Input Training Data，f_i(t)為Function Set中的一個Function並Feature代入所有的Training Data
            * 意義上，這個Loss Function就是評估該輸入Function的**估測誤差**，就是所有Train Data代入預測Function所得的值 與 真實值的差距總和。*(用途)若某一天差距總和為零，就表示找到一個完美預測Train Data的Function。所以究此階段而言會希望找到Loss值最小的Function*
    3. 找出Best Loss Function
            * <img src="http://chart.googleapis.com/chart?cht=tx&chl=L(w, b) = f^{*} = argminL(f)" style="border:none;" alt="argument of the minimal of the function">
            * 也就是找到最適當的參數(w*, b*)組成的Function，其代入Loss Function能夠得到最小的Loss值，則這個Function with (w*, b*)就是我們要找的最佳Function
    4. 解出Best Loss Function
        * 方法一：該等式可以用線性數學的方法解
        * 方法二：使用Gradient Descent(只要Loss Function可微分就能使用)
            * 任意取參數w_0, b_o代入Loss Function，並計算Loss Function對w的偏微分及oss Function對b的偏微分，各自找到更新的參數值w'及b'
            * Gradient Descent等式：<img src="http://chart.googleapis.com/chart?cht=tx&chl=L(w, b) = w{}' = w^{0} - \eta \frac{\mathrm{d}L }{\mathrm{d} w}Lw=w_0" style="border:none;" alt="依照切線斜率檢視係數應該變大或變小，並可調整參數eta來控制變化量">
            * 就是對Loss Function的w, b各自做偏微分，再代入Gradient Descent式子即可更新參數，直到找到最佳的參數(w*, b*)
    5. 將Testing Data代入所找到的Linear Function並計算估計誤差
    6. 增加Linear Function的複雜度(增加次方、增加Feature)並重新找出最佳的Linear Function，再比較估計誤差
    7. 再提升：回到第一步 Redesign Model
        * 此提供一種猜測，不同物種、時期的Linear Function會不一樣，因此可以利用Delta Function為不同物種給予不同的Linear Function參數
    8. 再提升：回到第二步加上**Regularization**
        * Loss Function中再加上 Lamda*summation(w_i)^2，把(w_i)^2也看做距離值，就能想像這是要w_i的值要取小，整體Loss值才能變小
        * 意義：Function會變Smooth，也就是找到的最佳Function本身不容易受Input Data影響  
          說明：當Input Data x_i出現變化的時候，可以看做Loss Function的x_i和外加上一個Delta x值，把估計誤差的summation部分整理一下就能看到這個Delta x也會受w_i所影響，如果今天我們找的w_i是比較小的，就表示w_i參數對Input Data的影響也會變小。即輸入對輸出不這麼敏感，所以這就稱為平滑。
        * 平滑Function的意義在於我們猜測完美Function其實有考慮我們不知道的參數、次方，但正因為我們不知道，所以會把它們視為Testing Data的「雜訊」，則當Function越平滑，則Testing Data的雜訊影響Function輸出值的影響力就越小，偏離正確值的偏移量也就越少，相對而言Function就比較準(不那麼容易失準)
        * Lamda值越大就代表Regularization的影響越大，Function就越平滑
* 下一篇：https://www.youtube.com/watch?v=D_S6y0Jm6dQ&feature=youtu.be
            

    