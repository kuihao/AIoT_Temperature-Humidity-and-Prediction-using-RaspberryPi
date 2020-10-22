# 2020/10/22
1. **分析差異** 多元線性回歸公式解 與 SGD (Stochastic gradient descent, 隨機梯度下降法)
2. python 矩陣乘法
3. **學習隨機梯度下降法**
## 多元線性回歸
* https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/TensorFlow/TF%20%E7%AF%84%E4%BE%8B%20II%20(Regression).md
## Matrix Multiplication in Python
1. *numpy.dot()* ~= *numpy.matmul()* = *@*
    * *numpy.dot()* 是向量內積 (inner product)
    * 內積知識補充: https://ccjou.wordpress.com/2010/01/27/%E5%85%A7%E7%A9%8D%E7%9A%84%E5%AE%9A%E7%BE%A9/
    * *numpy.matmul()* 則是正常的矩陣乘法，運算子為 *@*
    * 這兩個函式的運算結果看來無異，且 python 對此兩種函式都支援矩陣自動轉置
      就算矩陣相乘的行與列沒有對應到，函式內部也會自動轉置
2. *numpy.multply()* 
    * 單純是矩陣內相對應的元素 (Entry) 互相乘積，小心別誤用
    * 可使用運算子 \*
## 學習隨機梯度下降法
* 來源：https://www.youtube.com/watch?v=yKKNr-QKz2Q&feature=youtu.be
* 我們可以用 gradient 符號 '∇L()' 來表示參數對 Loss Function 微分組成的向量
* Loss Function 的名字就象徵這是一個找尋輸出值越小越好的 Function，若針對好的函數輸出較大的值，<br>則會稱為 Objective Function
* **小心地調整 Learning rate**，當 Learning rate 過大會容易發生 Loss 值來回震盪無法收斂；<br>
  但是 Learning rate 過小則無法有效率地找尋最佳的參數值
* **請繪製 Loss value 與 Iteration times 的二維座標圖** 以觀察 Learning rate 的變化趨勢<br>
  使調整 Learning rate 的過程更有憑據及有效率
* **程式自動調整 Learning rate**
  * 理論上隨著 Loss 值下降，Learning rate 要越來越小，使 Loss 值容易收斂
  * 令 Learning rate = η、Epoch (時代, 即 Iteration) = t，則<br>
    可訂為 **η^t = η/sqrt(t+1)**，其中 η^t 表示第 t 輪的 η 更新值
  * **針對不同的參數應設置不同的 Learning rate** 
    * **AdaGrad** 是最容易實現這個想法的方法
      * 令 w 為參數、g 為 gradient、^t 讀作「上標 t」表示 Epoch = t
      * Vanilla Gradient Descent: w^(t+1) = w^t - η^t * g^t 
      * AdaGrad Gradient Descent: w^(t+1) = w^t - **η^t / sigma^t** * g^t，注意 η^t = η / sqrt(t+1)
        * **sigma^t** 是 Epoch = t 以前所有 w 所計算的微分值的 RMS (root mean aquare)<br>
          就是 Gradient 的平方值再開根號 **sqrt( mean_t( pow( g_t, 2 ) ) )**<br>
          其中 mean_t( g_t ) 表示要加總過去 t 時間裡(含現在)每個時期的 Gradient，注意 mean 是除以 t+1
      * 將新的 Learning rate 與 AdaGrad 的想法組合可得到新的式子：<br>
        **「w^(t+1) = w^t - (η / sqrt(Σ_t(pow( g_t, 2)))) * g^t」**裡面的 t+1 剛好抵銷
      * 當函數有考慮多項參數的時候，只考慮參數與損失函數的一次微分呈正比，這是不夠的
      * 從兩組參數的 Loss value 等高線圖可以看出決定 Gradient 增加量的因素還要考慮與二次微分呈反比，因此 AdaGrad 的分母相當於是把參數 Gradient 的二次微分也考慮進來(對一次微分作 Gradient samplimg 的感覺)
     
