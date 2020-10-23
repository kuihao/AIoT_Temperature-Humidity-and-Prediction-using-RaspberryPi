# 2020/10/23 各種 Gradient Descent 及技巧
0. 定義 Gradient 的符號為 '∇'<br>
   定義 w 為 Machine Learning Model 的 Features 的全重值參數的向量<br>
   定義 L(w) 為對 Loss Function 'L' 輸入參數向量 w<br>
     則 ∇L(w) 表示對 L(w) 取 Gradient，意思是「w 對 L 趨近零取一次微分」<br>
     計算表達式為 ∇L(w) = dL/dw -> 0<br><br>
   定義 Gradient 的 Learning rate 符號為 η<br>
   定義 ^t 表示第 t 輪 Iteration<br>
   定義 x^n、y^n、∇L^n() 表示從數據集 (Data set) 抽樣序號 n 的資料，簡稱樣本 n (Simple n)

1. **Vanilla Gradient Descent:** w^(t+1) = w^t - η^t * ∇L(w^t) 
    * 最原始的方法，全靠手動調整 Learning rate
    * 提升效能方法：
        * 不同參數設置不同的 Learning rate 係數
        * Learning rate 要設為變數，隨著時間增加，Learning rate 要變小
2. **AdaGrad Gradient Descent:** w^(t+1) = w^t - (η / sqrt(SS(∇L(w^t)))) * ∇L(w^t)，其中 SS 表示「sum of squared」，先將此刻以前的 Gradient 取平方再加總。
3. **Stochastic Gradient Descent (SGD, 隨機梯度下降法)**
    * 公式：「w^(t+1) = w^t - η * **∇L^n**(w^t)」
    * 原 Loss Function 為 ( Σ_n( y^n - ( Σ( w*x^n ) + b ) )**2
        * SSE (Sum of Squared estimate of Errors) 又稱 SSR (Sum of Squared Residuals)<br> 或 RSS (Residual Sum of Squares)
    * 此 Loss Function 改為 L^n(w) = ( y^n - ( Σ( w*x^n) + b) )**2
        * **新Loss Function 只要 SR 不要 SSR**，只針對單一樣本 n 作 SR (Squared Residuals)，<br>不再將所有的樣本都加總
    * 每考慮一個 Sample 就直接計算 Gradient 並更新參數 w。相較之下，Vanilla Gradient Descent 計算完 N 個 Samples 才更新 1 次參數，但同樣的 Samples 考慮時間，Stochastic Gradient Descent 已經更新了 N 次參數，更有效率地更新參數
4. Feature Scaling
    * 就是**數值標準化**，通常不同項目的原始數據 (此指不同的 Features) 會有不同的 Data Range (也就是最大值、最小值、單位換算之不同造成數據比例尺不一)，透過標準化將不同 Data Range 的項目歸於同一個比例尺度。
    * 這樣的好處是作 Gradient Descent 時，不論從參數起始點為何，每次更新參數的幅度皆會類似，降低起始點選擇的差異。
    * 第二項好處是能提升 Gradient Descent 的效率，Feature Scaling 的 Loss value 等高線圖繪接近正圓 (以二維參數的概念繪製圖型)，由於每次更新參數都是照著等直線的法線方向前進，因此圖型越接近正圓，則每次參數的**更新的方向**會更加對準中心最低點
    * 常用的方法有 ( Mean / Standard Deviation ) 和 ( 差距 / 全距 )
5. Gradient Descent Theory (Base on *Calculus*)
    * *Taylor series 泰勒級數*
    * 如何每次都能在 Loss value 等高線圖中找到最小值作為前進方向呢？(也就是這個法線方向的 -η * Gradient 的原理)
    * 用參數值起點 (a, b, ...) 對 Loss Function 作泰勒展開，也就是對多項式作一次微分之後，高次項對 Loss 值的影響力會微乎其微，式子就能簡化為 **L(w) ~= s + u(w1-a) + v(w2-b) + ...(若有更多Features)**，其中以下皆為常數： s = L(a, b, ...)、u = dL(a,b,...)/dw1、v = dL(a,b,...)/dw2、...依此類推。<br>
    以只考慮兩個 Features 的情況來看，「小範圍」等同是等高線上，起點(a, b)為圓心的一個小圓圈，**我們希望能在起點(a, b)附近找到一個最小的Loss值，所以相當於求半徑為[u, v]構成之圓圈範圍內的某個向量η*[(w1-a), (w2-b)]**<br>
    從 L(w) ~= s + u(w1-a) + v(w2-b) 可看出使 L(w) 變小的關鍵在於其中的內積式子「u(w1-a) + v(w2-b)」，只要找到能使內積值最小的那組(w1, w2)就能使 Loss 值最小，(因為 s 是常數非變數所以忽略它， s 不影響 Loss 值變小)。<br>
    拆開 u(w1-a) + v(w2-b) 變成 [w1-a, w2-b] dot [u, v]，簡單來說就是當 [w1-a, w2-b] 與 [u, v] 反向(斜率相同、方向相反)，並再配個係數 η 給 [w1-a, w2-b] 盡量調整大至圓半徑，則就找到這組 [w1-a, w2-b]
    * 簡化上述過程最中得到「找尋小範圍內最小的 Loss值的方程式」為：<br>
      [w1, w2, ...]^T = [a, b, ...]^T - η*[u, v, ...]^T<br>
      其中<br>
        1. 為了方便計算 η 是看作 [w1, w2] 會與 [u, v] 的長度呈正比，所以把係數乘給 [u, v]<br>
        2. 把 [w1-a, w2-b] 的 [a, b] 拆出來並移項
        3. [w1, w2, ...]^T 就是新的參數值
        4. [a, b, ...]^T 就是舊的參數值
        5. η 就是 Learning rate
        6. [u, v, ...]^T 就是 Gradient ∇L([a, b])
      以上就是 Gradient Descent 的計算式由來
    * 注意! Gradient Descent 的計算式要成立，則表示前提 L(w) ~= s + u(w1-a) + v(w2-b) 要先成立，前提成立的要件是 Taylor series 成立，Taylor series 成立的要件是起始點取半徑值 d 劃圓，這個圓半徑 d 要足夠小以滿足泰勒展開的精度<br>
    由於 η 與 半徑 d 是成正比，所以 **η 要足夠小、無窮小**的時候，Gradient Descent的前提才會成立，不過實作上其實 η 不太大即可。換句話說，若 η 不夠小，則 Gradient Descent 後的 Loss 值反而會變大。
6. Gradient Descent 的限制
    1. 會停在 Local minimization
    2. 會停在 Settle point 微分值是 0 的時候
    * 實務上我們不好確定微分值極小是否是接近最佳解

