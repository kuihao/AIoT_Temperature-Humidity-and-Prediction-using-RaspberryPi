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
    * 用參數值起點 (a, b, ...) 對 Loss Function 作泰勒展開，也就是對多項式作一次微分之後，高次項對 Loss 值的影響力會微乎其微，式子就能簡化為 **L(w) ~= s + u(w1-a) + v(w2-b) + ...(若有更多Features)**，其中以下皆為常數： s = L(a, b, ...)、u = dL(a,b,...)/dw1、v = dL(a,b,...)/dw2、...依此類推。<br><br>
    以只考慮兩個 Features 的情況來看，「小範圍」等同是等高線上，起點(a, b)為圓心的一個小圓圈，**我們希望能在起點(a, b)附近找到一個最小的Loss值，所以相當於求半徑為[u, v]構成之圓圈範圍內的某個向量η*[(w1-a), (w2-b)]**<br><br>
    從 L(w) ~= s + u(w1-a) + v(w2-b) 可看出使 L(w) 變小的關鍵在於其中的內積式子「u(w1-a) + v(w2-b)」，只要找到能使內積值最小的那組(w1, w2)就能使 Loss 值最小，(因為 s 是常數非變數所以忽略它， s 不影響 Loss 值變小)。<br><br>
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
    * 實務上我們不好確定微分值極小是否是接近最佳解，若停在 Settle point 就像是走到了平缓的高原
7. New Optimizer in Deep Learning
    * 上課影片：https://youtu.be/4pUmZ8hXlHM
    * 補充資料：https://ppt.cc/fGFg1x
    * On-line 最佳化
        * 實務上一次只會拿到一個 Feature 進行訓練 (尚未明白其意思 2020/20/23)
    * Off-line 最佳化
        * 一次取得所有 Training Datas 的 Inputs (Featured) 及 Outputs 進行訓練
        * 實務上的困難之處，資料量過大，難以擁有足夠的空間存放資料
    * SGD (Stochastic Gradient Descent)
        * 最基本方法
    * SGDM (SGD with Momentum)
        * **用途**是修正 SGD 容易卡在 Local minimization (Gradient 微分值過小趨近零) 的問題
        * 定義 Momentum (發音:/muh·men·tm/) 為過去時間點 -η * ∇L(w) 的歷史總和，以下為演算法：
            0. 令 Momentum 為 v^t
            1. 令 λ 是一個常數(也許是小數?)
            2. Loop t form 1 to iteration:
            3.   if (t = 1) then v^t = 0
            4.   else:
            5.    v^t = λ * v^(t-1) + (-η) * ∇L(w^(t-1))
            6.    w^t = w^(t-1) + v^t
        * 參數更新之公式為 **「w^t = w^(t-1) + λ * v^(t-1) + (-η) * ∇L(w^(t-1))」**
        * 意義是每次的參數調整幅度會參考過去歷史的前進幅度，當此次 gradients 趨近零時，會足夠動力繼續往前走
    * 以下為 Adaptive Learning Rate，使 Learning Rate 可以動態調整
    * AdaGrad
        * 令 ∇L(w^t) 為 g^t
        * 令 SS() 為 Summation all t of gradients' squared，先取平方再將所有梯度值加總
        * 公式為：**「w^(t+1) = w^t - (η / sqrt(SS(g^t)) * ∇L(w^t)」**
        * 意義是自動調整 Learning rate，當過去時間點的 gradients 突然出現極大反差(巨增或巨減)，則將此次的 Learning rate 調小或調大
        * sqrt(SS(g^t) 是過去 Sum of gradients' squared，當上次的 gradients 突然暴增，則此值會變大，與 Learning rate 成反比(因為放在分母)，使得此次 Learning rate 變小 
        * **缺點：** 控制 Learning rate 分母部分的 sqrt(SS(g^t)) (有人用sigama^t稱之)，觀察即可發現它是一個**單調函數 (Monotonic function)**，裡面是 Sum of gradients' square，即使外加開平方，也完全沒有縮小或減少的成分在裡面，因次可以判定它是 非遞減函數 (Nondecreasing function)<br>
        這個問題就是 **「自始 Gradient 值就是極大，則 sqrt(SS(g^t)) 就一直很大」** 那麼 Learning rate 就會一開始就變得很小很小，整個參數找尋就變得很沒效率。
    * RMSProp (Root-Mean-Square prop, 方均根傳播)
        * **改進 AdaGrad 有 Monoton 的問題**
        * 使用**指數移動平均 (exponential moving average，EMA或EXMA)**的想法<br>
        就是以指數式遞減加權的移動平均<br>
        效果是各數值的加權影響力隨時間而指數式遞減，越近期的數據加權影響力越重，但較舊的數據也給予一定的加權值
        * 公式：**「w^t = w^(t-1) - (η / sqrt(v^t)) * ∇L(w^(t-1))」**<br>
        其中 v^t 就是時間為 t 時的 EMA，其演算法為：
            0. 令 ∇L(w^(t-1)) 簡化為 g^(t-1)
            1. Loop t from 0 to iteration:
            2.   if (t = 0) then v^t = 0
            3.   else
            4.    v^t = α * v^(t-1) + (1-α) * g^(t-1)**2
        * **缺點：** RMSProp 的公式裡缺乏如同 SGDM 的歷史向前推動力，因此容易卡在 Local minima (平坦高原)
    * Adam
        * 融合上述優點以改進缺點，SGDM + RMSProp
        * 2015年提出的方法
        * 公式：**「w^t = w^(t - 1) - (η / sqrt(v^t') * ε) * m^t'」**其中<br>
            定義 m^t = β_1 * m^(t-1) + (1-β_1) * g^t <br>
            定義 m^t' = m^t / (1 - β_1^t) [註1]<br>
            定義 v^1 = (g^0)**2 <br>
            定義 v^t = β_2 * v^(t-1) + (1-β_2)(g^(t-1)**2) <br>
            定義 v^t' = v^t / (1 - β_2^t) [註1]<br>
            定義 β_1 = 0.9 <br>
            定義 β_2 = 0.999 <br>
            定義 ε = 10^(-8) ...(發音:/epsilon/) <br><br>
          註1: 這個步驟稱為 de-biasing 避免時間 t 剛開始時，gradient 太小，導致 m^t、v^t 越來越大，給除上一個小於 1 的數值就能使他們穩定。這是簡仲明助教說的，真實原因有待考證。
        * **缺點：** 若長期都是 gradient 很小的情況，即使突然 gradient 暴增，新的 gradient 的影響力其實還是不大，2018 年有的改進方法就是 AMSGrad，但是 AMSGrad 的方法中使 v^t 又加強記憶過去出現過的最大值，彷彿又回到 AdaGrad 的問題。後來 2019 年出了 AdaBound，可以處理巨增反差及巨降反差
8. Optimizer 與 大神級的 Model
    * 2018 年 Google 發表 BERT (用於語意分析、Q&A)，是由 Adam 訓練而成的
    * Transformer (用於翻譯)，是 BERT 的 decoder，由 Adam 訓練而成
    * 2017 年 Tacotron 最早的逼真語音生成的模型，由 Adam 訓練而成
    * YOLO，電腦視覺之影像偵測，早期最快的 Model，由 SGDM 訓練而成
    * Mask R-CNN，影像辨識，由 SGDM 訓練而成
    * ResNet 早期辨識影像分類，由 SGDM 訓練而成
    * Big-GAN 近年強大的影像生成 Model，由 Adam 訓練而成
    * MAMO 近年擁有快速學習能力的神經網路演算法，由 Adam 訓練而成

    * 從分析上可見 Adam 是訓練快，但預測效果較差；SGDM 是訓練慢，但預測效果較好
    * 結論而言，目前改善 Adam 較困難，但是改善 SGDM 似乎有機會
    * SGDM 本身無法動態調整 Learning rate，若能動態調整就能加速學習
        * Cyclical LR、SGDR：給週期性的 Learning rate 變化，當過於收斂時就使 Learning rate 變大，反之變小。
        * One-Cycle LR
            


