# 機器學習筆記：錯誤率與Function改善方法
0. ref:Hung-yi Lee ,ML Lecture 2: Where does the error come from?, https://www.youtube.com/watch?v=D_S6y0Jm6dQ&feature=youtu.be
1. 定義：完美估計Function為f^hat、ML估計Function稱為Estimator用f* 表示，其中會影響f*偏離f^hat的因素有Bias及Viarance
2. 如何估計Bias及Viarance
    * 統計學中，我們的目標是找到真實的Population Mean 'Mu'及真實的Viarance 'Sigma Squre'，有兩條抽樣思路：(1)抽很多次計算期望值 (2)多抽一點 
        * 單抽一次、抽很少次的資料的Mean 'm'很難打中真實的'Mu'，但依**照常態分布**(高斯分布、鍾型分配)可以知道，大量採樣n次的平均數Means，在68%的信心水準之下(設定估計值的信賴區間在在一個標準誤差範圍內)，會有68%次的信賴區間能夠涵蓋到真實的'Mu'
        * Mean的估計
        不過此時我們先別管信心水準的問題把這件事情簡化來說，可以直接對所有採樣的Means取期望值(**對抽樣Means取平均**)，則在常態分布的分布狀態下**期望值就會是我們追求的'Mu'**<br>
        *用m_i估計的Mu是無Bias的狀況*
            * 補充：中央極限定理：樣本數 n 足夠大時，不管母體資料是什麼分布，也不管母體的資料是連續或離散、對稱或不對稱、右偏或是左偏，甚至單峰或是多峰都會有以下三種性質：<br>
                1. 隨機變數(抽樣資料) x 的分配會接近鐘形的常態分布
                2. 隨機變數 x 的平均數與原母體平均數相同都是µ
                3. 隨機變數 x 的標準差(稱為標準誤)與原母體的標準差不同，變成只有 σ
        * 多抽一點會影響Variance：'m'的散布情況:<br>Var[m]= σ^2/n<br>當抽樣的n越大時，則'm'的散布就會越集中'Mu'。*計算m的Variance是無Bias的狀況*
        * Variance的估計：<br>因為我們實際上不知道真實的σ^2，所以要用抽樣資料計算m之後，再計算抽樣Varience 's^2'，再用s^2去估計Variance。<br>然而，s^2的分布對於σ^2是有偏移的，計算(s_i)^2的期望值：<br>
        E[(s_i)^2] = (N-1)/N * σ^2<br>
        因此*這個估計算法是有Bias的*，不過n越大時，s^2的分布也會靠近σ^2
        * **小結-如何評估目前f-star對於f-hat的Error是來自Bias還是Variance？** <br>作一次抽樣可得出f*，再作大量抽樣可計算許多f*進而得出f-bar(抽樣的靶心)；比較靶心周圍的散布情形，若太散就是Variance造成的Error；若f-bar沒對上f-hat，就是Bias造成的Error
3. Linear Function的狀況
    * 越複雜的Model則散布越開，即造成Error的Variance越大
        * 越簡單的Function受到Input Data的影響就越小
    * 越簡單的Model，即造成Error的Bias就越大
        * E[f*] = f-bar，則Bias為f-bar與f-hat有多接近？
        * 不在乎散布，只算期望值(平均)
        * 決定一個Model相當於固定了Function Set Space，越複雜的Model就越有機會涵蓋到真實的f-hat
    * **當Error來自Variance過大，就稱為「Overfittimg」，當Error來自Bias過大，就稱為「Underfittimg」**
4. 如何Impove Model？先判斷Model的Error是來自於Variance或Bias
    * 當Model不夠貼合、Match於Input Data時，就是Underfitting、Bias大<br>**此時需要Redesign Model(增加Feature或提升次方)**

    * 若Training Data Error小，但是Testing Data Error大，就表示Overfitting、Variance大<br>**此時需要增加Train Data重Train，或是Regularization(於Loss Function後加上一個使參數越小越好的Term，用weight控制平滑度，但這招會傷害Bias)**
    * 於Bias及Variance皆調小，取得Error最小之最佳平衡
5. 訓練資料的分割要點
    * 若只將Data分成Training Set與Testing Set，最後得出的Function Model預測能力仍會小於真實的Testing set的預估情況
    * 因此Testing Set還要多分割一塊用來模擬未知Testing Set的情形，即分成Public Testing Set及Privite Testing Set
    * Training Set也分成兩組：Training Set及Validation Set，用Validation Set來篩選Model，再將Training Set及Validation Set都作為Training Set訓練出f*
    * N-ford Cross Validation：多分割幾次Validation Set，Train完Model後選擇平均Error最小的為f*
    * 這樣的方法是要確保Public Set的結果能夠和Private Set的結果一致，反映真實的情境


