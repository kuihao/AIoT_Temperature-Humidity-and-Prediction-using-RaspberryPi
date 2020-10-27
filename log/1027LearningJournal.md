# 2020/10/27
## Model 提升
1.  **Redesign model**
    在 Linear regression 中，可以嘗試提升次方或是增加/刪除部分 Feature
---
    Model: y' = w1 * x + w2 * x^2 
    Loss： L(w1, w2) = Sum[(y - y_i')^2] = Sum[(y_i' - y)^2]
    Gradient:
        dL/dw1 = Sum[ (2) * (y_i' - y) * (x) ]
        dL/dw2 = Sum[ (2) * (y_i' - y) * (x^2) ]
---
2.  **加上 Regularization**
    概念：使 Function 平滑化(Smooth)，於 Loss Function 中加上特定 Feature 之權重參數的平方，也就是考慮該參數越小越好。特定參數越小就表示使特定 Feature 對整體 Function 的影響力不那麼高。這麼做有效的原因是完美的 Function 可能有我們沒考慮到的 Feature，因此平滑的 Function 可以避免 Overfitting、過度參考目前的 Feature，而能更反映真實的預測。

3.  如何分析是 Variance 還是 Bias 導致 Error?
    * (1) Bias：將所有的實驗結果**加總算平均(計算期望值)**，得到的中心點與真正靶心之間的誤差距離稱為 Bias，越簡單的 Model 難打中靶心，因為 Bias 大，反之越複雜的 Moedel 的 Bias 越小。
        * 狀況：**越簡單的 Model，雖越集中卻也越難涵蓋到靶心、Bias 越大 --> Underfitting**
        * 實例：**Training data 的 error 就很大 --> Bias 大，Underfitting**
        * **改善方向**：Redesign Model 增加複雜度使其能包含到完美 Model 以降低 Training data error
    * (2) 越複雜(越高次方)的 Model 則 **Variance** 越大(實驗的預測 Model 越發散)；反之則越集中，越接近完美 Model。(說明：最簡單的 Model 就是常數函數，不會發散意味著每次實驗都很一致。)
        * 狀況：**越複雜 Model，預測越發散、Variance 越大 --> Overfitting**
        * 實例：**Training data error 小，但 Testing data error 大 --> Variance Overfitting**
        * **改善方向**：降低 Variance，使之更加集中
            * (1) 增加訓練資料，不會傷害 Bias(準度)，但實務上難蒐集
            * (2) 加上 Regularization，但會傷害 Bias，可能會沒包含到完美 Function，使準度下降
## Classfication
1.  Input: Features
    Output: 接受或拒絕(對錯)、分類、影像分類辨識