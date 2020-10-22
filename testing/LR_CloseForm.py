'''
作者：Kuihao Chang
此會示範如何用「公式解」計算線性回歸(二元一次及多元線性)
參考：https://medium.com/@chih.sheng.huang821/%E7%B7%9A%E6%80%A7%E5%9B%9E%E6%AD%B8-linear-regression-3a271a7453e
'''
import matplotlib.pyplot as plt
import numpy as np

'''
* 定義線性回歸方程式 
Linear Function: y_prime_i = a * x_i + b
'''
# x 為自變數，可想成輸入的數據值
x_i = [ 338., 333., 328., 207., 226., 25., 179., 60., 208.,  606.]
# y_prime 為應變數，可想成輸出的預測值，數學上 y_prime應寫作「y'」
# 符號「'」讀音「Prime」，這裡 Prime 不是「質數」的意思，而要解釋為「角分」
y_prime_i = 0 # y_prime 初始化
# y_hat 為真實數據值，可視為自然界完美 Function 的正確解答
y_i = [ 640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
'''
* Loss Function: 希望找到最小的殘差(有稱為 Error, 或稱 Residual, 或稱 Bias)的
函數參數組合，使用最小平方法(Least Square)，
又稱 LSE 或 OLS estimator(Ordinary Least Squares Estimator, 最小平方估計式)。
* Least Square: 公式的「解的目標」是找出「所有訓練樣本的誤差平方和(Sum Square Error, SSE)
最接近 0 的那組參數。」 
y_prime_i = a * x_i + b 的 SSE 公式為【 (Σ_i^n(y_i - y_prime_i) **2) / n 】(1<=i<=n, n=10)
也有人使用 MSE(Mean Squared Error, 均方誤差) 作 Loss Function
* 定義微分運算子 (微分符號): 常見的表示法為「∂(partial derivative, 偏導數)」或「d(differential, 微分)」
其實微分就是計算切線斜率。
'''
# 解二元一次的 Loss Function (此為 SSE)，只要解「參數對SSE趨近於零」的微分值即可
def SGD_Loss(a, b):
    # 先算好 mean(x_i), mean(y_i)
    mean_x_i = np.mean(x_i)
    mean_y_i = np.mean(y_i)
    # dL/da -> 0: a = Σ_i^n[(y_i - mean(y_i)) (x_i - mean(x_i))] / Σ_i^n(x_i - mean(x_i)) **2
    n = 10
    for i in range(n):
        a += (y_i[i] - mean_y_i) * (x_i[i] - mean_x_i)
    temp = 0
    for i in range(n):
        temp += (x_i[i] - mean_x_i) **2
    a /= temp
    # dL/db -> 0: b = mean(y_i) - a * mean(x_i)
    b = mean_y_i - a * mean_x_i
    payload = np.array([a, b]) # payload 只是網路程式的用詞，就是載體別想太多
    return payload
a = 0
b = 0
parameter = SGD_Loss(a, b)
a = parameter[0]
b = parameter[1]
print(  'a:', a, 
        '\n' + 'b:', b
    )
plt.plot(x_i, y_i, 'o', c='blue')
x_pos = np.linspace(0, 610, 600)
y_pos = (a * x_pos) + b
plt.plot(x_pos, y_pos)

print('Loss: ', np.sqrt(np.sum(np.power(( np.dot(x_i, a)+b ) - y_i, 2))/len(x_i)))
plt.show()
