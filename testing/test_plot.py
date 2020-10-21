'''
此練習程式碼參考自: https://ithelp.ithome.com.tw/articles/10225391
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
# 繪製網格
 * [觀念] 網格看上去是表格，其實繪製方法是找出兩軸的交會點，由點來劃出格子，
    因此畫製網格其實就是畫點，要給 X軸、Y軸 兩軸的二維矩陣座標才能畫出來
 * 產生等差數列的函式
  * 產生出等差數列1: np.arange(起始<float>, 結束(不含)<float>, 公差<float>)
  * 產生出等差數列2: np.linspace(起始<int>, 結束(含)<int>, 公差<int>)
 * 分別產生 X, Y 網格： X, Y = np.meshgrid(x, y)
  * 輸入的 x, y 都要是一維向量
 * 繪圖
  * plt.plot( 輸入x軸座標, 輸入y軸座標, marker='點的樣式', color='點的顏色', linestyle='線的樣式' )
  * 輸入之座標可以是二維陣列
  * 現在 X, Y 都是二維陣列，因此畫圖時是根據兩陣列相對應的元素 (Entry) 各取為
    點的 (x, y) 座標，因此二維陣列為 [6 * 6] 就會有 36 個點了
 * 用 contourf 建立等高線圖
  * 把所有東西都組裝在一起，然後用 plt.contourf() 來繪製等高線圖
  * plt.contourf 前面的三個參數 X, Y, f(X, Y) 分別是 x 軸的網格、y 軸的網格、
    以及網格上的點的值（高度）
 * 加上等高線的數值
  * 先用 plt.contour (不是 plt.contourf)
'''
'''
# 簡單的表格點繪製
 * # x = np.array([0,1,2,3,4,5])
 * # y = np.array([0,2,4,6,8,10])
 * # Z =  np.zeros((len(y), len(x)))   # len(y) rows, len(x) cols
 * # X, Y = np.meshgrid(x, y)
 * # X = X.astype('float32')
 * # for i in range(6):
 * #     X[i][5] = 0.5
 * # print(X, '\n', Y)
 * # plt.plot(X, Y, marker="o", color="red", linestyle="none")
 * # plt.show()
'''
'''
# 彩色等高線繪圖練習 plt.contourf()
 * ## 高度函數(此為圓方程式)
 * #def f(x,y):
 * #    return ( x**2 +  y**2)
 * #
 * ## 建立網格
 * #n = 1000
 * #x = np.linspace(-50, 50, n)
 * #y = np.linspace(-50, 50, n)
 * #X,Y = np.meshgrid(x, y)
 * #
 * #
 * ## 繪製等高線圖
 * ## 第四個參數 10 決定了這張圖上要有幾層的等高線，數字越大，圖面上等高線的密度就會越高
 * ## 第五個參數 alpha 決定顏色的透明度
 * ## 第六參數 cmap，也就是 colormap，決定了等高線圖的顏色組成
 * ## 除了 jet 還有更多 colormap 的可用參數: 
 * ## https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
 * ## 上底色
 * #plt.contourf(X, Y, f(X, Y), 8, alpha=.6, cmap=plt.cm.jet)
 * ## 畫線
 * #C = plt.contour(X, Y, f(X, Y), 8, colors='black')
 * ## 等高線上數值 (label)
 * #plt.clabel(C, inline=True, fontsize=10)
 * #plt.show()
'''
'''
課堂測試數據Demo
程式碼參考：https://colab.research.google.com/drive/1l_2jQ2t6FEPwTB6G7kiIFNOLRJOFU8LF?usp=sharing#scrollTo=TOAvOQ09fffp
程式碼破譯間接參考：https://ithelp.ithome.com.tw/articles/10225858
'''
# step 1. 引入 NumPy 與 Matplotlib
# import matplotlib.pyplot as plt
# import numpy as np

# step 2. 建立數值陣列
# x_data = [n] # 就是 Model Function 的 Input data
# y_data = [n] # 就是 Model Function 的 Output data (監督式學習就是 Ground Truth、測試時的輸出預測值)
x_data = [ 338., 333., 328., 207., 226., 25., 179., 60., 208.,  606.]
y_data = [ 640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

# step 2.5. 假設模型 例如：y_data = w * x_data + b
# step 3. 建立網格資料 
# 繪製的「彩色的等高線圖就是 Loss Function Space」，兩個維度分別代表參數 w 與 b
# 以下的 x, y 並不是指訓練資料，而是 Linear Model 的兩個參數 (w, b) 的自行設定的可能範圍
# 分別建立兩個一維等差陣列 x 與 y，其範圍是 # 到 #，每隔 # 產生一個值。
x = np.arange(-200,-100,1) #bias shape:(100,) 
y = np.arange(-5,5,0.1) #weight shape:(100,) 
# 接著，我們建立了一個二維陣列 Z 來儲存每一個 X[n](bias), Y[n](weight) 位置上的「損失函數值」
# 注意陣列 Z 的存取，根據 pyplot 定義的參數，它會直接將矩陣內容繪製在畫布上，
# 因此 x, y 座標跟矩陣儲存示正好相反的 (y對應矩陣的列, x對應矩陣的行)， Z[y座標][x座標]
# 損失函數 L 為 L(a, b) = ((y_data[n] - (w*x_data[n] + b) )**2)/n
Z =  np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)
for i in range(len(x)):
    for j in range(len(y)):
        # Z 矩陣相當於整張圖，每個 Entry 存放座標 (b, w) 代入損失函數的 Loss 值
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        # 將所有訓練資料帶入計算該座標 (b, w)的 Loss 值
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] +  (y_data[n] - (w*x_data[n] + b))**2
        Z[j][i] = Z[j][i]/len(x_data)

# step 4. 決定參數
# 這裡決定 w 與 b 的起始點（從哪裡開始走）、learning rate（每一步走多遠）、
# iteration 次數（我們要走多少步）、
# 以及最後我們用兩個陣列，來分別儲存每一步我的走到的位置
# 決定 w 與 b 的起始點
w = -4
b = -120
# 決定 learning rate
lr = 0.0000001 
# 決定 iteration 的次數
iteration = 100000 
        # v2 - ada
        # b_lr = 0.0
        # w_lr = 0.0
# 儲存每一次 iterate 後的結果
w_history = [w]
b_history = [b]

# step 5. 執行梯度下降
# 執行梯度下降
for i in range(iteration):
    w_grad = 0.0
    b_grad = 0.0
    
    # 計算損失函數分別對 w 和 b 的偏微分
    for n in range(len(x_data)):
        w_grad = w_grad  - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
        b_grad = b_grad  - 2.0*(y_data[n] - b - w*x_data[n])*1.0
    # v2 - ada
    # b_lr = b_lr + b_grad**2
    # w_lr = w_lr + w_grad**2

    # 更新 w, b 位置 
    w = w - lr * w_grad
    b = b - lr * b_grad
    # v2 - ada
    # b = b - lr/np.sqrt(b_lr) * b_grad 
    # w = w - lr/np.sqrt(w_lr) * w_grad
    
    # 紀錄 w, b 的位置 
    w_history.append(w)
    b_history.append(b)

# strp 6. 繪圖
# 建立等高線圖
plt.contourf(x,y,Z, 50, alpha=0.5, cmap=plt.get_cmap('jet')) # Contour line Region = 50
# 繪製目標點(隱藏的)
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='red')
# 繪製起點
plt.plot([b_history[0]], [w_history[0]], 's', ms=12, markeredgewidth=3, color='orange')   # starting parameter
# 繪製 w,b iteration 的結果
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black') # ms = markersize, lw = linewidth 
# 繪製結束時的參數
plt.plot([b_history[-1]], [w_history[-1]], 'x', ms=12, markeredgewidth=3, color='orange')   # ending parameter
# 定義圖形範圍
plt.xlim(-200,-100)
plt.ylim(-5,5) # plt.axis([-200, -100, -5, 5])
# 繪製 x 軸與 y 軸的標籤
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
# 標上等高線數值
C = plt.contour(x, y, Z, 10, colors='black', linewidths=0.05)
plt.clabel(C, inline=True, fontsize=10)

# 畫出訓練資料與 Linear Function 的關係圖
fig = plt.figure()
paint = fig.add_subplot()
plt.plot(x_data, y_data, 'o', c='blue')
x_pos = np.linspace(0, 610, 600)
y_pos = w_history[-1] * x_pos + b_history[-1] 
plt.plot(x_pos, y_pos)

plt.show()
# print(b_history[-1], w_history[-1])

# step 7. 結果出爐
# 圖形上的每個黑點，都代表著我們的每一步。我們看到這些黑點從 b = 0 & a = 0（也就是圖形的原點）開始向 1 點鐘方向前進。根據我們對等高線的理解，這樣子切過一條又一條的等高線，不是在上升就是在下降。
# 當然這裡我們是不斷的下降高度。這裡我加上等高線的值，讓你可以更清楚的了解。
# 然而走到某一個階段之後，有個轉彎，往另外一個方向前進。雖然圖形上不是很明顯，但實際上我們仍然不斷的下降當中。
# 那麼，最後我們走到哪裡呢？這裡讓我們印出走了一萬步之後，最後一步的位置
# print(b_history[-1], w_history[-1]) （看起來其實距離目標點 1.97, 7.22 很接近了，但還是不夠接近呢）
# 也就是說，機器走到目前為止，認為 y = 7.2328846321328095 * x + 1.8477027996627977 是一條最好的線，能夠最好的「預測」油耗與里程之間的關係。這就是機器學習如何預測的過程！
# 快的示範了如何用 Python 來實作梯度下降法，但是過程中似乎還有許多問題需要討論，像是
# 
# 我是怎麼知道目標點的？
# 怎麼知道要從哪裡開始？
# learning rate 要怎麼設定？
# 要怎麼減少 iteration 的數量
# 要怎麼知道自己走到的地方低不低呢？